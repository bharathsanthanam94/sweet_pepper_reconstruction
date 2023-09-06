import os
import numpy as np

from ipb_loaders.ipb_base import IPB_Base


class Pheno4D(IPB_Base):

    def __init__(self, data_source=None, path_in_dataserver='/export/datasets/Pheno4D', overfit=False, species='Tomato', split='train', unittesting=False):
        super().__init__(data_source, path_in_dataserver, overfit, unittesting)
        assert species in ['Tomato', 'Maize'], 'species {} not recognized'.format(species)
        self.split = split
        self.species = species
        self.plant_ids = self.get_plant_ids()
        self.files = self.get_filenames()
        self.files.sort()
        self.num_semantic_classe = self.get_num_semantic_classes()

    def get_filenames(self):
        files = []
        for plant_id in self.plant_ids:
            for date in os.listdir(os.path.join(self.data_source, plant_id)):
                if self.is_annotated(date):
                    file_path = os.path.join(self.data_source, plant_id, date)
                    files.append(file_path)
        return files

    def get_plant_ids(self):
        plant_ids = []
        for id in os.listdir(self.data_source):
            if self.species in id:
                plant_ids.append(id)
        plant_ids.sort()
        return plant_ids

    def get_num_semantic_classes(self):
        if self.species == 'Tomato':
            return 3
        elif self.species == 'Maize':
            return 2
        else:
            assert False, 'species {} not recognized'.format(self.species)

    @staticmethod
    def is_annotated(fname):
        if '_a.' in fname:
            return True
        return False

    @staticmethod
    def visualize_labels(points, labels):

        import open3d as o3d
        import matplotlib.pyplot as plt

        labels = np.unique(labels, return_inverse=True)[1]
        ground_mask = labels == 0
        normalized_labels = (labels - labels.min()) / (labels.max() - labels.min() if labels.max() - labels.min() > 0 else 1)
        colors = plt.get_cmap("tab20")(normalized_labels)[:, :3]
        colors[ground_mask] = np.array((0.3, 0.3, 0.3))
        viz_pcd = o3d.geometry.PointCloud()
        viz_pcd.points = o3d.utility.Vector3dVector(points)
        viz_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([viz_pcd])

    def get_labels(self, data):
        if self.species == 'Tomato':
            instance = data[:, -1]
            semantic = np.zeros_like(instance)
            semantic[instance == 1] = 1
            semantic[instance > 1] = 2
            return semantic, instance
        elif self.species == 'Maize':
            leaf_tip_instance = data[:, -1]
            leaf_collar_instance = data[:, -2]
            semantic = np.zeros_like(leaf_tip_instance)
            semantic[leaf_tip_instance > 0] = 1
            return semantic, leaf_tip_instance, leaf_collar_instance
        else:
            assert False, 'species {} not recognized'.format(self.species)

    @staticmethod
    def collate(batch):
        item = {}
        for key in batch[0].keys():
            if key == 'extra':
                item[key] = {}
                for extra_key in batch[0][key].keys():
                    item[key][extra_key] = []
            else:
                item[key] = []

        for key in item.keys():
            for data in batch:
                if key == 'extra':
                    for extra_key in batch[0][key].keys():
                        item[key][extra_key].append(data[key][extra_key])
                else:
                    item[key].append(data[key])

        return item

    def __len__(self):
        if self.overfit:
            return 1
        return len(self.files)

    def __getitem__(self, index):

        raw_data = np.loadtxt(self.files[index])
        points = raw_data[:, :3]

        item = {}
        item['points'] = points
        if self.species == 'Tomato':
            semantic, instance = self.get_labels(raw_data)
            item['semantic'] = semantic.astype(int)
            item['instance'] = instance.astype(int)
        elif self.species == 'Maize':
            semantic, instance, leaf_collar_instance, = self.get_labels(raw_data)
            item['semantic'] = semantic.astype(int)
            item['instance'] = instance.astype(int)
            item['extra'] = {}
            item['extra']['leaf_collar_instance'] = leaf_collar_instance.astype(int)
        else:
            assert False, 'species {} not recognized'.format(self.species)

        item['filename'] = self.files[index]
        return item