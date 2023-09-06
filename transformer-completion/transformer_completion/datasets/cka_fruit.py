from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import copy
import numpy as np
import open3d as o3d

class CKAFruit(Dataset):

    def __init__(self, data_source, N_points=1e5):
        self.data_source = data_source
        self.fruit_list = self.get_fruits()
        self.N_points = int(N_points)

    def __len__(self):
        return len(self.fruit_list)

    def __getitem__(self, index):
        # pcd = pcd.uniform_down_sample(every_k_points=100)

        item = {}
        item['filename'] = self.fruit_list[index]['filename']
        item['points'] = self.fruit_list[index]['points']
        item['colors'] = self.fruit_list[index]['colors']
        # item['normals'] = np.asarray(pcd.normals)

        return item

    def get_fruits(self):
        fruit_list = {}
        count = 0
        for date in os.listdir(self.data_source):
            for row in os.listdir(os.path.join(self.data_source,date)):
                row_path = os.path.join(self.data_source,date, row)
                if os.path.isfile(row_path): continue
                row_pcd = o3d.io.read_point_cloud(os.path.join(self.data_source,date,row,'before/metashape/scaled_pointcloud.ply'))
                for fruit in os.listdir(os.path.join(self.data_source,date,row, 'fruits_measured')):
                    fruit_path = os.path.join(self.data_source,date,row, 'fruits_measured', fruit)
                    if os.path.isfile(fruit_path): continue                    
                    bb_path = os.path.join(fruit_path, 'tf/bounding_box.npz')
                    tf_path = os.path.join(fruit_path, 'tf/tf.npz')
                    if os.path.isfile(bb_path) and os.path.isfile(tf_path):
                        bb = np.load(bb_path)['arr_0']
                        tf = np.load(tf_path)['arr_0']

                        tf_pcd = copy.deepcopy(row_pcd)
                        bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bb[0], max_bound=bb[1])
                        bb.color = [1,0,0]
                        tf_pcd.transform(np.linalg.inv(tf))
                        fruit_pcd = tf_pcd.crop(bb)
                        # o3d.visualization.draw_geometries([fruit_pcd,bb],  mesh_show_back_face=True, mesh_show_wireframe=True)

                        fruit_list[count] = {
                            'filename': date + '_' + row + '_' + fruit,
                            'points': np.array(fruit_pcd.points),
                            'colors': np.array(fruit_pcd.colors)
                        }

        return fruit_list

    def get_pcd(self, path):
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.translate(-mesh.get_center()) # TODO replace with ground truth poses
        pcd = mesh.sample_points_uniformly(self.N_points)
        return pcd

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

class CKAFruitDatasetModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = CKAFruit(data_source=cfg.PATH)

    def train_dataloader(self):

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate,
            shuffle=not self.cfg.MODEL.OVERFIT,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.train_loader

    def val_dataloader(self):

        self.val_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.val_loader

    def test_dataloader(self):

        self.test_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dd = CKAFruit(data_source='/media/federico/federico_hdd/pan2023iros/data')
    dl = DataLoader(dd, batch_size=1, collate_fn=dd.collate)
    for item in dl:
        batch_size = len(item['points'])
        for b_idx in range(batch_size):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(item['points'][0])
            print(pcd.get_center())
            # o3d.visualization.draw_geometries([pcd],  mesh_show_back_face=True, mesh_show_wireframe=True)



