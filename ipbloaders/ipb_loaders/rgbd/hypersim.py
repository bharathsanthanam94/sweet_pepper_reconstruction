import h5py
import torch
import os
import numpy as np

from ipb_loaders.ipb_base import IPB_Base


class HyperSim(IPB_Base):

    def __init__(self,
                 data_source=None,
                 split='train',
                 scenes=[],
                 output_trafo=None,
                 output_size=400,
                 path_in_dataserver='/export/datasets/HyperSim/data/',
                 unittesting=False):
        super().__init__(data_source, path_in_dataserver, unittesting)
        """
        Parameters
        ----------
        root : str, path to the ML-Hypersim folder
        mode : str, option ['train','val]
        """
        self._output_size = output_size
        self._mode = split

        self._load(data_source, split)

        self._output_trafo = output_trafo

    def __getitem__(self, index):

        with h5py.File(self.images[index], 'r') as f:
            img = np.array(f['dataset'])
        img[img > 1] = 1
        img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)  # C H W

        with h5py.File(self.depth[index], 'r') as f:
            depth = np.array(f['dataset'])
        depth = torch.from_numpy(depth).type(torch.float32)[None, :, :]  # C H W
        depth[depth.isnan()] = 0

        with h5py.File(self.semantic[index], 'r') as f:
            label = np.array(f['dataset'])
        semantic = torch.from_numpy(label).type(torch.float32)[None, :, :]  # C H W

        with h5py.File(self.instance[index], 'r') as f:
            label = np.array(f['dataset'])
        instance = torch.from_numpy(label).type(torch.float32)[None, :, :]  # C H W

        label[label > 0] = label[label > 0] - 1

        rgbd = torch.cat((img, depth), dim=0)

        semantic[semantic == -1] = 255
        instance[instance == -1] = 255

        sample = {}
        sample['image'] = rgbd
        sample['semantic'] = semantic.type(torch.int64)[0, :, :]
        sample['instance'] = instance.type(torch.int64)[0, :, :]
        sample['filename'] = self.images[index]

        return sample

    def __len__(self):
        return self.length

    def _load(self):
        self.images = []
        self.depth = []
        self.semantic = []
        self.instance = []

        path = os.path.join(self.data_source, 'images/')
        listdir = os.listdir(path)
        for scan in listdir:
            scene_path = os.path.join(path, scan + '/images')
            for scene in os.listdir(scene_path):
                if 'hdf5' in scene:
                    files = os.listdir(os.path.join(scene_path, scene))
                    for file in files:
                        filepath = os.path.join(scene_path, scene, file)
                        if 'depth_meters.hdf5' in file:
                            self.depth.append(filepath)
                        elif 'semantic.hdf5' in file:
                            self.semantic.append(filepath)
                        elif 'semantic_instance.hdf5' in file:
                            self.instance.append(filepath)
                        elif 'color.hdf5' in file:
                            self.images.append(filepath)
                        else:
                            continue

        self.images.sort()
        self.depth.sort()
        self.semantic.sort()
        self.instance.sort()
        self.length = len(self.images)

    @staticmethod
    def get_classes():
        base = os.path.dirname(__file__)
        scenes = np.load(os.path.join(base, 'cfg/scenes.npy')).tolist()
        sceneTypes = list(set(scenes))
        sceneTypes.sort()
        return sceneTypes