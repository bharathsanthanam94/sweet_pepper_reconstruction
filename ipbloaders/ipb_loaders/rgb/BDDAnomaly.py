import os
import numpy as np
from PIL import Image
from imageio import imread

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr
from torchvision.transforms.functional import InterpolationMode as im

from ipb_loaders.ipb_base import IPB_Base
"""
https://github.com/bdd100k/bdd100k
BDD for Semantic Segmentation:
    0:  road
    1:  sidewalk
    2:  building
    3:  wall
    4:  fence
    5:  pole
    6:  traffic light
    7:  traffic sign
    8:  vegetation
    9:  terrain
    10: sky
    11: person
    12: rider
    13: car
    14: truck
    15: bus
    16: train
    17: motorcycle
    18: bicycle

    255 is used for “unknown” category, and will not be evaluated.
"""


class BDDAnomaly(IPB_Base):
    r"""Dataloader for BDD-like dataset. Tailored to BDDAnomaly and BDD100K
    """
    def __init__(self, data_source=None, size=(360, 680), split='train', overfit=False, path_in_dataserver='/export/datasets/BDDAnomaly', unittesting=False):
        r"""
        Create BDDAnomaly object.
        Input:
        - path: absolute path to dataset root folders. Its subfolders are `images`, `semantic`, `instance`
        - size: target image size for resize
        - mode: one of `train`, `val`, `test`
        - overfit: if True, forces the val dataset to train set and returns 4 images only
        """
        super().__init__(data_source, path_in_dataserver, unittesting)

        self.path = data_source
        self.overfit = overfit
        if self.overfit:
            self.mode = 'train'
        else:
            self.mode = split

        self.transform = Transform(size[0], size[1])

        self.image_path = os.path.join(self.path, 'images', self.mode)
        self.semantic_path = os.path.join(self.path, 'semantic', self.mode)

        self.image_list = os.listdir(self.image_path)
        self.image_list.sort()
        self.semantic_list = os.listdir(self.semantic_path)
        self.semantic_list.sort()

        assert len(self.image_list) == len(self.semantic_list), "#images != #labels"
        self.len = len(self.image_list)

    def __getitem__(self, index):
        sample = {}
        image = Image.open(os.path.join(self.image_path, self.image_list[index]))
        semantic = np.asarray(imread(os.path.join(self.semantic_path, self.semantic_list[index])))

        if self.transform is not None:
            image, semantic = self.transform(image, semantic)

        # label is uint8: [0, 255]. With +1, 255 becomes 0 and all labels are +1
        semantic += 1

        sample['image'] = image
        sample['semantic'] = semantic.long()
        sample['filename'] = self.image_list[index]

        return sample

    def __len__(self):
        if self.overfit:
            return 1
        return self.len


class Transform:
    def __init__(self, target_height, target_width):
        self.resize_img = tr.Resize((target_height, target_width), interpolation=im.BILINEAR)
        self.resize_lab = tr.Resize((target_height, target_width), interpolation=im.NEAREST)
        self.tensorize = tr.ToTensor()

    def __call__(self, image, label):
        image = self.tensorize(image)
        image = self.resize_img(image)

        label = torch.from_numpy(label).unsqueeze(0)
        label = self.resize_lab(label).squeeze()

        return image, label
