import os
import cv2
import numpy as np

from ipb_loaders.ipb_base import IPB_Base


class WurPepper(IPB_Base):

    def __init__(self, data_source=None, path_in_dataserver='/export/datasets/Synthetic_Pepper_RGBD', overfit=False, unittesting=False, split='train'):
        super().__init__(data_source, path_in_dataserver, overfit, unittesting)

        img_dir_path = os.path.join(self.data_source, 'synthetic_image_color')
        depth_dir_path = os.path.join(self.data_source, 'synthetic_label_depth_grayscale')
        label_dir_path = os.path.join(self.data_source, 'synthetic_label_class_grayscale/synthetic_label_class_all_grayscale')

        self.images = [os.path.abspath(os.path.join(img_dir_path, f)) for f in os.listdir(img_dir_path)]
        self.dephts = [os.path.abspath(os.path.join(depth_dir_path, f)) for f in os.listdir(depth_dir_path)]
        self.labels = [os.path.abspath(os.path.join(label_dir_path, f)) for f in os.listdir(label_dir_path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        fname = self.images[index]

        img = cv2.imread(fname) / 255
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)

        depth = cv2.imread(self.dephts[index], cv2.IMREAD_GRAYSCALE) / 255
        invalid_mask = depth == 0
        depth = 1 - depth
        depth[invalid_mask] = 0
        depth = np.expand_dims(depth, 2)

        labels = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)

        item = {}
        item['image'] = np.concatenate((img, depth), axis=-1).transpose(2, 0, 1)
        item['semantic'] = labels.astype(int)
        item['filename'] = fname
        return item
