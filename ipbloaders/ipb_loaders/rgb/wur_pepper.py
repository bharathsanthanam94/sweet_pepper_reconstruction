import os
import cv2
import numpy as np

from ipb_loaders.ipb_base import IPB_Base


class WurPepper(IPB_Base):

    def __init__(self, data_source=None, path_in_dataserver='/export/datasets/Synthetic_Pepper', overfit=False, unittesting=False, split='train'):
        super().__init__(data_source, path_in_dataserver, overfit, unittesting)

        img_dir_path = os.path.join(self.data_source, 'empirical_image_color')
        label_dir_path = os.path.join(self.data_source, 'empirical_label_class_grayscale/empirical_label_class_all_grayscale')

        self.images = [os.path.abspath(os.path.join(img_dir_path, f)) for f in os.listdir(img_dir_path)]
        self.labels = [os.path.abspath(os.path.join(label_dir_path, f)) for f in os.listdir(label_dir_path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        fname = self.images[index]

        img = cv2.imread(fname) / 255

        labels = cv2.imread(self.labels[index], cv2.IMREAD_GRAYSCALE)

        item = {}
        item['image'] = img.transpose(2, 0, 1)
        item['semantic'] = labels.astype(int)
        item['filename'] = fname
        return item
