import os
import cv2
import torch

from ipb_loaders.ipb_base import IPB_Base


class Dummy(IPB_Base):
    def __init__(self, data_source=None, path_in_dataserver='/home/federico/NOBACKUP/datachallenge_trial/loader_test/', split='train', unittesting=False):
        super().__init__(data_source, path_in_dataserver, unittesting)
        self.split = split
        self.images = os.listdir(os.path.join(self.data_source, split + '/images'))
        self.annotations = os.listdir(os.path.join(self.data_source, split + '/annotations'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        annotation_name = self.annotations[idx]

        # loading image
        image = cv2.imread(
            os.path.join(os.path.join(self.data_source, self.split + '/images'), image_name),
            cv2.IMREAD_UNCHANGED,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # loading annotation
        annotation = cv2.imread(
            os.path.join(os.path.join(self.data_source, self.split + '/annotations'), annotation_name),
            cv2.IMREAD_UNCHANGED,
        )

        sample = {
            'image': torch.tensor((image / 255).astype("float")).permute(2, 0, 1),
            'semantic': torch.tensor(annotation.astype("int16")),
            'filename': image_name,
        }

        return sample