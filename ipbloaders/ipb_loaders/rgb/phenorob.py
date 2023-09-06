import torch
import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F

from ipb_loaders.ipb_base import IPB_Base


class PhenoRob(IPB_Base):

    def __init__(self,
                 data_source=None,
                 path_in_dataserver='/export/datasets/PhenorobDataChallenge/final_patch_extract/phenorob_challenge_dataset',
                 split='train',
                 overfit=False,
                 unittesting=False):
        super().__init__(data_source, path_in_dataserver, unittesting)

        self.data_source = data_source
        self.split = split

        self.data_path = os.path.join(data_source, split)
        self.overfit = overfit

        self.image_list = [x for x in os.listdir(os.path.join(self.data_path, "images")) if ".png" in x]

        self.image_list.sort()

        self.len = len(self.image_list)

        # preload the data to memory
        self.field_list = os.listdir(self.data_path)
        self.data_frame = self.load_data(self.data_path, self.image_list, self.field_list)
        self.data_frame["image_list"] = self.image_list

    @staticmethod
    def load_data(data_path, image_list, field_list):
        data_frame = {}
        for field in tqdm(field_list):
            data_frame[field] = []
            for image in tqdm(image_list):
                image = cv2.imread(
                    os.path.join(os.path.join(data_path, field), image),
                    cv2.IMREAD_UNCHANGED,
                )
                if len(image.shape) > 2:
                    sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    sample = torch.tensor(sample).permute(2, 0, 1)
                else:
                    sample = torch.tensor(image.astype("int16"))

                data_frame[field].append(sample)
        return data_frame

    def __getitem__(self, index):
        sample = {}

        for field in self.field_list:
            sample[field] = self.data_frame[field][index]

        partial_crops = sample["semantics"] == 3
        partial_weeds = sample["semantics"] == 4

        # 1 where there's stuff to be ignored by instance segmentation, 0 elsewhere
        sample["ignore_mask"] = torch.logical_or(partial_crops, partial_weeds).bool()

        # remove partial plants
        sample["semantics"][partial_crops] = 1
        sample["semantics"][partial_weeds] = 2

        # remove instances that aren't crops or weeds
        sample["plant_instances"][sample["semantics"] == 0] = 0
        sample["leaf_instances"][sample["semantics"] == 0] = 0

        # make ids successive
        sample["plant_instances"] = torch.unique(sample["plant_instances"] + sample["semantics"] * 1e6, return_inverse=True)[1]
        sample["leaf_instances"] = torch.unique(sample["leaf_instances"] + sample["semantics"] * 1e6, return_inverse=True)[1]
        sample["leaf_instances"][sample["semantics"] == 2] = 0

        sample["image_name"] = self.data_frame["image_list"][index]

        item = {
            'semantic': sample['semantics'],
            'instance': sample['plant_instances'],
            'image': sample['images'] / 255,
            'filename': sample['image_name'],
            'extra': {
                'leaf_instances': sample['leaf_instances'],
                'leaf_visibility': sample['leaf_visibility'],
                'plant_visibility': sample['plant_visibility'],
                'ignore_mask': sample['ignore_mask']
            }
        }

        return item

    def __len__(self):
        return self.len


class PhenoRobDetection(PhenoRob):

    def __init__(self,
                 data_source=None,
                 path_in_dataserver='/export/datasets/PhenorobDataChallenge/final_patch_extract/phenorob_challenge_dataset',
                 split='train',
                 overfit=False,
                 unittesting=False):
        super().__init__(data_source, path_in_dataserver, split, overfit, unittesting)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        bbox, cls_vec, _ = self.masks_to_boxes(item)
        item = {'image': item['image'], 'boxes': bbox, 'boxes_class': cls_vec, 'filename': item['filename']}
        return item

    def masks_to_boxes(self, item):

        masks = F.one_hot(item['instance']).permute(2, 0, 1)

        # binary masks have now class_id instead of just ones
        cls_exploded = masks * item['semantic'].unsqueeze(0)
        cls_exploded = torch.reshape(cls_exploded, (cls_exploded.shape[0], cls_exploded.shape[1] * cls_exploded.shape[2]))
        # cls_vec contains the class_id for each masks
        cls_vec, _ = torch.max(cls_exploded, dim=1)

        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]
        bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area == 0)]

        return bounding_boxes, cls_vec, bounding_boxes_area
