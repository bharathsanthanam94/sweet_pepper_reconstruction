import torch
import torchvision
import os
from PIL import Image, ImageFile
import numpy as np
import torch.nn.functional as F
from ipb_loaders.ipb_base import IPB_Base
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
""" BOSCH DATASET """


class BoschSugarBeets(IPB_Base):
    def __init__(self,
                 data_source=None,
                 split='train',
                 overfit=False,
                 path_in_dataserver='/export/datasets/bosch',
                 unittesting=False):
        super().__init__(data_source, path_in_dataserver, unittesting)

        self.datapath = data_source
        self.mode = split
        self.overfit = overfit
        self.path_in_dataserver = path_in_dataserver

        if self.overfit:
            self.datapath += '/images/train'
        else:
            self.datapath += '/images/' + self.mode

        self.all_imgs = [os.path.join(self.datapath, x) for x in os.listdir(self.datapath) if ".png" in x]
        self.all_imgs.sort()

        global_annotations_path = os.path.join(self.datapath.replace('images', 'annos'), 'global')
        parts_annotations_path = os.path.join(self.datapath.replace('images', 'annos'), 'parts')
        self.global_instance_list = [os.path.join(global_annotations_path, x) for x in os.listdir(global_annotations_path) if ".semantic" in x]
        self.parts_instance_list = [os.path.join(parts_annotations_path, x) for x in os.listdir(parts_annotations_path) if ".semantic" in x]
        self.global_instance_list.sort()
        self.parts_instance_list.sort()

        self.transform = torchvision.transforms.ToTensor()

        self.len = len(self.all_imgs)

    def get_centers(self, mask):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

        masks = F.one_hot(mask.long())
        masks = masks.permute(2, 0, 1)[1:, :, :]
        num, H, W = masks.shape

        center_mask = torch.zeros((H, W), device=masks.device, dtype=torch.float)

        for submask in masks:
            if submask.sum() == 0:
                continue
            x, y = torch.where(submask != 0)
            xy = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
            mu, _ = torch.median(xy, dim=1, keepdim=True)
            center_idx = torch.argmin(torch.sum(torch.abs(xy - mu), dim=0))
            center = xy[:, center_idx]
            center_mask[center[0], center[1]] = 1.

        return center_mask

    def get_offsets(self, mask, centers):
        if mask.sum() == 0:
            return torch.zeros((0, 4), device=mask.device, dtype=torch.float)

        masks = F.one_hot(mask.long())
        masks = masks.permute(2, 0, 1)[1:, :, :]
        num, H, W = masks.shape

        total_mask = torch.zeros((H, W, 2), device=masks.device, dtype=torch.float)

        for submask in masks:
            coords = torch.ones((H, W, 2))
            tmp = torch.ones((H, W, 2))
            tmp[:, :, 1] = torch.cumsum(coords[:, :, 0], 0) - 1
            tmp[:, :, 0] = torch.cumsum(coords[:, :, 1], 1) - 1

            current_center = torch.where(submask * centers)

            offset_mask = (tmp - torch.tensor([current_center[1], current_center[0]])) * submask.unsqueeze(2)
            total_mask += offset_mask

        return total_mask

    def __getitem__(self, index):

        if self.mode == 'pt':
            img_loc = os.path.join(self.datapath, self.all_imgs[index])
            img = Image.open(img_loc).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor

        # TRAINING MODE
        # load image
        sample = {}

        image = Image.open(self.all_imgs[index])
        width, height = image.size
        sample['image'] = self.transform(image).squeeze()

        global_annos = np.fromfile(self.global_instance_list[index], dtype=np.uint32)
        global_annos = global_annos.reshape(height, width)

        parts_annos = np.fromfile(self.parts_instance_list[index], dtype=np.uint32)
        parts_annos = parts_annos.reshape(height, width)

        global_instances = global_annos >> 16  # get upper 16-bits

        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        global_instance_ids = np.unique(global_instances)[1:]  # no background
        global_instances_successive = np.zeros_like(global_instances)
        for idx, id_ in enumerate(global_instance_ids):
            instance_mask = global_instances == id_
            global_instances_successive[instance_mask] = idx + 1
        global_instances = global_instances_successive

        assert np.max(global_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

        parts_instances = parts_annos >> 16  # get upper 16-bits
        # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
        parts_instance_ids = np.unique(parts_instances)[1:]  # no background
        parts_instances_successive = np.zeros_like(parts_instances)
        for idx, id_ in enumerate(parts_instance_ids):
            instance_mask = parts_instances == id_
            parts_instances_successive[instance_mask] = idx + 1
        parts_instances = parts_instances_successive

        assert np.max(parts_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'

        global_instances = np.uint8(global_instances)
        parts_instances = np.uint8(parts_instances)

        sample['global_instances'] = torch.from_numpy(global_instances)
        sample['parts_instances'] = torch.from_numpy(parts_instances)
        semantic = sample['global_instances'].bool().long()
        sample['global_centers'] = self.get_centers(sample['global_instances'])
        center_masks = F.one_hot((sample['global_centers'] * sample['global_instances']).long())[:, :, 1:]
        center_masks = torchvision.transforms.GaussianBlur(11, 5.0)(center_masks.permute(2, 0, 1).unsqueeze(0).float()).squeeze()
        sample['global_offsets'] = self.get_offsets(sample['global_instances'], sample['global_centers']).permute(2, 0, 1)
        sample['global_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic

        sample['parts_centers'] = self.get_centers(sample['parts_instances'])
        sample['parts_offsets'] = self.get_offsets(sample['parts_instances'], sample['parts_centers']).permute(2, 0, 1)
        center_masks = F.one_hot((sample['parts_centers'] * sample['parts_instances']).long())[:, :, 1:]
        center_masks = torchvision.transforms.GaussianBlur(7, 3.0)(center_masks.permute(2, 0, 1).unsqueeze(0).float()).squeeze()
        sample['parts_centers'] = (torch.max(center_masks, dim=0)[0] / torch.max(center_masks)) * semantic

        sample['loss_masking'] = torch.ones(sample['global_instances'].shape)  # 1 where loss has to be computed, 0 elsewhere

        final_sample = {}
        final_sample['image'] = sample['image']
        final_sample['semantic'] = sample['global_instances'].bool().long()
        final_sample['instance'] = sample['global_instances'].long()
        final_sample['extra'] = {}
        final_sample['extra']['leaf_instance'] = sample['parts_instances'].long()
        final_sample['extra']['loss_masking'] = sample['loss_masking']
        final_sample['extra']['plant_centers'] = sample['global_centers']
        final_sample['extra']['leaf_centers'] = sample['parts_centers']
        final_sample['extra']['plant_offsets'] = sample['global_offsets']
        final_sample['extra']['leaf_offsets'] = sample['parts_offsets']

        del sample
        return final_sample

    def __len__(self):
        if self.overfit:
            return self.overfit
        return self.len
