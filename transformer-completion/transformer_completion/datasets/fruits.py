# import sys
# sys.path.append('/home/bharath/Desktop/thesis/code/pepper_transformer/transformer-completion/transformer_completion/datasets')
from ipb_loaders.pointcloud.igg_fruit import IGGFruit
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import numpy as np

class IGGFruitPretrain(IGGFruit):
    def __init__(self, data_source, split, sensor, overfit, precomputed_augmentation=False):
        super().__init__(
            data_source=data_source, split=split,
            sensor=sensor, overfit=overfit,
            precomputed_augmentation=precomputed_augmentation
        )
    
    @staticmethod
    def create_2nd_view(item):

        points = item['points']
        normals = item['normals']

        dim_1 = np.random.randint(0, 3)
        positive_1 = np.random.randint(0, 1)

        if positive_1:
            condition = points[:, dim_1] > 0
        else:
            condition = points[:, dim_1] < 0

        points_2nd_view = points[condition]
        normals_2nd_view = normals[condition]

        dim_2 = np.random.randint(0, 3)
        if dim_1 == dim_2:
            return points_2nd_view, normals_2nd_view

        positive_2 = np.random.randint(0, 1)
        if positive_2:
            condition = points_2nd_view[:, dim_2] > 0
        else:
            condition = points_2nd_view[:, dim_2] < 0

        points_2nd_view = points_2nd_view[condition]
        normals_2nd_view = normals_2nd_view[condition]

        return points_2nd_view, normals_2nd_view


    def __getitem__(self, index):
        item = super().__getitem__(index)

        points_2nd_view, normals_2nd_view = self.create_2nd_view(item)
        item['points_2nd_view'] = points_2nd_view
        item['normals_2nd_view'] = normals_2nd_view
        
        return item

class IGGFruitPretrainDatasetModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        precomputed_aug = self.cfg[self.cfg.MODEL.DATASET].AUGMENTATION
        self.train_dataset = IGGFruitPretrain(
            data_source=cfg.PATH,
            split='train',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT,
            precomputed_augmentation=precomputed_aug)
        self.test_dataset = IGGFruitPretrain(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'test',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT)
        self.val_dataset = IGGFruitPretrain(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'val',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT)

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
            dataset=self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.val_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.val_loader

    def test_dataloader(self):

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.test_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader

class IGGFruitDatasetModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        precomputed_aug = self.cfg[self.cfg.MODEL.DATASET].AUGMENTATION
        self.train_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT,
            precomputed_augmentation=precomputed_aug)
        self.test_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'test',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT)
        self.val_dataset = IGGFruit(
            data_source=cfg.PATH,
            split='train' if self.cfg.MODEL.OVERFIT else 'val',
            sensor=self.cfg[self.cfg.MODEL.DATASET].SENSOR,
            overfit=self.cfg.MODEL.OVERFIT)

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
            dataset=self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.val_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.val_loader

    def test_dataloader(self):

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.test_dataset.collate,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        return self.test_loader
