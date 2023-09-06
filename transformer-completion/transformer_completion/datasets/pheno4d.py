from ipb_loaders.pointcloud.pheno4d import Pheno4D
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d

class Pheno4DLeaf(Pheno4D):
    def __init__(self, data_source, overfit):
        super().__init__(data_source=data_source, overfit=overfit)

    def __getitem__(self, index):
        leaf_item = {}
        leaf_item['extra'] = {}

        item = super().__getitem__(index)
        leaf_ids = np.unique(item['instance'])
        leaf_id = np.random.choice(leaf_ids[1:])
        
        if self.overfit:
            leaf_id = 3

        xyz = item['points'][item['instance'] == leaf_id]
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.translate(-pcd.get_center())
        pcd.scale(0.001,center=np.array([0,0,0]))
        pcd.translate(np.asarray([0,0,0.005]))

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))

        pcd.orient_normals_to_align_with_direction()                                                  

        leaf_item['points'] = np.asarray(pcd.points)
        leaf_item['normals'] = np.asarray(pcd.normals)
        leaf_item['filename'] = item['filename']
        leaf_item['extra']['gt_points'] =  np.asarray(pcd.points)
        leaf_item['extra']['gt_normals'] =  np.asarray(pcd.normals)

        return leaf_item



class Pheno4DLeafDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = Pheno4DLeaf(data_source='/media/federico/federico_hdd/Pheno4D', overfit=cfg.MODEL.OVERFIT)
        self.test_dataset = Pheno4DLeaf(data_source='/media/federico/federico_hdd/Pheno4D', overfit=cfg.MODEL.OVERFIT)
        self.val_dataset = Pheno4DLeaf(data_source='/media/federico/federico_hdd/Pheno4D', overfit=cfg.MODEL.OVERFIT)

    def train_dataloader(self):

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=self.train_dataset.collate,
            shuffle=False,
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