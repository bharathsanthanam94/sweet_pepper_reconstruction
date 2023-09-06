from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import copy
import torch
import numpy as np
import open3d as o3d

def visualize_o3d(shape_list, name="Open3d", show_frame=False, non_blocking=False):
    # o3d.visualization.draw_geometries(shape_list, mesh_show_back_face=True, mesh_show_wireframe=True, window_name=name)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in shape_list:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    if show_frame:
        opt.show_coordinate_frame = True
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    # opt.window_name=name
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    if not non_blocking:
        viewer.run()
        viewer.destroy_window()
    else:
        return viewer


def visualize_labels(pcd_with_labels, label_field):
    import matplotlib.pyplot as plt
    labels = pcd_with_labels.point[label_field].numpy()
    labels = np.unique(labels, return_inverse=True)[1]
    ground_mask = labels == 0
    # labels[labels == 0] = labels.max() + 1
    print("unique", np.unique(labels))
    normalized_labels = (labels - labels.min()) / (
        labels.max() - labels.min() if labels.max() - labels.min() > 0 else 1
    )
    colors = plt.get_cmap("tab20")(normalized_labels)[:, :3]
    colors[ground_mask] = np.array((0.3, 0.3, 0.3))
    viz_pcd = pcd_with_labels.to_legacy()
    viz_pcd.colors = o3d.utility.Vector3dVector(colors)
    visualize_o3d([viz_pcd])


def o3d_to_torch(tensor):
    return torch.utils.dlpack.from_dlpack(tensor.to_dlpack())

def torch_to_o3d(tensor):
    return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

class SBUB3D(Dataset):

    def __init__(self, data_source, split, N_points=1e4):
        self.split = split
        self.data_source = data_source
        self.N_points = int(N_points)
        self.leaf_list = self.get_leaves()

    def __len__(self):
        return len(self.leaf_list)

    def __getitem__(self, index):
        item = {}
        item['extra'] = {}
        item['filename'] = self.leaf_list[index]['filename']
        item['points'] = self.leaf_list[index]['points']
        item['colors'] = self.leaf_list[index]['colors']
        item['extra']['gt_points'] = self.leaf_list[index]['extra']['gt_points']

        return item

    def get_leaves(self):
        leaves = {}
        count = 0

        device_id = torch.cuda.current_device()
        device = o3d.core.Device("CUDA:{}".format(device_id))
        for pcd_name in os.listdir(os.path.join(self.data_source, 'nadir_view', self.split)):
            fname = os.path.join(self.data_source, 'nadir_view', self.split, pcd_name)
            pcd_in = o3d.t.io.read_point_cloud(fname)
            # t_labels_in = o3d.core.Tensor)pcd_in.point['leaf_ids'][:,0], device=device)

            labels_in = pcd_in.point['leaf_ids'][:,0]
            pcd_in_labelled = pcd_in.select_by_mask(labels_in >= 0)
            # to torch tensor
            labels_in = o3d_to_torch(pcd_in_labelled.point['leaf_ids'][:,0])

            pcd_gt = o3d.t.io.read_point_cloud(os.path.join(self.data_source, 'multi_view', self.split, pcd_name))
            labels_gt = pcd_gt.point['leaf_ids'][:,0]
            pcd_gt_labelled = pcd_gt.select_by_mask(labels_gt >= 0)
            labels_gt = o3d_to_torch(pcd_gt_labelled.point['leaf_ids'][:,0])

            # import ipdb;ipdb.set_trace( )

            for label in torch.unique(labels_in):
                indices_in = torch.argwhere(labels_in==label)
                indices_gt = torch.argwhere(labels_gt==label)
                leaf_pcd_in = pcd_in_labelled.select_by_index(indices_in[:,0].numpy()).to_legacy()
                leaf_pcd_gt = pcd_gt_labelled.select_by_index(indices_gt[:,0].numpy()).to_legacy()
                if len(np.asarray(leaf_pcd_in.points)) <= 50:
                    continue

                idx = np.random.choice(len(indices_in), size=self.N_points)
                leaf_pcd_in = leaf_pcd_in.select_by_index(idx)
                leaf_pcd_in = leaf_pcd_in.remove_duplicated_points()

                leaf_center = leaf_pcd_in.get_center()
                leaf_pcd_in.translate(-leaf_center)
                leaf_pcd_gt.translate(-leaf_center)

                leaves[count] = {
                    'filename': fname,
                    'points': np.array(leaf_pcd_in.points),
                    'colors': np.array(leaf_pcd_in.colors),
                    'extra': {
                        'gt_points': np.array(leaf_pcd_gt.points),
                        'gt_colors': np.array(leaf_pcd_gt.colors)
                        }
                }
                count += 1

        return leaves

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

class SBUB3DDatasetModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = SBUB3D(data_source=cfg.PATH, split='train')
        self.test_dataset = SBUB3D(data_source=cfg.PATH, split='test')
        self.val_dataset = SBUB3D(data_source=cfg.PATH, split='val')

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

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dd = SBUB3D(data_source='/media/federico/federico_hdd/shape_completion/Leaf/SBUB3D_plants/', split='test')
    dl = DataLoader(dd, batch_size=1, collate_fn=dd.collate)
    for item in dl:
        batch_size = len(item['points'])
        for b_idx in range(batch_size):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(item['points'][0])
            if len(item['points'][0]) <= 50:
                print(item['filename'][0])
            if len(item['extra']['gt_points'][0]) == 0:
                import ipdb;ipdb.set_trace()
            
            # if len(pcd.points) < 50:
            #     full_pcd = o3d.io.read_point_cloud(item['filename'][0])
            #     pcd_t = o3d.t.geometry.PointCloud().from_legacy(full_pcd)
            #     pcd_t.point['labels'] = np.array(full_pcd.normals)[:,0][:,None]
            #     import ipdb;ipdb.set_trace()
            # print(pcd.get_center())
            print(len(pcd.points))
            # o3d.visualization.draw_geometries([pcd],  mesh_show_back_face=True, mesh_show_wireframe=True)

