#!/usr/bin/env python3

import cv2
import csv
import yaml
import os
import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from skimage import io

from ipb_loaders.ipb_base import IPB_Base


class BUP20SweetPepper(IPB_Base):

    def __init__(
            self,
            data_source=None,
            path_in_dataserver='/export/datasets/BUP20_sweet_pepper_rgbd_dataset/post-processed/',
            bup20_gt_on=True, # if this is on, we only use the images with ground truth segmentation annotations (about 300 images for the whole bup20 dataset)
            date_sequence=None, # None:use all or "20200924"
            row_id=None, # None:use all or "row4"
            get_depth=True, # Get the depth image
            get_insseg=True, # Get the instance segmentation labels for each image
            generate_pc=False, # Generate point cloud from the rgbd images
            use_wheel_odom=True,  # if false, we use rgbd odometry
            depth_trunc=1.4, # unit: m, truncated the depth > this value when generating the point cloud
            unittesting=False):

        super().__init__(data_source, path_in_dataserver, unittesting)

        ## TODO: add split (train, val, test)

        self.with_ins_seg = get_insseg
        self.with_depth = get_depth
        self.generate_pc = generate_pc and get_depth

        if date_sequence is None and row_id is None: # use all the data
            folder_list = ['20200924/row1', '20200924/row2', '20200924/row3', '20200924/row4', '20200924/row5',
                           '20201001/row2', '20201001/row3', '20201001/row4', '20201001/row5', '20201001/row6']
            self.generate_pc = False
        elif date_sequence == 'pred': # use all the data that containing the instance predictions
            folder_list = ['20200924/row4', '20200924/row5',
                           '20201001/row4', '20201001/row5', '20201001/row6']
            self.generate_pc = False
        else: # use a specific data sequence with a sequence name and a row id
            folder_list = [os.path.join(date_sequence, row_id)]

        self.rgb_images = []
        self.depth_images = []
        self.insseg_images = []

        for sequence_folder in folder_list:
            sequence_base_path = os.path.join(self.data_source, sequence_folder)

            if bup20_gt_on: # if this is on, we only use the images with ground truth segmentation annotations
                rgb_folder_name = 'seg_rgb'
                depth_folder_name = 'seg_depth'
                seg_folder_name = 'seg'
            else: # or we use all the images
                rgb_folder_name = 'rgb'
                depth_folder_name = 'depth'
                seg_folder_name = 'pred_seg' # output by the neural network
                
            rgb_img_dir_path = os.path.join(sequence_base_path, rgb_folder_name)

            # image file paths
            self.rgb_images += [os.path.abspath(os.path.join(rgb_img_dir_path, f)) for f in os.listdir(rgb_img_dir_path)]
            
            if self.with_depth:
                depth_img_dir_path = os.path.join(sequence_base_path, depth_folder_name)
                self.depth_images += [os.path.abspath(os.path.join(depth_img_dir_path, f)) for f in os.listdir(depth_img_dir_path)]

            if self.with_ins_seg:
                insseg_img_dir_path = os.path.join(sequence_base_path, seg_folder_name) # together with predictions and annotations
                self.insseg_images += [os.path.abspath(os.path.join(insseg_img_dir_path, f)) for f in os.listdir(insseg_img_dir_path)]
            
            if self.generate_pc:
                wheel_odom_path = os.path.join(sequence_base_path, 'odometry.csv')
                rgbd_odom_path = os.path.join(sequence_base_path, 'rgbd_odom.csv')

                if use_wheel_odom:
                    self.odom_tfs = self.csv_odom_to_transforms(wheel_odom_path)
                else:
                    self.odom_tfs = self.csv_odom_to_transforms(rgbd_odom_path)

                # transformation from camera to body frame
                self.T_bc = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

                cam_param_path = os.path.join(sequence_base_path, 'params.yaml')
                with open(cam_param_path, "r") as stream:
                    try:
                        cam_param = yaml.safe_load(stream)
                        self.intrinsics = np.array(cam_param['intrinsics'])
                        self.extrinsics = np.array(cam_param['extrinsics'])
                    except yaml.YAMLError as exc:
                        print(exc)

        self.depth_trunc = depth_trunc

        # print(len(self.rgb_images))

    def csv_odom_to_transforms(self, path):
        from pyquaternion import Quaternion
        odom_tfs = {}
        with open(path, mode='r') as f:
            reader = csv.reader(f)
            # get header and change timestamp label name
            header = next(reader)
            header[0] = 'ts'
            # Convert string odometry to numpy transfor matrices
            for row in reader:
                odom = {l: row[i] for i, l in enumerate(header)}
                # Translation and rotation quaternion as numpy arrays
                trans = np.array([float(odom[l]) for l in ['tx', 'ty', 'tz']])
                quat = Quaternion(np.array([float(odom[l]) for l in ['qx', 'qy', 'qz', 'qw']]))
                rot = quat.rotation_matrix
                # Build numpy transform matrix
                odom_tf = np.eye(4)
                odom_tf[0:3, 3] = trans
                odom_tf[0:3, 0:3] = rot
                # Add transform to timestamp indexed dictionary
                odom_tfs[odom['ts']] = odom_tf

        return odom_tfs

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):

        item = {}

        rgb_fname = self.rgb_images[index]
        bgr_img = cv2.imread(rgb_fname)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # bgr to rgb # HxWx3
        rgb_img_t = rgb_img.transpose((2, 0, 1)) / 255.  # 3xHxW, convert to between 0 and 1

        fname = Path(rgb_fname).stem
        item['filename'] = fname

        if self.with_depth:
            depth_fname = self.depth_images[index]
            
            # depth_img = cv2.imread(depth_fname, cv2.IMREAD_GRAYSCALE)  # cv2 read would rescale the depth, not nice
            depth_img = io.imread(depth_fname)
            depth_img = depth_img.astype(np.float32)

            depth_img_t = depth_img / 1000.  # HxW  # to unit m
        else:
            depth_img_t = np.zeros_like(rgb_img_t[0,:,:])

        rgbd_img_t = np.concatenate((rgb_img_t, np.expand_dims(depth_img_t, axis=0)), axis=0)  # 4xHxW
        # if depth is not available, we will anyway not use it in the further training

        # 4xHxW torch tensor of the rgbd image
        item['image'] = torch.tensor(rgbd_img_t, dtype=torch.float32)

        # 2d instance segmentation
        if self.with_ins_seg:
            label_fname = self.insseg_images[index]
            label_img = cv2.imread(label_fname, cv2.IMREAD_GRAYSCALE)
            item['instance'] = torch.tensor(label_img, dtype=torch.short)
            sem_img = label_img.copy()
            sem_img[label_img>0] = 1
            item['semantic'] = torch.tensor(sem_img, dtype=torch.short)

        if self.generate_pc:
            # robot pose
            cur_pose_body = self.odom_tfs[fname]
            cur_pose_cam = np.matmul(cur_pose_body, self.T_bc)
            item['pose'] = cur_pose_cam  # camera pose

            # generate point cloud from rgbd image (not tansformed with the pose yet)
            h, w, _ = rgb_img.shape

            K_mat = self.intrinsics
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
            intrinsic_o3d.set_intrinsics(
                height=h,
                width=w,
                fx=K_mat[0, 0],
                fy=K_mat[1, 1],
                cx=K_mat[0, 2],
                cy=K_mat[1, 2],
            )

            rgb_img_o3d = o3d.geometry.Image(rgb_img)
            depth_img_o3d = o3d.geometry.Image(depth_img) # mm unit, combined with depth_scale=1000.

            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img_o3d, depth_img_o3d, \
                depth_scale=1000., depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)

            rgbd_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsic_o3d, \
                self.extrinsics, project_valid_depth_only=True)
            
            # rgbd_pcd = rgbd_pcd.transform(cur_pose_cam) # to world coordinate system

            pc_xyz = np.array(rgbd_pcd.points)
            pc_colors = np.array(rgbd_pcd.colors)
            item['points'] = pc_xyz
            item['colors'] = pc_colors
            # print(pc_xyz)

        return item