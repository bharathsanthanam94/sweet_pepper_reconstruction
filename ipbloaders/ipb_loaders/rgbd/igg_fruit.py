#!/usr/bin/env python3

import cv2 as cv
import json
import os
import numpy as np
import open3d as o3d

from ipb_loaders.ipb_base import IPB_Base


class IGGFruit(IPB_Base):

    def __init__(self, data_source=None, path_in_dataserver='/export/datasets/igg_fruit/processed/SweetPepper/p1', get_color=False, unittesting=False):

        super().__init__(data_source, path_in_dataserver, unittesting)
        self.data_source = data_source

        # loading output of realsense registration
        registration_outputs = self.get_registration_outputs()
        self.Ks = registration_outputs['K']
        self.poses = registration_outputs['pose']
        self.bboxs = registration_outputs['bbox']

        self.files = self.get_instance_filenames()
        self.get_color = get_color

    def get_registration_outputs(self):
        """ 
        Load registration parameters
        
        Args:
      
        Returns: 
            data: dictionary containing intrinsic, boudning box 
                  and poses of each frame of each fruit
        """

        k_path = os.path.join(self.data_source, 'realsense/intrinsic.json')
        pose_path = os.path.join(self.data_source, 'tf/tf_allposes.npz')
        bbox_path = os.path.join(self.data_source, 'tf/bounding_box.npz')

        k = self.load_K(k_path)
        bbox = np.load(bbox_path)['arr_0']
        pose = np.load(pose_path)['arr_0']

        return {'K': k, 'bbox': self.bbox2dict(bbox), 'pose': pose}

    def get_instance_filenames(self):
        """ 
        Load image names
        
        Returns: 
            files: list of file names ( of rgb frames) to be used for training/testing
        """

        files = []
        max_frame_id = len(self.poses)
        lst_images = os.listdir(self.data_source + '/realsense/color/')
        lst_images.sort()
        for fname in lst_images:
            frame_id = int(fname[:-4]) - 1
            if frame_id == max_frame_id:
                break
            files.append(self.data_source + '/realsense/color/' + fname)

        files.sort()
        return files

    @staticmethod
    def load_K(path):
        """ 
        Load intrinsic params
        
        Args:
            path: path to json
                        
        Returns: 
            k: intrinsic matrix
        """
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F')
        return k

    @staticmethod
    def bbox2dict(bb):
        """
        Convert bounding box from array with shapes 2x3 to dictionary

        Args:
            bb: bounding box extreme points

        Returns:
            box: bounding box as dictionary
        """

        x_min, y_min, z_min = bb[0]
        x_max, y_max, z_max = bb[1]

        box = {}
        box['xmin'] = x_min
        box['xmax'] = x_max
        box['ymin'] = y_min
        box['ymax'] = y_max
        box['zmin'] = z_min
        box['zmax'] = z_max
        return box

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # ugly stuff to get consistent ids
        # realsense self.idxs start with 1
        frame_id = int(self.files[idx].split('/')[-1][:-4]) - 1

        bgr = cv.imread(self.files[idx])
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

        depth_path = self.files[idx].replace('color', 'depth')
        depth = np.load(depth_path.replace('png', 'npy'))

        rgb = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image(depth)

        pose = self.poses[frame_id]
        invpose = np.linalg.inv(pose)

        h, w, _ = bgr.shape

        K = self.Ks
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            height=h,
            width=w,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1000, depth_trunc=1.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, invpose, project_valid_depth_only=True)

        xyz = np.array(pcd.points)
        colors = np.array(pcd.colors)
        sample = {'points': xyz, 'colors': colors, 'pose': pose}
        return sample
