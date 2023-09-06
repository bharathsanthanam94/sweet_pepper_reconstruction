import glob
import os

import numpy as np
import pandas as pd
from trimesh import transform_points

from ipb_loaders.ipb_base import IPB_Base


class KITTIOdometry(IPB_Base):

    def __init__(
        self,
        data_source: str = None,
        path_in_dataserver: str = '/export/datasets/kitti-odometry',
        sequence: int = 0,
        unittesting=False,
        overfit=False,
        correct_scan=True,
        min_range=2.5,
        max_range=120.0,
        apply_pose=True,
    ):
        """Simple KITTI DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """

        super().__init__(data_source, path_in_dataserver, overfit, unittesting)

        # Config stuff
        self.sequence = str(int(sequence)).zfill(2)
        self.kitti_sequence_dir = os.path.join(data_source, "sequences", self.sequence)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")
        self.correct_scan = correct_scan
        self.min_range = min_range
        self.max_range = max_range
        self.apply_pose = apply_pose

        # Read stuff
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))
        self.poses = self.load_poses(os.path.join(data_source, f"poses/{self.sequence}.txt"))
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))

        # Cache
        self.use_cache = True

    def __getitem__(self, idx):
        item = {}
        item['points'] = self.scans(idx)
        item['pose'] = self.poses[idx]
        item['filename'] = self.scan_files[idx]
        return item

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(idx, self.scan_files[idx])

    def read_point_cloud(self, idx: int, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = self._correct_scan(points) if self.correct_scan else points[:, :3]
        points = points[np.linalg.norm(points, axis=1) <= self.max_range]
        points = points[np.linalg.norm(points, axis=1) >= self.min_range]
        points = transform_points(points, self.poses[idx]) if self.apply_pose else None
        return points

    @staticmethod
    def _correct_scan(scan: np.ndarray):
        """Corrects the calibration of KITTI's HDL-64 scan.

        Taken from PyLidar SLAM
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)
        return corrected_scan

    def load_poses(self, poses_file):

        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = pd.read_csv(poses_file, sep=" ", header=None).values
        n = poses.shape[0]
        poses = np.concatenate((poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1)
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict
