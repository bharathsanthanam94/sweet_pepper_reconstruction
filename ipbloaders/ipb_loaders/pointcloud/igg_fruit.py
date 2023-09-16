import os
import numpy as np
import open3d as o3d
import json
from PIL import Image
from ipb_loaders.ipb_base import IPB_Base
import torch

class IGGFruit(IPB_Base):

    def __init__(self,
                 data_source=None,
                 path_in_dataserver='/export/datasets/igg_fruit/processed/SweetPepper/',
                 overfit=False,
                 unittesting=False,
                 split='train',
                 sensor='realsense',
                 precomputed_augmentation=False):
        super().__init__(data_source, path_in_dataserver, overfit, unittesting)

        assert sensor in ['laser', 'realsense'], 'sensor {} not recognized'

        self.sensor = sensor
        self.split = split

        with open(self.data_source + '/split.json', 'r') as file:
            self.splits = json.load(file)

        self.fruit_list = self.get_file_paths()
        self.precomputed_augmentation = precomputed_augmentation
        if self.precomputed_augmentation and self.split == 'train' and self.sensor == 'laser':
            self.fruit_list += self.get_augmentation_file_path()

    def get_augmentation_file_path(self):
        aug_list = []
        aug_data_source = self.data_source + self.precomputed_augmentation
        for fid in self.splits[self.split]:
            for aug_idx in range(10):
                aug_list.append(os.path.join(aug_data_source, fid + '_0' + str(aug_idx), 'laser/fruit.ply'))
        return aug_list

    def get_file_paths(self):
        self.fragment_step = 1
        fruit_list = []
        for fid in self.splits[self.split]:
            if self.sensor == 'realsense':
                poses = np.load(os.path.join(self.data_source, fid, 'tf/tf_allposes.npz'))['arr_0']

                depth_frames = self.absolute_file_paths(fid, 'realsense/depth/')
                rgb_frames = self.absolute_file_paths(fid, 'realsense/color/')

                for idx in range(0, len(poses) - self.fragment_step, self.fragment_step):

                    fragment_dict = {}

                    fragment_dict['pose'] = poses[idx:idx + self.fragment_step]
                    fragment_dict['d'] = depth_frames[idx:idx + self.fragment_step]
                    fragment_dict['rgb'] = rgb_frames[idx:idx + self.fragment_step]

                    fruit_list.append(fragment_dict)
                
        
            else:
                poses = np.load(os.path.join(self.data_source, fid, 'tf/tf_allposes.npz'))['arr_0']

                # depth_frames = self.absolute_file_paths(fid, 'realsense/depth/')
                rgb_frames = self.absolute_file_paths(fid, 'realsense/color/')

                # for idx in range(0, len(poses) - self.fragment_step, self.fragment_step):
                for idx in range(0, 2 - self.fragment_step, self.fragment_step):


                    fragment_dict = {}

                    fragment_dict['pose'] = poses[idx:idx + self.fragment_step]
                    # fragment_dict['d'] = depth_frames[idx:idx + self.fragment_step]
                    fragment_dict['rgb'] = rgb_frames[idx:idx + self.fragment_step]
                    fragment_dict['laser_path']=os.path.join(self.data_source, fid, 'laser/fruit.ply')
                    fruit_list.append(fragment_dict)
                    

                #original implementation
                # fruit_list.append(os.path.join(self.data_source, fid, 'laser/fruit.ply'))
            

        return fruit_list

    def get_fruit_id(self, item):
        if self.sensor == 'realsense':
            fid, _ = item['rgb'][0].split(self.sensor)
        else:
            #original implementation
            # fid, _ = item.split(self.sensor)

            fid, _ = item['laser_path'].split(self.sensor)
        return fid

    def get_pcd(self, item, fruit_id):
        if self.sensor == 'laser':
            # TODO remove point before return?
            return o3d.io.read_point_cloud(item['laser_path'])
        else:
            pcd = o3d.geometry.PointCloud()
            K = self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json'))
            for rgb, d, pose in zip(item['rgb'], item['d'], item['pose']):
                frame_pcd = self.pcd_from_rgbd(rgb, d, pose, K)
                pcd += frame_pcd
            return pcd

    def absolute_file_paths(self, fruit_id, directory):
        paths = [os.path.join(self.data_source, fruit_id, directory, frame) for frame in os.listdir(os.path.join(self.data_source, fruit_id, directory))]
        paths.sort()
        return paths

    @staticmethod
    def get_gt_pcd(fruit_id):
        return o3d.io.read_point_cloud(fruit_id + 'laser/fruit.ply')

    @staticmethod
    def pcd_from_rgbd(rgb, d, pose, K):
        rgb_frame = o3d.io.read_image(rgb)
        d_frame = np.load(d)

        extrinsic = np.eye(4)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            height=d_frame.shape[0],
            width=d_frame.shape[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame),
                                                                  o3d.geometry.Image(d_frame),
                                                                  depth_scale=1000.0,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
        return pcd.transform(pose)

    @staticmethod
    def load_K(path):
        with open(path, 'r') as f:
            data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F')
        return k
    
    @staticmethod
    def load_intrinsics(intrinsics_mat):
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            height=720,
            width=1280,
            fx=intrinsics_mat[0, 0],
            fy=intrinsics_mat[1, 1],
            cx=intrinsics_mat[0, 2],
            cy=intrinsics_mat[1, 2],
        )
        return intrinsics

    @staticmethod
    def get_bounding_box(fruit_id):
        bbox = np.load(os.path.join(fruit_id, 'tf/bounding_box.npz'))['arr_0']
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])

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

    def __len__(self):
        if self.overfit:
            return 1
        return len(self.fruit_list)

    def __getitem__(self, index):
        fruit_id = self.get_fruit_id(self.fruit_list[index])
        pcd = self.get_pcd(self.fruit_list[index], fruit_id)
        # import ipdb;ipdb.set_trace()

        if self.sensor == 'realsense':
            bbox = self.get_bounding_box(fruit_id)
            pcd = pcd.crop(bbox)
        # if self.sensor == 'realsense':
        #     pcd = pcd.uniform_down_sample(every_k_points=100)

        gt = self.get_gt_pcd(fruit_id)

        item = {}
        item['points'] = np.asarray(pcd.points)

        item['extra'] = {}
        item['extra']['gt_points'] = np.asarray(gt.points)
        item['extra']['gt_normals'] = np.asarray(gt.normals)

        # Load back the saved features
        '''
        with open(fruit_id+'data.pth', 'rb') as f:
            features = torch.load(f) 
        '''
        
        
        if self.sensor == 'realsense':
            item['filename'] = self.fruit_list[index]['rgb'][0]
            item['colors'] = np.asarray(pcd.colors)
            # item['image']=np.asarray(Image.open(item['filename']))  #Extra line added to get RGB image as tensors 
            item['image']=item['filename']
            item['extrinsics']=self.fruit_list[index]['pose'].squeeze() #extra line added to get extrinsics
            intrinsics_mat=self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json')) #extra line added to get intrinsics
            item['intrinsics']=self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json')) #extra line added to get intrinsics
            # item['RGB_feats']=features["/home/bharath/Desktop/thesis/code/data/sweet_pepper_master_copy/sweet_pepper_RGBfeats_subset/p1/realsense/"+"color/"+item['image'].split("color/")[1]].to('cpu')
            

        else:
            item['filename'] = self.fruit_list[index]
            item['normals'] = np.asarray(pcd.normals)
            #added for RGB projection 
            # item['image']=np.asarray(Image.open(self.fruit_list[index]['rgb'][0]))  #Extra line added to get RGB image as tensors 
            
            item['image']=self.fruit_list[index]['rgb'][0]
            item['extrinsics']=self.fruit_list[index]['pose'].squeeze()#extra line added to get extrinsics
            # intrinsics_mat=self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json')) #extra line added to get intrinsics
            item['intrinsics']= self.load_K(os.path.join(fruit_id, 'realsense/intrinsic.json')) #extra line added to get intrinsics
            # item['RGB_feats']=features["/home/bharath/Desktop/thesis/code/data/sweet_pepper_master_copy/sweet_pepper_RGBfeats_subset/p1/realsense/"+"color/"+item['image'].split("color/")[1]].to('cpu')

        return item


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dd = IGGFruit(data_source='/data1/bsanthanam/thesis/data/sweet_pepper_RGBfeats_subset/', sensor='realsense', precomputed_augmentation=False)
    dl = DataLoader(dd, batch_size=1, collate_fn=dd.collate)
    
    for item in dl:
    
        
        for idx,item in enumerate(dl):
            
            pt = item['points'][0]
            gt = item['extra']['gt_points'][0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt)
            # o3d.visualization.draw_geometries([pcd])
            gcd = o3d.geometry.PointCloud()

            gcd.points = o3d.utility.Vector3dVector(gt)
            colors=item['colors'][0]
            pcd.colors=o3d.utility.Vector3dVector(colors)
            gcd.paint_uniform_color([.5,.7,.5])
            # o3d.visualization.draw_geometries([pcd,gcd])
            o3d.io.write_point_cloud("/data1/bsanthanam/thesis/pepper_transformer/ipbloaders/ipb_loaders/pointcloud/gt.ply",pcd)
            o3d.io.write_point_cloud("/data1/bsanthanam/thesis/pepper_transformer/ipbloaders/ipb_loaders/pointcloud/pcd.ply",gcd)
            import ipdb;ipdb.set_trace()
        



        

        # print('batch size: ', len(item['points']))
    print("total datapoints: ",count)
