'''
Modified RGB projection pipeline using open3d

'''
'''
1. Extract features from RGB Image
2. Get the template mesh and project the image to mesh
3. Replace the pixel colors (RGB) with the features extracted from ResNet
'''
import time
import os
import copy
import json
import torch
import trimesh
import ipdb
import open3d as o3d
from PIL import Image
import ipdb
from torch import nn
import numpy as np
from .mod_resnet_feature import extract_features
# from .resnet_feature import extract_features
class RGBfeatureprojection(nn.Module):
        def __init__(self,resnet_layer):
            super().__init__()
            self.resnetlayer=resnet_layer

       

        def forward(self,rgb_filename,vertices_mesh,faces_mesh,cam_extrinsics,intrinsics_mat):
              
              
              vertices= vertices_mesh.cpu().numpy().squeeze(0).astype(np.float32)
              faces=faces_mesh.cpu().numpy().squeeze(0).astype(np.uint32)
              # vertices,faces=self.get_mesh_attr(vertices,faces,cam_extrinsics)

              scene=o3d.t.geometry.RaycastingScene()
              geom_id=scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
              rays = scene.create_rays_pinhole(intrinsic_matrix=o3d.core.Tensor(intrinsics_mat),
                                 extrinsic_matrix=o3d.core.Tensor(np.linalg.inv(cam_extrinsics)),
                                 width_px=1280,
                                 height_px=720)
              #perform RGB projection
              ans =scene.cast_rays(rays)
              
              tri_ids=ans['primitive_ids'].numpy()
              mask_inf=tri_ids== 4294967295
              tri_ids[mask_inf]=0
              vert_ids=faces[tri_ids]
              vert_ids[mask_inf]=[0,0,0]
              device=torch.device("cuda")
              vert_ids = vert_ids.astype(np.int64)
              vert_ids_tensor=torch.tensor(vert_ids).to(device)
              # vertex_features=0.5*np.ones((2562,512),dtype=np.uint8)
              vertex_features = 0.5 * torch.ones((2562, 512), dtype=torch.uint8).cuda()

              
              
              # image=o3d.io.read_image(rgb_filename)
              
              image_PIL = Image.open(rgb_filename)
              
              resnet_features = extract_features("resnet50", image_PIL)
              
              resnet_features=resnet_features.squeeze(0)
              image_array=torch.repeat_interleave(torch.repeat_interleave(resnet_features,8,dim=1),8,dim=2)
              
              image_array=image_array.permute(1,2,0)
              
              
              
              
            

              # image_array=np.asarray(image)

              #Extract image features from ResNet50
             
            
              # Assign vertex colors based on vert_ids
              vertex_features[vert_ids_tensor[:, :, 0]] = image_array
              vertex_features[vert_ids_tensor[:, :, 1]] = image_array
              vertex_features[vert_ids_tensor[:, :, 2]] = image_array
              
            
              # features_tensor=torch.from_numpy(vertex_features).unsqueeze(0)
              
              return vertex_features.unsqueeze(0)
              
              '''
              proj_mesh = o3d.geometry.TriangleMesh()
              proj_mesh.vertices = o3d.utility.Vector3dVector(vertices)
              proj_mesh.triangles = o3d.utility.Vector3iVector(faces)
              proj_mesh.vertex_colors= o3d.utility.Vector3dVector(vertex_colors/255)
              o3d.io.write_triangle_mesh("/data1/bsanthanam/thesis/pepper_transformer/transformer-completion/transformer_completion/models/proj_mesh.ply",proj_mesh)
              ipdb.set_trace()
              '''
              








      
if __name__ =="__main__":
      fruit_path="/home/bharath/Desktop/thesis/code/data/sweet_pepper_master_copy/sunburned_pepper_2/p21b/"
      # load RGB and depth 
      n_images=1
      depth_files = sorted(os.listdir(fruit_path + "/realsense/depth"))[0:n_images]
      d_frame = np.load(fruit_path + "/realsense/depth/" + depth_files[0])
      rgb_files = sorted(os.listdir(fruit_path + "/realsense/color"))[0:n_images]
      print(rgb_files)
      rgb_frame = o3d.io.read_image(fruit_path + "/realsense/color/" + rgb_files[0])
      
      #consider uniform depth frame
      # modify the depth map
      d_mod = np.ones(d_frame.shape)
      d_mod = 100* d_mod
      d_mod = d_mod.astype(np.float32)


      #Create template mesh
      ico = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.04)
      template = ico.subdivide_loop(number_of_iterations=4)
      coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    #   o3d.visualization.draw_geometries([template,coord_frame])
      pcl_mesh = o3d.io.read_point_cloud(fruit_path + "/laser/fruit.ply")
      fruit_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcl_mesh, depth=9
    )
      vertices = torch.from_numpy(np.asarray(template.vertices)).cuda().unsqueeze(0).float()
      faces = torch.from_numpy(np.asarray(template.triangles)).cuda().unsqueeze(0)

      #Load intrinsics
      in_path=fruit_path + "/realsense/intrinsic.json"
      with open(in_path, "r") as f:
        data = json.load(f)["intrinsic_matrix"]
        K = np.reshape(data, (3, 3), order="F")

        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            height=d_frame.shape[0],
            width=d_frame.shape[1],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        print("intrinsics_matrix:",data)

        #load extrinsics
        extrinsics=np.load(os.path.join(fruit_path, "tf/tf_allposes.npz"))["arr_0"][0]
        # import ipdb;ipdb.set_trace()
        feature_extractor=RGBfeatureprojection("layer3")
        pcl=feature_extractor(rgb_frame,vertices,faces,extrinsics,intrinsics)
        # pcl.export("trial.ply")


      