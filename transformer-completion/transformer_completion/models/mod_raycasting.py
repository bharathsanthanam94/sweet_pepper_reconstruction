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
# import trimesh
import ipdb
import open3d as o3d
from PIL import Image
import ipdb
from torch import nn
import numpy as np
from .mod_resnet_feature import extract_features
# from .resnet_feature import extract_features
class FpnFeatureProjection(nn.Module):
        def __init__(self,resnet_layer):
            super().__init__()
            self.resnetlayer=resnet_layer

       

        def forward(self,rgb_filename,vertices_mesh,faces_mesh,cam_extrinsics,intrinsics_mat,image_features):
              
              
              vertices= vertices_mesh.cpu().detach().numpy().squeeze(0).astype(np.float32)
              faces=faces_mesh.cpu().numpy().squeeze(0).astype(np.uint32)
              
              #create a raycasting scene and add the mesh
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
              #error in previous implementation
              # vertex_features = 0.5 * torch.ones((2562, 512), dtype=torch.uint8).cuda()
              vertex_features = torch.zeros((2562, 256), dtype=torch.float32).cuda()

              
              
              # image=o3d.io.read_image(rgb_filename)
              
            #   image_PIL = Image.open(rgb_filename)
            #   # resnet_features = extract_features("resnet50", image_PIL)
            #   resnet_features = extract_features(resnet, image_PIL)
            #   ipdb.set_trace()
              resnet_features=image_features.squeeze(0)
              image_array=torch.repeat_interleave(torch.repeat_interleave(resnet_features,8,dim=1),8,dim=2)
              image_array=image_array.permute(1,2,0)
              
              

              # image_array=np.asarray(image)

              #Extract image features from ResNet50
             
            
              # Assign vertex colors based on vert_ids
              vertex_features[vert_ids_tensor[:, :, 0]] = image_array
              vertex_features[vert_ids_tensor[:, :, 1]] = image_array
              vertex_features[vert_ids_tensor[:, :, 2]] = image_array
              
              attn_mask= torch.zeros((2562,2562),dtype=torch.float32)
              zero_rows=torch.all(vertex_features==0,dim=1)
              attn_mask[zero_rows]=1

              attn_mask=attn_mask.cuda()
              # features_tensor=torch.from_numpy(vertex_features).unsqueeze(0)
              # import ipdb;ipdb.set_trace()
              # vertex_features.to(dtype=torch.float16)
              return attn_mask, vertex_features.unsqueeze(0)
              
            
              







