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
import open3d as o3d
from torch import nn
import numpy as np
from .resnet_feature import extract_features
class RGBfeatureprojection(nn.Module):
        def __init__(self,resnet_layer):
            super().__init__()
            self.resnetlayer=resnet_layer
            
            

        def create_rays(self,trimesh_pcl):
              '''
              Create ray directions [origin and raydir]
              input: trimesh_pcl
              output: raydir[n_points x3], rayorigin [0,0,0][n_pointsx3]
              '''

             # Extract x, y, z coordinates
              x = trimesh_pcl.vertices[:,0]  
              y = trimesh_pcl.vertices[:,1]
              z = trimesh_pcl.vertices[:,2]

              # Create ray directions array
              n_raydir = np.vstack((x, y, z)).T 

              # Create ray origins array
              n_origin = np.zeros((len(x), 3))
              return n_raydir,n_origin
        
        def perform_ray_projection(self,trimesh_mesh,n_raydir,n_origin):
            #  ray = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh_mesh)
            #   # 3D loc of mesh where the ray hits, index of ray that hits the mesh and index of triangle
            #  locations, index_ray, index_tri = ray.intersects_location(
            #       n_origin, n_raydir, multiple_hits=False
            #   )

              #try this:
              # ray = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh_mesh)
              # 3D loc of mesh where the ray hits, index of ray that hits the mesh and index of triangle
             locations, index_ray, index_tri = trimesh_mesh.ray.intersects_location(
                  n_origin, n_raydir, multiple_hits=False
              )

             return locations,index_ray,index_tri
        
        def get_trimesh_object(self,vertices,faces,cam_extrinsics):
             
             #convert to numpy and CPU, trimesh wont support for now
             vertices=vertices.cpu().numpy().squeeze()
             faces=faces.cpu().numpy().squeeze()

             #TODO: keep it consistent in trimesh object , instead of open3D
             template_mesh = o3d.geometry.TriangleMesh()
             template_mesh.vertices = o3d.utility.Vector3dVector(vertices)
             template_mesh.triangles = o3d.utility.Vector3iVector(faces)
             transformed_temp=copy.deepcopy(template_mesh)
             #this is required to keep mesh and RGB in same frame before projecting
             transformed_temp.transform(np.linalg.inv(cam_extrinsics))
             vertices_trimesh = np.asarray(transformed_temp.vertices)
             faces_trimesh = np.asarray(transformed_temp.triangles)
              #convert the numpy information to trimesh object
             trimesh_mesh= trimesh.Trimesh(vertices=vertices_trimesh,faces=faces_trimesh,process=False)
             return trimesh_mesh
        
        def process_rgb(self,rgb,depth,cam_intrinsics):
             #create Point clouds
              rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),
                o3d.geometry.Image(depth),  # d_mod for equal depth
                depth_scale=1000.0,
                depth_trunc=1.0,
                convert_rgb_to_intensity=False,
            )
              #create pcl from rgbd
              pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsics)
              # o3d.io.write_point_cloud("pcl_open3d.ply", pcl)
              points = np.asarray(pcl.points)
              colors = np.asarray(pcl.colors)
              trimesh_pcl = trimesh.PointCloud(points, colors)
              return trimesh_pcl
        
        def read_data(self,rgb_filename,intrinsics_mat):
             
             rgb=o3d.io.read_image(rgb_filename)
             intrinsics = o3d.camera.PinholeCameraIntrinsic()
             intrinsics.set_intrinsics(
                  height=720,
                  width=1280,
                  fx=intrinsics_mat[0, 0],
                  fy=intrinsics_mat[1, 1],
                  cx=intrinsics_mat[0, 2],
                  cy=intrinsics_mat[1, 2],
              )
             return rgb,intrinsics


        def forward(self,rgb_filename,vertices,faces,cam_extrinsics,intrinsics_mat):
              
              rgb,cam_intrinsics=self.read_data(rgb_filename,intrinsics_mat)
              #convert vertices and faces to trimesh object
              # start_time=time.time()
              trimesh_mesh= self.get_trimesh_object(vertices,faces,cam_extrinsics)
              
              depth=100*np.ones((720,1280)).astype(np.float32)
              #Get RGB information as pcl to keep it in 3D [A way around before projection]
              trimesh_pcl=self.process_rgb(rgb,depth,cam_intrinsics)
              
  
              #Get rays from the RGB
              n_raydir,n_origin = self.create_rays(trimesh_pcl)
              
              #perform ray projection
              locations,index_ray,index_tri=self.perform_ray_projection(trimesh_mesh,n_raydir,n_origin)
            

              
              # Get face vertices using index_triis
              face_verts = trimesh_mesh.faces[index_tri]

              # Find closest vertex on each face
              min_dists = np.inf * np.ones(len(locations))
              min_verts = np.zeros(len(locations), dtype=int)
              # start_closevert=time.time()
              for i, face in enumerate(face_verts):
                  verts = trimesh_mesh.vertices[face]
                  # Compute distance to each vertex
                  dists = np.linalg.norm(verts - locations[i], axis=1)

                  # Take closest vertex
                  min_dists[i] = np.min(dists)
                  min_verts[i] = face[np.argmin(dists)]
              # end_closevert=time.time()
              # print("time for finding closes vertex:",end_closevert-start_closevert)
              # mesh.visual.vertex_colors = [255] * 3 * len(mesh.vertices)
              # Now assign each vertex to the hitting ray's RGB color

              # start_assign_color=time.time()
              for i in range(len(index_ray)):
                  color = trimesh_pcl.visual.vertex_colors[index_ray[i]]
                  # color.append(255)
                  trimesh_mesh.visual.vertex_colors[min_verts[i]] = color
              # end_assign_color=time.time()
              # print("time to assign colors: ",end_assign_color-start_assign_color)
              # trimesh_mesh.export("projected_mesh_fromlowres.ply")

              # trimesh.exchange.obj.export_obj(mesh, include_color=True)
              # min_verts now contains indices of closest vertex
              # on intersection face for each ray
              #Extract features from Resnet:
              img_array=np.asarray(rgb)
              # start_resnet=time.time()
              resnet_features = extract_features('resnet50',img_array)
              # end_resnet=time.time()
              # print("Time for ResNet features:",end_resnet-start_resnet)
              resnet_features=resnet_features.squeeze(0)
              pixel2features=torch.repeat_interleave(torch.repeat_interleave(resnet_features,8,dim=1),8,dim=2)
              
              
              #Find the index where no ray falls
              missing_index=set(range(trimesh_mesh.vertices.shape[0]))-set(face_verts.flatten())
              missing_idx_list=list(missing_index)
              #Initialize array to store features of mesh vertices
              features= np.zeros((trimesh_mesh.vertices.shape[0],pixel2features.shape[0]))

              #shrink the dimension of pixel2features
              pixel2features=pixel2features.permute(1,2,0)
              pix2feat_array=pixel2features.numpy()
              pix2feat_array=pix2feat_array.reshape(-1,512)

              #use only the features whose pixels when used as ray hits the mesh
              valid_features=pix2feat_array[index_ray]
              
              #append the valid_features to the mesh vertices
              features[min_verts] = valid_features

              features_tensor=torch.from_numpy(features).cuda().unsqueeze(0)
              # end_time=time.time()
              # print("time for entire forward method:",end_time-start_time)
              # import ipdb;ipdb.set_trace()
              return features_tensor


              # return trimesh_pcl






      
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


      