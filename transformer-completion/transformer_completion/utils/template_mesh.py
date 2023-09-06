import open3d as o3d
import torch
import numpy as np


class TemplateMesh():

    def __init__(self, type, dimensions):
        self.type = type
        self.dim = dimensions
        self.vertices, self.faces = self.create_3D_template_mesh(
        ) if self.dim == 3 else self.create_2D_template_mesh()

    def get_vertices_faces(self):
        return self.vertices, self.faces

    def create_3D_template_mesh(self):

        if self.type == 'sphere':
            template = o3d.geometry.TriangleMesh.create_sphere(radius=0.05,
                                                               resolution=50)

        elif self.type == 'ico':
            ico = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.04)
            template = ico.subdivide_loop(number_of_iterations=4)
        else:
            assert False, 'type should be sphere or ico'

        vertices = torch.from_numpy(np.asarray(
            template.vertices)).cuda().unsqueeze(0).float()
        faces = torch.from_numpy(np.asarray(
            template.triangles)).cuda().unsqueeze(0)

        return vertices, faces

    def create_2D_template_mesh(self):

        if self.type == 'sphere':
            template = o3d.geometry.TriangleMesh.create_sphere(radius=0.01,
                                                               resolution=50)

        elif self.type == 'ico':
            ico = o3d.geometry.TriangleMesh.create_icosahedron(radius=0.03)
            template = ico.subdivide_loop(number_of_iterations=4)
        else:
            assert False, 'type should be sphere or ico'

        box_coord = [[1, 1, 1], [-1, -1, -0.01]]
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=box_coord[1],
                                                  max_bound=box_coord[0])
        template = template.crop(box)

        vertices = torch.from_numpy(np.asarray(
            template.vertices)).cuda().unsqueeze(0).float()
        faces = torch.from_numpy(np.asarray(
            template.triangles)).cuda().unsqueeze(0)

        return vertices, faces
