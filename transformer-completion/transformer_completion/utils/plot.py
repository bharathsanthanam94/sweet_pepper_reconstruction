import numpy as np
import open3d as o3d
from matplotlib import cm, colors

norm = colors.Normalize(vmin=0, vmax=1)
colormap = cm.ScalarMappable(norm=norm)


def plot_instances(points, ins_pred, save=False, n=0):
    inst_colors = generate_inst_colors()
    for pts, pred in zip(points, ins_pred):
        ids = np.unique(pred)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        colors = np.zeros_like(pts)
        for i in ids:
            colors[pred == i] = inst_colors[i]
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        if save:
            o3d.io.write_point_cloud("val_pred/ins/" + str(n).zfill(6) + ".ply", pcd)
        else:
            o3d.visualization.draw_geometries([pcd])


def plot_semantics(points, sem_pred, color_map, save=False, n=0):
    for pts, pred in zip(points, sem_pred):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        colors = np.array([color_map[lbl.item()][::-1] for lbl in pred]) / 255
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        if save:
            o3d.io.write_point_cloud("val_pred/sem/" + str(n).zfill(6) + ".ply", pcd)
        else:
            o3d.visualization.draw_geometries([pcd])


def generate_inst_colors():
    max_inst_id = 100000
    inst_colors = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    inst_colors[0] = np.full((3), 0.9)
    return inst_colors
