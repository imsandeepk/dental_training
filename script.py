from vedo import *
import numpy as np
import open3d as O
from scipy.spatial import KDTree
import pyvista as pv
from collections import deque
import math


def toT(mesh):
    M = None
    try:
        M = vedo2trimesh(open3d2vedo(mesh))
    except:
        try:
            M = vedo2trimesh(mesh)
        except:
            M = mesh
    return M

def toO(mesh, mode = 'o'):
    if mode != 'o':
        mesh = vedo2open3d(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def vis(points, mesh = None): 
    cleaned_pcd = O.geometry.PointCloud()
    cleaned_pcd.points = O.utility.Vector3dVector(points)

    # O.io.write_point_cloud('/Users/vaibhavgupta/Downloads/testing_17_nov/debug/debug_'+  srt +'.pcd', cleaned_pcd)
    if mesh: 
        O.visualization.draw_geometries([cleaned_pcd, mesh])
    else : 
        O.visualization.draw_geometries([cleaned_pcd])

def showFile(mesh_path, label_path):
    # pathh = '/Users/vaibhavgupta/Downloads/testing_17_nov/'

    lower_reso_mesh = O.io.read_triangle_mesh(mesh_path)
    lower_reso_mesh = toO(lower_reso_mesh)

    print(lower_reso_mesh.triangles)
    
    prep_seg_arr = np.loadtxt(label_path)

    prep_face_indices = np.where(prep_seg_arr != 0)[0]
    selected_triangles = np.array(lower_reso_mesh.triangles)[prep_face_indices]  

    flattened_indices = selected_triangles.flatten()

    unique_vertex_indices = np.unique(flattened_indices)


    color = np.tile([0,1, 0], (len(lower_reso_mesh.vertices), 1)) 

    for ind in unique_vertex_indices :
        
        color[ind] = [1, 0, 0]  

    color = np.array(color, dtype=np.float64)

    # Assign the vertex colors
    lower_reso_mesh.vertex_colors = O.utility.Vector3dVector(color)

    O.visualization.draw_geometries([lower_reso_mesh])

# for i in range(3,170):
#     for j in range(0,50):
#         try:
#             mesh_path = f"./training_data/meshes2/clipped_mesh_{i}_{j}.stl"
#             label_path = f"./training_data/label2/clipped_mesh_label_{i}_{j}.seg"
#             print(mesh_path)
#             showFile(mesh_path, label_path)
#         except Exception as e:
#             print(e)
#             continue
        
mesh_path = f"full_mesh_0_31.ply"
label_path = f"full_mesh_label_0_31.seg"
showFile(mesh_path, label_path)