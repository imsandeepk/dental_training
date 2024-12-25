import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

mesh_path = "full_mesh_0_31.ply"

mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
mesh.remove_unreferenced_vertices()


center = mesh.get_center()

bbox = mesh.get_axis_aligned_bounding_box()
bbox.color = (1, 0, 0)

o3d.visualization.draw_geometries([mesh, bbox])