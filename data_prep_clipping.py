
import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from vedo import *
import logging

# Set up logging to write to a file
logging.basicConfig(filename='processing_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


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


def crop_mesh(mesh, center, size=10):
    min_bound = center - size
    max_bound = center + size
    try:
        print("Cropping mesh")
        mesh = toO(mesh)
        context = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
        print("Cropped mesh")
    except:
        context = vedo2open3d(mesh).crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))
    context.compute_triangle_normals()
    context.compute_vertex_normals() 
    context.remove_duplicated_vertices()
    context.remove_duplicated_triangles()
    return context

def get_context(mesh,center,context_size=16):
    ctx=crop_mesh(mesh, center,context_size)
    return ctx

def clean_mesh(mesh):
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.remove_unreferenced_vertices()
    return mesh

def get_center(mesh,label,idx):
    label_idx = [i for i,v in enumerate(label) if v == idx]
    tri = np.array(mesh.cells)[label_idx]
    vertices = mesh.points()
    tooth = o3d.geometry.TriangleMesh()
    tooth.triangles = o3d.utility.Vector3iVector(tri)
    tooth.vertices = o3d.utility.Vector3dVector(vertices)
    tooth = clean_mesh(tooth)
    return tooth.get_center()
    
def map_labels(small, original, labels):
    small = open3d2vedo(small)
    # Get triangle centers for both meshes
    small_tri_centers = small.cell_centers
    original_tri_centers = original.cell_centers

    # Build a KDTree from the original triangle centers
    tree = KDTree(original_tri_centers)

    # Map each triangle in the small mesh to the nearest triangle in the original mesh
    small_labels = []
    for center in small_tri_centers:
        _, idx = tree.query(center)
        small_labels.append(labels[idx])

    return small_labels


def read_mesh(mesh_path, label_path,tooth_no):
    lower_reso_mesh = load(mesh_path)
    labels = np.loadtxt(label_path)
    center = get_center(lower_reso_mesh,labels,tooth_no)
    context = get_context(lower_reso_mesh,center)
    new_labels = map_labels(context,lower_reso_mesh,labels)
    return context, new_labels

def color_mesh_by_triangle_labels(mesh, labels):
    # Compute required properties for Open3D mesh
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # Get triangles and vertices
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Initialize vertex colors
    vertex_colors = np.zeros((len(vertices), 3))  # Default: black
    triangle_counts = np.zeros(len(vertices))  # To track vertex usage

    # Define colors
    green = [0, 1, 0]  # Green for label 0
    red = [1, 0, 0]    # Red for other labels

    # Assign colors to triangles based on their labels
    for tri_idx, label in enumerate(labels):
        color = green if label == 0 else red
        for vert_idx in triangles[tri_idx]:
            vertex_colors[vert_idx] += color
            triangle_counts[vert_idx] += 1

    # Average the vertex colors
    vertex_colors = np.divide(vertex_colors, triangle_counts[:, None], where=triangle_counts[:, None] != 0)

    # Set vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh

output_mesh = 'output_meshes/'
output_labels = 'output_labels/'
os.makedirs(output_mesh, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)


seg_path = 'label3'
mesh_path = 'meshes3'
# List all files in the folder
mesh_files = os.listdir("meshes3")
seg_files = os.listdir("label3")

# Separate files by extension
seg_files = sorted([f for f in os.listdir(seg_path) if f.endswith(".seg")])
ply_files = sorted([f for f in os.listdir(mesh_path) if f.endswith(".ply")])

for seg_file in seg_files:
    # Extract the base name (everything except 'full_mesh_label_' and the '.seg' extension)
    base_name = seg_file.replace("full_mesh_label_", "").replace(".seg", "")
    
    # Find the corresponding .ply file
    matching_ply_file = f"full_mesh_{base_name}.ply"
    
    if matching_ply_file in ply_files:
        seg_path = "label3/"+seg_file
        ply_path = "meshes3/"+matching_ply_file
        print(f"Successfully loaded pair: {seg_file} and {matching_ply_file}")
        label_file = np.loadtxt("label3/"+seg_file)
        for i in range(11,15):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        for i in range(16,20):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        for i in range(21,27,-1):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        for i in range(31,35):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        for i in range(36,40):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        for i in range(47,42,-1):
            if i in label_file:
                try:
                    mesh,label = read_mesh(ply_path,seg_path,i)
                    old_name_mesh = matching_ply_file.replace(".ply", "")
                    old_name_label = seg_file.replace(".seg","")
                    print(f"Writing mesh {old_name_mesh}_clipped_mesh_{i}.ply")
                    o3d.io.write_triangle_mesh(f"{output_mesh}{old_name_mesh}_clipped_mesh_{i}.ply", mesh)
                    np.savetxt(f"{output_labels}{old_name_label}_clipped_mesh_label_{i}.seg", label)
                    break
                except Exception as e:
                    logging.error(f"Error processing tooth {i}: {e}")
                    continue
        



    
    






    













