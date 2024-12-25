import open3d as o3d
import numpy as np

def color(mesh, labels):
    """
    Colors the mesh based on the labels.
    Labels with a value of 0 will be green, and non-zero labels will be red.
    """
    labels = labels.astype(int)  # Ensure 32-bit integers
    print(f"Labels: {labels[:10]}")  # Debugging: Print the first few labels

    # Assign colors to vertices based on the labels
    num_vertices = len(mesh.vertices)
    vertex_colors = np.zeros((num_vertices, 3))  # Initialize to black

    # Map triangle labels to vertex colors
    triangle_colors = np.zeros((len(mesh.triangles), 3))
    triangle_colors[labels == 0] = [0, 1, 0]  # Green for label 0
    triangle_colors[labels != 0] = [1, 0, 0]  # Red for non-zero labels

    # Map the triangle colors to vertex colors
    for i, triangle in enumerate(mesh.triangles):
        triangle = np.array(triangle)
        vertex_colors[triangle] = triangle_colors[i]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh

def visualize(mesh, labels):
    if not mesh.has_triangles():
        print("Mesh does not have triangles. Ensure you are loading a valid triangle mesh.")
        return
    
    new_mesh = color(mesh, labels)
    new_mesh = mesh
    new_mesh = clean_mesh(new_mesh)
    o3d.visualization.draw_geometries([new_mesh])


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
# Paths to input files
label_path = "output_labels/full_mesh_label_914_31_clipped_mesh_label_31.seg"
mesh_path = "output_meshes/full_mesh_914_31_clipped_mesh_31.ply"

try:
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh:
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Load the labels
    labels = np.loadtxt(label_path)
    if len(labels.shape) != 1:
        raise ValueError("Labels file must contain one label per line.")

    # Visualize
    visualize(mesh, labels)

except FileNotFoundError as e:
    print(f"Error: {e}")

except ValueError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
