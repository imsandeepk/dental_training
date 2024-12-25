
import os
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def read_mesh(filepath):
    vertices = []
    faces = []
    preps = []
    logs = []
    geometric_tokens = ['v', 'f', 'vn']
    label_tokens = ['usemtl', 's']
    prep = 0
    logfile = 'logs.txt'
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            tokens = line.strip().split()
            if (tokens[0] not in geometric_tokens) and (tokens[0] not in label_tokens):
                continue
            if tokens[0] in geometric_tokens and len(tokens) != 4:
                msg = 'InconsistentFileContent |  {} has wrong cardinality of tokens: {}'.format(filepath, tokens)
                print(msg)
                return None, None, None, [msg]
            if tokens[0] == 'usemtl' and len(tokens) != 2:
                msg = 'InconsistentLabel |  usemtl string has no suffix {}'.format(filepath)
                print(msg)
                return None, None, None, [msg]
            if tokens[0] == "v":
                x, y, z = map(float, tokens[1:])
                vertices.append([x, y, z])
            elif tokens[0] == "vn":
                continue
            elif tokens[0] == 'usemtl' and 'g' in tokens[1]:
                prep = 0
            elif tokens[0] == 'usemtl' and 'g' not in tokens[1]:
                # prep = int(tokens[1])
                if 'p' in tokens[1] or 'P' in tokens[1]:
                    prep = int(tokens[1][:-1])
                else: 
                    prep = int(tokens[1])

            
            elif tokens[0] == "f":
                face = [int(c.split('/')[0]) - 1 for c in tokens[1:]]
                preps.append(prep)
                faces.append(face)
                
    return vertices, faces, preps, logs

def toO(mesh, mode = 'o'):
    if mode != 'o':
        mesh = vedo2open3d(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def write_log(log_file_path, message):
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')


def write_seg_file(preps, seg_file_path, log_file_path):
    
    with open(seg_file_path, 'w') as seg_file:
        for prep in preps:
            seg_file.write(f"{prep}\n")
    log_message = f"Segmentations written to: {seg_file_path}"
    print(log_message)
    write_log(log_file_path, log_message)


def save_decimated_mesh(vertices, faces, output_obj_path, target_faces, log_file_path):
    
    try:
    
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
       
        original_faces = len(mesh.faces)
        if original_faces > target_faces:
            mesh = mesh.simplify_quadratic_decimation(target_faces)
        
      
        mesh.export(output_obj_path)
        log_message = (
            f"Decimated mesh saved to: {output_obj_path} | "
            f"Original faces: {original_faces}, Decimated faces: {len(mesh.faces)}"
        )
        print(log_message)
        write_log(log_file_path, log_message)
    except Exception as e:
        log_message = f"Error while decimating mesh: {output_obj_path} | {str(e)}"
        print(log_message)
        write_log(log_file_path, log_message)
                  


def process_mesh_files_with_clipping(root_folder, log_file_path, itter, bbox_size=14):
    set_folders = sorted(os.listdir(root_folder))
    
    for subfolder in set_folders:
        # subfolder = '3002'
        print(f"Processing subfolder: {subfolder}")
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            if len(os.listdir(subfolder_path)) <= 2:
                try:
                    subfolder_path = os.path.join(subfolder_path, os.listdir(subfolder_path)[1])
                except:
                    subfolder_path = os.path.join(subfolder_path, os.listdir(subfolder_path)[0])

            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.obj') and ('lowerjaw' not in file_name and 'upperjaw' not in file_name and 'contact' not in file_name) and file_name[0] != '.':
                   file_path = os.path.join(subfolder_path, file_name)
                   try :
                    # Read the mesh data
                        vertices, faces, preps, logs = read_mesh(file_path)


                    # Initialize the mesh
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(vertices)
                        mesh.triangles = o3d.utility.Vector3iVector(faces)
                        mesh = toO(mesh)
                        # Get unique labels from preps
                        unique_labels = np.unique(preps)
                   except: 
                        #  continue
                    
                        log_message = f"Skipping file {file_path} due to errors: read mesh error"
                        print(log_message)
                        write_log(log_file_path, log_message)
                        continue
                    
                   for label in unique_labels:
                    try:
                        if label == 0:  # Skip unlabeled triangles
                            continue

                        print(f"Processing label: {label}")

                        # Get the indices of the faces belonging to the current label
                        face_indices = np.where(preps == label)[0]
                        label_faces = np.array(faces)[face_indices]

                        # Get the vertex indices of these faces
                        face_vertex_indices = np.unique(label_faces.flatten())

                        # Extract the labeled vertices
                        label_vertices = np.array(vertices)[face_vertex_indices]

                        # Compute the center of the labeled vertices
                        center = np.mean(label_vertices, axis=0)

                        # Define the bounding box for clipping
                        min_bound = center - bbox_size
                        max_bound = center + bbox_size

                        # Crop the mesh using the bounding box
                        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                        clipped_mesh = mesh.crop(aabb)
                        # O.visualization.draw_geometries([clipped_mesh])
                        clipped_mesh = toO(clipped_mesh)
                        # Generate the segmentation file
                        clipped_faces = np.asarray(clipped_mesh.triangles)
                        clipped_vertices = np.asarray(clipped_mesh.vertices)

                        # Compute barycenters of full mesh faces
                        full_barycenters = np.mean(np.array(vertices)[faces], axis=1)

                        # Compute barycenters of clipped mesh faces
                        clipped_barycenters = np.mean(clipped_vertices[clipped_faces], axis=1)

                        # Build a KDTree for the full mesh barycenters
                        kdtree = KDTree(full_barycenters)

                        # Map clipped faces to original faces using KDTree
                        clipped_preps = np.zeros(len(clipped_faces), dtype=int)
                        for i, barycenter in enumerate(clipped_barycenters):
                            distance, index = kdtree.query(barycenter)
                            if distance < 1e-6 and preps[index] != 0:
                                clipped_preps[i] = 1  # Mark as labeled
                    except: 
                        log_message = f"Skipping file {file_path} due to errors."
                        print(log_message)
                        write_log(log_file_path, log_message)
                        
                        # Save the clipped mesh as STL
                    destination_folder_meshes = '/Users/vaibhavgupta/Desktop/Dental AI/data_dec/meshes2'
                    destination_folder_label = '/Users/vaibhavgupta/Desktop/Dental AI/data_dec/label2'

                    output_stl_path = os.path.join(destination_folder_meshes, f"clipped_mesh_{itter}_{label}.stl")
                    o3d.io.write_triangle_mesh(output_stl_path, clipped_mesh)

                    # Save the segmentation file
                    output_seg_path = os.path.join(destination_folder_label, f"clipped_mesh_label_{itter}_{label}.seg")
                    write_seg_file(clipped_preps, output_seg_path, log_file_path)

                    print(f"Saved clipped mesh and segmentation for label {label} to {destination_folder_meshes}")

                        
                   print('itter sub:', itter)
                   itter += 1

    return itter

# root_folder = '/Volumes/One Touch/full arch labelling/full arch labelling'


folders = [
    "/Volumes/One Touch/full arch labelling/full arch labelling",
    "/Volumes/One Touch/full arch labelling/FULL ARCH LABELLING 2128-2199",
    "/Volumes/One Touch/full arch labelling/full arch labelling 2200-2402",
    "/Volumes/One Touch/full arch labelling/full arch labelling 2403-2427",
    "/Volumes/One Touch/full arch labelling/full arch labelling 2428-2451",
    "/Volumes/One Touch/full arch labelling/full arch labelling 2452-2475",
    "/Volumes/One Touch/full arch labelling/full arch labelling 3026-3050",
    "/Volumes/One Touch/full arch labelling/full arch labelling data",
    "/Volumes/One Touch/full arch labelling/labelling",
    "/Volumes/One Touch/full arch labelling/labelling data 2702",
    "/Volumes/One Touch/full arch labelling/new 1001-1050",
    "/Volumes/One Touch/full arch labelling/new 1051-1100"
]

log_file_path = os.path.join('/Users/vaibhavgupta/Desktop/Dental AI/data_dec/dental_ai_logs', 'log.txt')


with open(log_file_path, 'w') as log_file:
    log_file.write("Processing Log\n")

itter = 0
for fold in folders: 
        # fold = '/Users/vaibhavgupta/Desktop/Dental AI/sample_Data'
        print('Starting --', fold)
        write_log(log_file_path, '--')
        write_log(log_file_path, 'Starting --' + fold)
        write_log(log_file_path, '--')
        itter = process_mesh_files_with_clipping(fold, log_file_path, itter)
        print()
        write_log(log_file_path, '--')
        write_log(log_file_path, 'Ending --' + fold)
        write_log(log_file_path, '--')
        print('Ending --', fold)
        print()
        itter += 1
        print('itter:', itter)

