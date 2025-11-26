# import os
import torch
import trimesh
# print("Current working directory:", os.getcwd())


print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))



from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='C:/Users/BinaryBandit/Code/Projects/HUNYUAN3D-2/assets/demo.png')[0]


mesh.export('output_mesh.glb')

# # Initialize the texture synthesis pipeline
texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

# Convert if necessary
if not isinstance(mesh, trimesh.Trimesh):
    # Convert to a trimesh object if it's not already
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

# Get the number of faces in the original mesh
original_face_count = len(mesh.faces)

# Set the target number of faces
target_face_count = 50000


# Keep simplifying the mesh in steps until it reaches the target face count
while len(mesh.faces) > target_face_count:
    # Calculate reduction factor to bring face count closer to target
    reduction_factor = target_face_count / len(mesh.faces)
    
    # Simplify the mesh using the calculated reduction factor
    mesh = mesh.simplify_quadric_decimation(reduction_factor)

    # Print the current face count to see progress
    print(f"Current face count: {len(mesh.faces)}")


# # Generate texture for the mesh
textured_mesh = texture_pipeline(mesh, image='C:/Users/BinaryBandit/Code/Projects/HUNYUAN3D-2/assets/demo.png')

# # Save the textured mesh
textured_mesh.export('output_textured_mesh.glb')

