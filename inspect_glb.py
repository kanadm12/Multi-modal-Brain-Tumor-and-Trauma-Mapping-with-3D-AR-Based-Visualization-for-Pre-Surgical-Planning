import json
import struct

# Read GLB file
glb_path = r"C:\Users\Kanad\Desktop\BR_PROJECT\BEPROJECT-RUNPOD-DATA\BraTS_Optimized_Solution\ui_design\brats-viewer-ui\public\demo_brain_tumor.glb"

with open(glb_path, 'rb') as f:
    # GLB Header
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    length = struct.unpack('<I', f.read(4))[0]
    print(f'GLB Magic: {magic}, Version: {version}, Length: {length}')
    
    # JSON Chunk
    chunk_length = struct.unpack('<I', f.read(4))[0]
    chunk_type = f.read(4)
    json_data = f.read(chunk_length).decode('utf-8')
    
    gltf = json.loads(json_data)
    
    print('\n=== NODES ===')
    for i, node in enumerate(gltf.get('nodes', [])):
        name = node.get('name', 'NO_NAME')
        mesh = node.get('mesh', 'N/A')
        print(f'  Node {i}: name="{name}" mesh={mesh}')
    
    print('\n=== MESHES ===')
    for i, mesh in enumerate(gltf.get('meshes', [])):
        name = mesh.get('name', 'NO_NAME')
        print(f'  Mesh {i}: name="{name}"')
    
    print('\n=== MATERIALS ===')
    for i, mat in enumerate(gltf.get('materials', [])):
        name = mat.get('name', 'NO_NAME')
        color = mat.get('pbrMetallicRoughness', {}).get('baseColorFactor', 'N/A')
        print(f'  Material {i}: name="{name}" color={color}')
    
    print('\n=== SCENES ===')
    for i, scene in enumerate(gltf.get('scenes', [])):
        nodes = scene.get('nodes', [])
        print(f'  Scene {i}: nodes={nodes}')
