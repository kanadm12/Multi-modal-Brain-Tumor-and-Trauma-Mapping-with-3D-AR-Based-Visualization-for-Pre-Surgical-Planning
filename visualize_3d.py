# =============================================================================
# 3D MESH GENERATION MODULE
# 
# Converts segmentation masks to 3D meshes for visualization
# Uses Marching Cubes algorithm for surface extraction
# Exports to Three.js compatible format and GLTF for AR
# =============================================================================

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, generate_binary_structure

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not installed. Install with: pip install scikit-image")


# =============================================================================
# COLOR CONFIGURATION
# =============================================================================

# Tumor class colors (RGBA format, values 0-1)
TUMOR_COLORS = {
    1: {"name": "NCR", "label": "Necrotic Core", "color": [0.55, 0.0, 0.0, 1.0], "hex": "#8B0000"},      # Dark Red
    2: {"name": "ED", "label": "Edema", "color": [1.0, 0.84, 0.0, 0.7], "hex": "#FFD700"},              # Yellow (semi-transparent)
    3: {"name": "ET", "label": "Enhancing Tumor", "color": [1.0, 0.0, 0.0, 1.0], "hex": "#FF0000"},     # Bright Red
}

BRAIN_COLOR = {"color": [0.9, 0.9, 0.95, 0.15], "hex": "#E5E5F2"}  # Light blue-gray, very transparent


# =============================================================================
# MESH GENERATOR CLASS
# =============================================================================

class MeshGenerator:
    """Generate 3D meshes from segmentation masks"""
    
    def __init__(self, 
                 smoothing_iterations: int = 2,
                 step_size: int = 2,
                 decimate_ratio: float = 0.3):
        """
        Initialize mesh generator.
        
        Args:
            smoothing_iterations: Number of Gaussian smoothing iterations
            step_size: Step size for marching cubes (higher = less detail, faster)
            decimate_ratio: Target ratio for mesh decimation (lower = fewer triangles)
        """
        self.smoothing_iterations = smoothing_iterations
        self.step_size = step_size
        self.decimate_ratio = decimate_ratio
        
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for mesh generation")
    
    def generate_meshes(self, 
                        prediction: np.ndarray, 
                        brain_data: np.ndarray,
                        output_dir: Path,
                        generate_brain_surface: bool = True) -> Dict[str, Any]:
        """
        Generate 3D meshes from segmentation prediction.
        
        Args:
            prediction: Segmentation mask (D, H, W) with values 0-3
            brain_data: Original MRI data for brain surface extraction
            output_dir: Directory to save mesh files
            generate_brain_surface: Whether to generate brain surface mesh
            
        Returns:
            Dictionary with mesh data for Three.js and statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "brain_mesh": None,
            "tumor_meshes": {},
            "tumor_stats": {},
            "bounding_box": None,
            "center_of_mass": None
        }
        
        # Calculate volume statistics
        voxel_volume_mm3 = 1.0  # Assuming 1mm isotropic, adjust based on spacing
        
        # Generate brain surface mesh
        if generate_brain_surface:
            brain_mesh = self._extract_brain_surface(brain_data)
            if brain_mesh:
                result["brain_mesh"] = brain_mesh
        
        # Generate tumor meshes for each class
        for class_id in [1, 2, 3]:
            mask = (prediction == class_id).astype(np.float32)
            
            if mask.sum() < 10:  # Skip if too few voxels
                continue
            
            # Calculate statistics
            volume_voxels = int(mask.sum())
            volume_mm3 = volume_voxels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000.0
            
            # Get centroid
            coords = np.argwhere(mask > 0)
            centroid = coords.mean(axis=0).tolist() if len(coords) > 0 else [0, 0, 0]
            
            # Get bounding box
            if len(coords) > 0:
                bbox_min = coords.min(axis=0).tolist()
                bbox_max = coords.max(axis=0).tolist()
                dimensions = (np.array(bbox_max) - np.array(bbox_min)).tolist()
            else:
                bbox_min = bbox_max = dimensions = [0, 0, 0]
            
            # Store statistics
            class_info = TUMOR_COLORS[class_id]
            result["tumor_stats"][class_info["name"]] = {
                "class_id": class_id,
                "label": class_info["label"],
                "volume_voxels": volume_voxels,
                "volume_mm3": round(volume_mm3, 2),
                "volume_cm3": round(volume_cm3, 3),
                "centroid": [round(c, 1) for c in centroid],
                "bounding_box": {
                    "min": bbox_min,
                    "max": bbox_max,
                    "dimensions": dimensions
                },
                "color": class_info["hex"]
            }
            
            # Generate mesh
            mesh_data = self._extract_surface_mesh(mask, class_id)
            if mesh_data:
                result["tumor_meshes"][class_info["name"]] = mesh_data
        
        # Calculate overall tumor metrics
        total_tumor_mask = (prediction > 0).astype(np.float32)
        if total_tumor_mask.sum() > 0:
            coords = np.argwhere(total_tumor_mask > 0)
            result["center_of_mass"] = coords.mean(axis=0).tolist()
            result["bounding_box"] = {
                "min": coords.min(axis=0).tolist(),
                "max": coords.max(axis=0).tolist()
            }
        
        # Save mesh data as JSON for Three.js
        json_path = output_dir / "mesh_data.json"
        with open(json_path, 'w') as f:
            json.dump(self._prepare_json_output(result), f)
        
        # Generate GLTF file for AR
        gltf_path = output_dir / "brain_model.gltf"
        self._export_gltf(result, gltf_path)
        
        return result
    
    def _extract_brain_surface(self, brain_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract brain surface from MRI data"""
        try:
            # Create brain mask using thresholding
            threshold = np.percentile(brain_data[brain_data > 0], 10) if np.any(brain_data > 0) else 0
            brain_mask = (brain_data > threshold).astype(np.float32)
            
            # Smooth the mask
            brain_mask = gaussian_filter(brain_mask, sigma=1.5)
            
            # Extract surface using marching cubes
            verts, faces, normals, _ = measure.marching_cubes(
                brain_mask, 
                level=0.5,
                step_size=self.step_size * 2,  # Coarser for brain surface
                allow_degenerate=False
            )
            
            # Decimate mesh for performance
            verts, faces = self._decimate_mesh(verts, faces, ratio=0.2)
            
            return {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "normals": normals.tolist() if normals is not None else None,
                "color": BRAIN_COLOR["color"],
                "opacity": 0.15,
                "vertex_count": len(verts),
                "face_count": len(faces)
            }
        except Exception as e:
            print(f"⚠️ Failed to extract brain surface: {e}")
            return None
    
    def _extract_surface_mesh(self, mask: np.ndarray, class_id: int) -> Optional[Dict[str, Any]]:
        """Extract surface mesh from binary mask"""
        try:
            # Smooth the mask for cleaner surface
            smoothed = gaussian_filter(mask, sigma=1.0)
            
            # Apply marching cubes
            verts, faces, normals, _ = measure.marching_cubes(
                smoothed, 
                level=0.5,
                step_size=self.step_size,
                allow_degenerate=False
            )
            
            # Decimate for performance
            verts, faces = self._decimate_mesh(verts, faces, ratio=self.decimate_ratio)
            
            color_info = TUMOR_COLORS[class_id]
            
            return {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "normals": normals.tolist() if normals is not None else None,
                "color": color_info["color"],
                "hex_color": color_info["hex"],
                "opacity": color_info["color"][3],
                "vertex_count": len(verts),
                "face_count": len(faces)
            }
        except Exception as e:
            print(f"⚠️ Failed to extract mesh for class {class_id}: {e}")
            return None
    
    def _decimate_mesh(self, vertices: np.ndarray, faces: np.ndarray, ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple mesh decimation by removing every nth vertex.
        For production, use proper decimation libraries like PyMeshLab.
        """
        if ratio >= 1.0 or len(vertices) < 100:
            return vertices, faces
        
        # Simple approach: subsample vertices uniformly
        keep_every = max(1, int(1.0 / ratio))
        
        # Create a mapping from old to new indices
        old_to_new = {}
        new_vertices = []
        
        for i in range(0, len(vertices), keep_every):
            old_to_new[i] = len(new_vertices)
            new_vertices.append(vertices[i])
        
        # Also keep vertices that are part of remaining faces
        new_faces = []
        for face in faces:
            new_face = []
            valid = True
            for idx in face:
                # Find nearest kept vertex
                nearest_kept = (idx // keep_every) * keep_every
                if nearest_kept >= len(vertices):
                    nearest_kept = len(vertices) - 1
                if nearest_kept not in old_to_new:
                    valid = False
                    break
                new_face.append(old_to_new[nearest_kept])
            
            if valid and len(set(new_face)) == 3:  # Valid triangle
                new_faces.append(new_face)
        
        return np.array(new_vertices), np.array(new_faces)
    
    def _prepare_json_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare result for JSON serialization"""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        return convert_numpy(result)
    
    def _export_gltf(self, result: Dict[str, Any], output_path: Path):
        """Export meshes to GLTF format for AR applications"""
        
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "BraTS 3D Pipeline"
            },
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "materials": []
        }
        
        node_index = 0
        buffer_data = bytearray()
        
        # Add brain mesh
        if result.get("brain_mesh") and result["brain_mesh"].get("vertices"):
            self._add_mesh_to_gltf(
                gltf, buffer_data, result["brain_mesh"], 
                "Brain", BRAIN_COLOR["color"], node_index
            )
            gltf["scenes"][0]["nodes"].append(node_index)
            node_index += 1
        
        # Add tumor meshes
        for tumor_name, mesh_data in result.get("tumor_meshes", {}).items():
            if mesh_data and mesh_data.get("vertices"):
                color = mesh_data.get("color", [1, 0, 0, 1])
                self._add_mesh_to_gltf(
                    gltf, buffer_data, mesh_data,
                    tumor_name, color, node_index
                )
                gltf["scenes"][0]["nodes"].append(node_index)
                node_index += 1
        
        # Write buffer to file
        if buffer_data:
            buffer_path = output_path.with_suffix('.bin')
            with open(buffer_path, 'wb') as f:
                f.write(buffer_data)
            
            gltf["buffers"].append({
                "uri": buffer_path.name,
                "byteLength": len(buffer_data)
            })
        
        # Write GLTF
        with open(output_path, 'w') as f:
            json.dump(gltf, f, indent=2)
    
    def _add_mesh_to_gltf(self, gltf: Dict, buffer_data: bytearray, 
                          mesh_data: Dict, name: str, color: List[float], 
                          node_index: int):
        """Add a mesh to GLTF structure"""
        vertices = np.array(mesh_data["vertices"], dtype=np.float32)
        faces = np.array(mesh_data["faces"], dtype=np.uint32)
        
        if len(vertices) == 0 or len(faces) == 0:
            return
        
        # Calculate bounds
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        
        # Add position accessor
        buffer_offset = len(buffer_data)
        vertices_bytes = vertices.tobytes()
        buffer_data.extend(vertices_bytes)
        
        pos_accessor_idx = len(gltf["accessors"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": buffer_offset,
            "byteLength": len(vertices_bytes),
            "target": 34962  # ARRAY_BUFFER
        })
        gltf["accessors"].append({
            "bufferView": len(gltf["bufferViews"]) - 1,
            "componentType": 5126,  # FLOAT
            "count": len(vertices),
            "type": "VEC3",
            "min": v_min.tolist(),
            "max": v_max.tolist()
        })
        
        # Add index accessor
        buffer_offset = len(buffer_data)
        indices = faces.flatten().astype(np.uint32)
        indices_bytes = indices.tobytes()
        buffer_data.extend(indices_bytes)
        
        idx_accessor_idx = len(gltf["accessors"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": buffer_offset,
            "byteLength": len(indices_bytes),
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })
        gltf["accessors"].append({
            "bufferView": len(gltf["bufferViews"]) - 1,
            "componentType": 5125,  # UNSIGNED_INT
            "count": len(indices),
            "type": "SCALAR"
        })
        
        # Add material
        mat_idx = len(gltf["materials"])
        gltf["materials"].append({
            "name": f"{name}_material",
            "pbrMetallicRoughness": {
                "baseColorFactor": color,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            },
            "alphaMode": "BLEND" if color[3] < 1.0 else "OPAQUE",
            "doubleSided": True
        })
        
        # Add mesh
        mesh_idx = len(gltf["meshes"])
        gltf["meshes"].append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": pos_accessor_idx},
                "indices": idx_accessor_idx,
                "material": mat_idx
            }]
        })
        
        # Add node
        gltf["nodes"].append({
            "name": name,
            "mesh": mesh_idx
        })


# =============================================================================
# CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    import SimpleITK as sitk
    
    parser = argparse.ArgumentParser(description="Generate 3D mesh from segmentation")
    parser.add_argument("--segmentation", type=str, required=True, help="Path to segmentation NIfTI")
    parser.add_argument("--brain", type=str, help="Path to brain MRI NIfTI (for surface)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load segmentation
    seg_img = sitk.ReadImage(args.segmentation)
    prediction = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
    
    # Load brain data
    if args.brain:
        brain_img = sitk.ReadImage(args.brain)
        brain_data = sitk.GetArrayFromImage(brain_img).astype(np.float32)
    else:
        brain_data = np.zeros_like(prediction, dtype=np.float32)
    
    # Generate meshes
    generator = MeshGenerator()
    result = generator.generate_meshes(prediction, brain_data, Path(args.output))
    
    print(f"✅ Meshes saved to: {args.output}")
    print(f"   Tumor statistics: {result['tumor_stats']}")
