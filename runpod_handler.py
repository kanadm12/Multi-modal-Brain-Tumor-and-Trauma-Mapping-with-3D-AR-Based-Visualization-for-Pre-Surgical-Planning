# =============================================================================
# RUNPOD SERVERLESS HANDLER
# 
# GPU-accelerated inference endpoint for BraTS tumor segmentation
# Receives MRI scans, returns GLTF 3D model with color-coded tumor regions
#
# Deploy: runpodctl deploy --name brats-inference --image <your-dockerhub>/brats-runpod
# =============================================================================

import os
import sys
import json
import base64
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# RunPod SDK
import runpod

# Set up paths
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(SCRIPT_DIR / "unet_modified_83_38.pth"))

# Lazy load heavy imports
_inference_engine = None
_mesh_generator = None


def get_inference_engine():
    """Lazy load inference engine to speed up cold starts"""
    global _inference_engine
    if _inference_engine is None:
        print("🧠 Loading StableUNet3D model...")
        from inference_stable import StableUNet3DInference
        _inference_engine = StableUNet3DInference(MODEL_PATH)
        print(f"✅ Model loaded on device: {_inference_engine.device}")
    return _inference_engine


def get_mesh_generator():
    """Lazy load mesh generator"""
    global _mesh_generator
    if _mesh_generator is None:
        print("🔧 Initializing mesh generator...")
        from visualize_3d import MeshGenerator
        _mesh_generator = MeshGenerator()
        print("✅ Mesh generator ready")
    return _mesh_generator


def decode_and_save_file(file_data: Dict[str, str], output_dir: Path) -> Path:
    """
    Decode base64 file data and save to disk.
    
    Args:
        file_data: Dict with 'filename' and 'content' (base64)
        output_dir: Directory to save file
        
    Returns:
        Path to saved file
    """
    filename = file_data.get("filename", "scan.nii.gz")
    content_b64 = file_data.get("content", "")
    
    # Decode base64
    file_bytes = base64.b64decode(content_b64)
    
    # Save to disk
    file_path = output_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    return file_path


def encode_file_to_base64(file_path: Path) -> str:
    """Read file and encode to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for BraTS inference.
    
    Input event structure:
    {
        "input": {
            "files": [
                {"filename": "patient_t1.nii.gz", "content": "<base64>"},
                {"filename": "patient_t1ce.nii.gz", "content": "<base64>"},
                {"filename": "patient_t2.nii.gz", "content": "<base64>"},
                {"filename": "patient_flair.nii.gz", "content": "<base64>"}
            ],
            "patient_info": {"name": "...", "age": "..."},  # Optional
            "options": {
                "generate_report": true,
                "return_gltf_base64": true,  # If false, only returns mesh_data JSON
                "tta_enabled": true
            }
        }
    }
    
    Output structure:
    {
        "status": "success",
        "tumor_stats": {...},
        "volumes": {...},
        "mesh_data": {...},  # Three.js compatible mesh data
        "gltf_base64": "...",  # Base64 encoded GLTF (if requested)
        "gltf_bin_base64": "...",  # Base64 encoded GLTF binary buffer
        "report_html": "...",  # HTML report (if requested)
    }
    """
    try:
        input_data = event.get("input", {})
        files = input_data.get("files", [])
        patient_info = input_data.get("patient_info", {})
        options = input_data.get("options", {})
        
        # Validate input
        if not files:
            return {"status": "error", "error": "No MRI files provided"}
        
        if len(files) < 4:
            return {
                "status": "error", 
                "error": f"Expected 4 MRI modalities (T1, T1ce, T2, FLAIR), got {len(files)}"
            }
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Save input files
            print(f"📂 Saving {len(files)} MRI files...")
            for file_data in files:
                saved_path = decode_and_save_file(file_data, input_dir)
                print(f"   Saved: {saved_path.name}")
            
            # Run inference
            print("🔬 Running tumor segmentation...")
            inference_engine = get_inference_engine()
            prediction, volumes, stats = inference_engine.predict(input_dir)
            print(f"✅ Segmentation complete. Tumor volumes: {volumes}")
            
            # Load reference MRI for brain surface extraction
            import SimpleITK as sitk
            t1ce_files = list(input_dir.glob("*t1ce*")) + list(input_dir.glob("*t1c*"))
            if t1ce_files:
                ref_img = sitk.ReadImage(str(t1ce_files[0]))
            else:
                ref_img = sitk.ReadImage(str(list(input_dir.glob("*.nii*"))[0]))
            brain_data = sitk.GetArrayFromImage(ref_img).astype(np.float32)
            
            # Generate 3D meshes
            print("🎨 Generating 3D model...")
            mesh_generator = get_mesh_generator()
            mesh_result = mesh_generator.generate_meshes(
                prediction=prediction,
                brain_data=brain_data,
                output_dir=output_dir,
                generate_brain_surface=True
            )
            print("✅ 3D model generated")
            
            # Build response
            response = {
                "status": "success",
                "tumor_stats": mesh_result.get("tumor_stats", {}),
                "volumes": volumes,
                "center_of_mass": mesh_result.get("center_of_mass"),
                "bounding_box": mesh_result.get("bounding_box"),
            }
            
            # Include mesh data for Three.js
            response["mesh_data"] = {
                "brain_mesh": mesh_result.get("brain_mesh"),
                "tumor_meshes": mesh_result.get("tumor_meshes", {}),
            }
            
            # Encode GLTF files if requested
            if options.get("return_gltf_base64", True):
                gltf_path = output_dir / "brain_model.gltf"
                gltf_bin_path = output_dir / "brain_model.bin"
                
                if gltf_path.exists():
                    response["gltf_base64"] = encode_file_to_base64(gltf_path)
                if gltf_bin_path.exists():
                    response["gltf_bin_base64"] = encode_file_to_base64(gltf_bin_path)
            
            # Generate report if requested
            if options.get("generate_report", False):
                try:
                    from report_generator import ReportGenerator
                    report_gen = ReportGenerator()
                    report = report_gen.generate_report(
                        tumor_stats=mesh_result.get("tumor_stats", {}),
                        patient_info=patient_info,
                        output_dir=output_dir
                    )
                    
                    # Read HTML report
                    html_path = output_dir / "report.html"
                    if html_path.exists():
                        response["report_html"] = html_path.read_text()
                except Exception as e:
                    print(f"⚠️ Report generation failed: {e}")
                    response["report_error"] = str(e)
            
            print("🎉 Processing complete!")
            return response
            
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ Error: {error_msg}")
        print(error_trace)
        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace
        }


# =============================================================================
# HEALTH CHECK & WARMUP
# =============================================================================

def health_check(_event: Dict) -> Dict[str, str]:
    """Health check endpoint for RunPod"""
    return {"status": "healthy", "model_loaded": _inference_engine is not None}


def warmup():
    """Pre-load model during container startup"""
    print("🔥 Warming up...")
    try:
        get_inference_engine()
        get_mesh_generator()
        print("✅ Warmup complete - ready for inference!")
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")


# =============================================================================
# RUNPOD ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BraTS 3D Tumor Segmentation - RunPod Serverless")
    print("=" * 60)
    
    # Warmup on startup (optional - enables faster first request)
    if os.getenv("WARMUP_ON_START", "true").lower() == "true":
        warmup()
    
    # Start RunPod serverless worker
    runpod.serverless.start({
        "handler": handler,
        "health_check": health_check
    })
