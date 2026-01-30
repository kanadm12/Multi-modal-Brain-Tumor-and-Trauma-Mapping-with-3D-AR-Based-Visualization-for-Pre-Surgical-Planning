# =============================================================================
# BRATS INFERENCE PIPELINE - FASTAPI BACKEND
# 
# Endpoints:
# - POST /api/session/create - Create analysis session
# - POST /api/upload/{session_id} - Upload MRI scans
# - POST /api/predict/{session_id} - Run segmentation inference
# - GET /api/status/{session_id} - Get prediction status
# - GET /api/mesh/{session_id} - Get 3D mesh data for Three.js
# - GET /api/report/{session_id} - Get clinical report
# - GET /api/report/{session_id}/pdf - Download PDF report
# - GET /api/model/{session_id}/gltf - Download 3D model
# =============================================================================

import os
import sys
import uuid
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import SimpleITK as sitk

# Import pipeline modules
from inference import BraTSInference
from visualize_3d import MeshGenerator
from report_generator import ReportGenerator

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
OUTPUT_DIR = WORKSPACE_DIR / "pipeline_outputs"
CHECKPOINT_DIR = WORKSPACE_DIR / "checkpoints"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

# Global inference model
inference_engine: Optional[BraTSInference] = None
mesh_generator: Optional[MeshGenerator] = None
report_generator: Optional[ReportGenerator] = None


# =============================================================================
# LIFESPAN (Load model on startup)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global inference_engine, mesh_generator, report_generator
    
    print("=" * 60)
    print("🚀 BraTS 3D Segmentation API Starting...")
    print("=" * 60)
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint()
    
    if checkpoint_path:
        try:
            inference_engine = BraTSInference(str(checkpoint_path))
            print(f"✅ Model loaded from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            inference_engine = None
    else:
        print("⚠️ No checkpoint found. Running in demo mode.")
        inference_engine = None
    
    # Initialize mesh generator and report generator
    mesh_generator = MeshGenerator()
    report_generator = ReportGenerator()
    
    print("✅ Mesh generator initialized")
    print("✅ Report generator initialized")
    print("=" * 60)
    print("🎉 Pipeline ready!")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("🧹 Shutting down...")
    if inference_engine:
        del inference_engine
    torch.cuda.empty_cache()


def find_best_checkpoint() -> Optional[Path]:
    """Find the best checkpoint from training"""
    if not CHECKPOINT_DIR.exists():
        return None
    
    # Look for best model files
    patterns = ["*best*.pth", "fold_*_best.pth", "*.pth"]
    for pattern in patterns:
        checkpoints = list(CHECKPOINT_DIR.glob(pattern))
        if checkpoints:
            # Return most recent
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="BraTS 3D Brain Tumor Segmentation API",
    description="Upload MRI scans, get 3D tumor visualization and clinical reports",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PatientInfo(BaseModel):
    name: str
    age: Optional[str] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    disorder: Optional[str] = None
    description: Optional[str] = None


class DoctorInfo(BaseModel):
    name: str
    email: str
    designation: str
    hospital: str


class SessionCreateRequest(BaseModel):
    patient: PatientInfo
    doctor: DoctorInfo


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


class PredictionStatus(BaseModel):
    session_id: str
    status: str  # "created", "uploaded", "processing", "completed", "failed"
    progress: int  # 0-100
    message: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "BraTS 3D Brain Tumor Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "create_session": "POST /api/session/create",
            "upload": "POST /api/upload/{session_id}",
            "predict": "POST /api/predict/{session_id}",
            "status": "GET /api/status/{session_id}",
            "mesh": "GET /api/mesh/{session_id}",
            "report": "GET /api/report/{session_id}",
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None
    }


@app.post("/api/session/create", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new analysis session"""
    session_id = str(uuid.uuid4())[:8]
    
    # Create session directory
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Store session info
    sessions[session_id] = {
        "patient": request.patient.model_dump(),
        "doctor": request.doctor.model_dump(),
        "status": "created",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "upload_dir": str(session_dir),
        "output_dir": str(OUTPUT_DIR / session_id),
        "files": [],
        "prediction_path": None,
        "mesh_data": None,
        "report": None,
        "error": None
    }
    
    return SessionResponse(
        session_id=session_id,
        status="created",
        message="Session created successfully. Upload MRI files next."
    )


@app.post("/api/upload/{session_id}")
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    """Upload MRI scan files (NIfTI format)"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session_dir = Path(session["upload_dir"])
    
    uploaded_files = []
    
    for file in files:
        # Validate file extension
        if not (file.filename.endswith('.nii') or file.filename.endswith('.nii.gz')):
            continue
        
        # Save file
        file_path = session_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        uploaded_files.append(file.filename)
    
    session["files"] = uploaded_files
    session["status"] = "uploaded"
    session["progress"] = 5
    
    # Validate we have required modalities
    required_modalities = ["t1", "t1ce", "t2", "flair"]
    found_modalities = []
    for f in uploaded_files:
        f_lower = f.lower()
        for mod in required_modalities:
            if mod in f_lower:
                found_modalities.append(mod)
    
    missing = set(required_modalities) - set(found_modalities)
    
    return {
        "session_id": session_id,
        "uploaded_files": uploaded_files,
        "file_count": len(uploaded_files),
        "status": "uploaded",
        "modalities_found": list(set(found_modalities)),
        "modalities_missing": list(missing) if missing else None,
        "ready_for_prediction": len(missing) == 0
    }


@app.post("/api/predict/{session_id}")
async def run_prediction(session_id: str, background_tasks: BackgroundTasks):
    """Run tumor segmentation on uploaded MRI scans"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] not in ["uploaded", "failed"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid session status: {session['status']}. Expected 'uploaded'."
        )
    
    if not session["files"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Start async prediction
    session["status"] = "processing"
    session["progress"] = 10
    session["error"] = None
    
    background_tasks.add_task(run_inference_task, session_id)
    
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Prediction started. Poll /api/status/{session_id} for progress."
    }


async def run_inference_task(session_id: str):
    """Background task for running inference"""
    session = sessions[session_id]
    
    try:
        session_dir = Path(session["upload_dir"])
        output_dir = Path(session["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update progress
        session["progress"] = 15
        session["message"] = "Loading MRI scans..."
        
        # Run inference
        if inference_engine:
            session["message"] = "Running AI segmentation..."
            prediction, input_data = inference_engine.predict(session_dir)
            session["progress"] = 50
        else:
            # Demo mode - generate sample segmentation
            session["message"] = "Generating demo segmentation..."
            prediction, input_data = generate_demo_prediction(session_dir)
            session["progress"] = 50
        
        # Save prediction
        session["message"] = "Saving segmentation..."
        prediction_path = output_dir / "segmentation.nii.gz"
        save_nifti(prediction, prediction_path)
        session["prediction_path"] = str(prediction_path)
        session["progress"] = 60
        
        # Generate 3D meshes
        session["message"] = "Generating 3D model..."
        mesh_data = mesh_generator.generate_meshes(
            prediction=prediction,
            brain_data=input_data,
            output_dir=output_dir
        )
        session["mesh_data"] = mesh_data
        session["progress"] = 80
        
        # Generate report
        session["message"] = "Generating clinical report..."
        report = report_generator.generate(
            prediction=prediction,
            patient_info=session["patient"],
            doctor_info=session["doctor"],
            output_dir=output_dir
        )
        session["report"] = report
        session["progress"] = 100
        
        session["status"] = "completed"
        session["message"] = "Analysis complete!"
        
    except Exception as e:
        session["status"] = "failed"
        session["error"] = str(e)
        session["message"] = f"Error: {str(e)}"
        print(f"❌ Inference failed for session {session_id}: {e}")
        import traceback
        traceback.print_exc()


def generate_demo_prediction(session_dir: Path):
    """Generate demo prediction when no model is loaded"""
    # Load first available file to get dimensions
    nii_files = list(session_dir.glob("*.nii*"))
    if not nii_files:
        raise ValueError("No NIfTI files found in upload directory")
    
    img = sitk.ReadImage(str(nii_files[0]))
    data = sitk.GetArrayFromImage(img).astype(np.float32)
    
    # Create fake segmentation (for demo purposes)
    prediction = np.zeros(data.shape, dtype=np.uint8)
    center = np.array(data.shape) // 2
    
    # Create spherical "tumors" for demo
    z, y, x = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Offset center slightly for realism
    tumor_center = center + np.array([5, -10, 15])
    
    dist = np.sqrt(
        (z - tumor_center[0])**2 + 
        (y - tumor_center[1])**2 + 
        (x - tumor_center[2])**2
    )
    
    # NCR (necrotic core) - innermost
    prediction[dist < 12] = 1
    
    # ET (enhancing tumor) - ring around NCR
    prediction[(dist >= 12) & (dist < 22)] = 3
    
    # ED (edema) - outer ring
    prediction[(dist >= 22) & (dist < 38)] = 2
    
    return prediction, data


def save_nifti(data: np.ndarray, path: Path):
    """Save numpy array as NIfTI file"""
    img = sitk.GetImageFromArray(data.astype(np.int16))
    sitk.WriteImage(img, str(path))


@app.get("/api/status/{session_id}", response_model=PredictionStatus)
async def get_prediction_status(session_id: str):
    """Get prediction status and progress"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return PredictionStatus(
        session_id=session_id,
        status=session["status"],
        progress=session.get("progress", 0),
        message=session.get("message", session.get("error", ""))
    )


@app.get("/api/mesh/{session_id}")
async def get_mesh_data(session_id: str):
    """Get 3D mesh data for Three.js rendering"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction not completed. Current status: {session['status']}"
        )
    
    if not session["mesh_data"]:
        raise HTTPException(status_code=404, detail="Mesh data not found")
    
    return session["mesh_data"]


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Get clinical report data"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction not completed. Current status: {session['status']}"
        )
    
    if not session["report"]:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return session["report"]


@app.get("/api/report/{session_id}/pdf")
async def download_report_pdf(session_id: str):
    """Download report as PDF"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    pdf_path = Path(session["output_dir"]) / "clinical_report.pdf"
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found")
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"brain_tumor_report_{session_id}.pdf"
    )


@app.get("/api/model/{session_id}/gltf")
async def download_3d_model(session_id: str):
    """Download 3D model as GLTF for AR applications"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    gltf_path = Path(session["output_dir"]) / "brain_model.gltf"
    
    if not gltf_path.exists():
        raise HTTPException(status_code=404, detail="3D model not found")
    
    return FileResponse(
        path=str(gltf_path),
        media_type="model/gltf+json",
        filename=f"brain_tumor_3d_{session_id}.gltf"
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its files"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Remove upload directory
    upload_dir = Path(session["upload_dir"])
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    # Remove output directory
    output_dir = Path(session["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Remove from sessions
    del sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
