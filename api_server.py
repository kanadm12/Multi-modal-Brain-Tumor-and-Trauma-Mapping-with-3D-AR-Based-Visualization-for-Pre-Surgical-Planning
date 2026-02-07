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
from inference_stable import StableUNet3DInference, create_inference_engine
from visualize_3d import MeshGenerator
from report_generator import ReportGenerator

# Import authentication & database
from database import connect_to_mongo, close_mongo_connection
from auth import get_current_user, require_role
from crud import create_user, authenticate_user, get_user_sessions, invalidate_session, invalidate_all_sessions, create_audit_log
from schemas import UserCreate, UserLogin, Token, UserResponse
from models import UserInDB
from fastapi import Depends

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
OUTPUT_DIR = WORKSPACE_DIR / "pipeline_outputs"
CHECKPOINT_DIR = Path(os.getenv("MODEL_CHECKPOINT_DIR", "checkpoints"))
# Specific model path (if set, uses this instead of auto-detection)
MODEL_PATH = os.getenv("MODEL_PATH", None)

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

# Global inference model
inference_engine: Optional[StableUNet3DInference] = None
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
    
    # Connect to MongoDB
    print("🔌 Connecting to MongoDB...")
    await connect_to_mongo()
    print("✅ MongoDB connected")
    
    # Load ML model - prioritize pretrained StableUNet3D
    print("🧠 Loading AI model...")
    checkpoint_path = None
    
    # Priority 1: Use MODEL_PATH environment variable if set
    if MODEL_PATH:
        checkpoint_path = Path(MODEL_PATH)
        if not checkpoint_path.exists():
            print(f"⚠️ MODEL_PATH specified but file not found: {MODEL_PATH}")
            checkpoint_path = None
    
    # Priority 2: Auto-detect best checkpoint (prioritizes unet_modified_83_38.pth)
    if not checkpoint_path:
        checkpoint_path = find_best_checkpoint()
    
    if checkpoint_path:
        try:
            print(f"📦 Loading StableUNet3D model from: {checkpoint_path}")
            inference_engine = StableUNet3DInference(str(checkpoint_path))
            print(f"✅ StableUNet3D model loaded successfully!")
            print(f"   Using device: {inference_engine.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"⚠️ Running in DEMO MODE (fake predictions)")
            inference_engine = None
    else:
        print("⚠️ No checkpoint found. Running in DEMO MODE.")
        print("   Place your model file (.pth) in the 'checkpoints/' directory")
        print("   Or set MODEL_PATH environment variable")
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
    
    # Close MongoDB connection
    print("🔌 Closing MongoDB connection...")
    await close_mongo_connection()
    print("✅ MongoDB disconnected")


def find_best_checkpoint() -> Optional[Path]:
    """Find the best checkpoint - prioritizes the pretrained StableUNet3D model"""
    # Priority 1: Check for pretrained StableUNet3D model in current directory
    pretrained_model = Path("unet_modified_83_38.pth")
    if pretrained_model.exists():
        print(f"   Found pretrained model: {pretrained_model.name}")
        return pretrained_model
    
    # Priority 2: Check in checkpoint directory
    if not CHECKPOINT_DIR.exists():
        print(f"   Checkpoint directory not found: {CHECKPOINT_DIR}")
        return None
    
    # Check for the pretrained model in checkpoint dir
    checkpoint_pretrained = CHECKPOINT_DIR / "unet_modified_83_38.pth"
    if checkpoint_pretrained.exists():
        print(f"   Found pretrained model: {checkpoint_pretrained.name}")
        return checkpoint_pretrained
    
    # Check if CHECKPOINT_DIR itself is a PyTorch model directory
    data_pkl = CHECKPOINT_DIR / "data.pkl"
    if data_pkl.exists():
        print(f"   Found PyTorch model at: {data_pkl}")
        return data_pkl
    
    # Look for best model files (in priority order)
    patterns = [
        "*best*.pth",           # Files with 'best' in name
        "fold_*_best.pth",      # Cross-validation best models
        "checkpoint_*.pth",     # Training checkpoints
        "*.pth"                 # Any PyTorch model
    ]
    
    for pattern in patterns:
        checkpoints = list(CHECKPOINT_DIR.glob(pattern))
        if checkpoints:
            best = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"   Found model: {best.name}")
            return best
    
    print(f"   No .pth files found in {CHECKPOINT_DIR}")
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

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@app.post("/api/signup", response_model=UserResponse, tags=["Authentication"])
async def signup(user_data: UserCreate):
    """
    Register a new user.
    
    Creates a new account with email, password, and user details.
    Password is automatically hashed for security.
    
    Returns:
        UserResponse with user info (without password)
    """
    try:
        user = await create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/api/login", response_model=Token, tags=["Authentication"])
async def login(credentials: UserLogin):
    """
    Login and get JWT access token.
    
    Verifies email and password, then returns a JWT token.
    Include this token in the Authorization header for protected endpoints.
    
    Returns:
        Token with access_token and user info
    """
    token = await authenticate_user(credentials.email, credentials.password)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


@app.get("/api/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    This is a protected endpoint - requires valid JWT token.
    Returns the user info from the token.
    """
    return current_user


@app.post("/api/logout", tags=["Authentication"])
async def logout(current_user: UserResponse = Depends(get_current_user)):
    """
    Logout from current session.
    
    Invalidates all active sessions for the user.
    Client should also delete the stored token.
    """
    try:
        count = await invalidate_all_sessions(current_user.id)
        await create_audit_log(
            user_id=current_user.id,
            action="logout",
            details={"sessions_invalidated": count}
        )
        return {
            "message": "Logged out successfully",
            "sessions_invalidated": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")


@app.get("/api/sessions", tags=["Sessions"])
async def get_analysis_sessions(current_user: UserResponse = Depends(get_current_user)):
    """
    Get all analysis sessions (patient scans) created by the logged-in doctor.
    
    Returns all patients and scans associated with this doctor for the dashboard.
    """
    try:
        from crud import get_doctor_sessions
        sessions = await get_doctor_sessions(current_user.id)
        
        # Format for frontend dashboard
        result = []
        for session in sessions:
            result.append({
                "session_id": session["session_id"],
                "patient_name": session.get("patient_name", "Unknown"),
                "patient_age": session.get("patient_age"),
                "created_at": session["created_at"].isoformat() if isinstance(session.get("created_at"), datetime) else session.get("created_at", ""),
                "status": session.get("status", "unknown"),
                "has_report": session.get("status") == "completed"
            })
        
        return result
    except Exception as e:
        print(f"❌ Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@app.get("/api/my-sessions", tags=["Sessions"])
async def get_my_analysis_sessions(current_user: UserResponse = Depends(get_current_user)):
    """
    Get all analysis sessions (patient sessions) created by the logged-in doctor.
    
    Returns all patients and scans associated with this doctor.
    """
    try:
        from crud import get_doctor_sessions
        doctor_sessions = await get_doctor_sessions(current_user.id)
        return {
            "doctor_id": current_user.id,
            "doctor_name": current_user.full_name,
            "total_sessions": len(doctor_sessions),
            "sessions": doctor_sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "BraTS 3D Brain Tumor Segmentation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "signup": "POST /api/signup",
            "login": "POST /api/login",
            "me": "GET /api/me (protected)",
            "logout": "POST /api/logout (protected)",
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
async def create_session(
    request: SessionCreateRequest,
    current_user: UserResponse = Depends(get_current_user)  # REQUIRES LOGIN!
):
    """Create a new analysis session (PROTECTED - requires login)"""
    from crud import create_analysis_session, create_patient_data
    from datetime import timedelta
    
    session_id = str(uuid.uuid4())[:8]
    
    # Create session directory
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR / session_id
    
    # Save session to DATABASE (linked to logged-in doctor)
    try:
        db_session_id = await create_analysis_session(
            user_id=current_user.id,
            session_id=session_id,
            patient_info=request.patient.model_dump(),
            doctor_info={
                "name": current_user.full_name,
                "email": current_user.email,
                "designation": request.doctor.designation,
                "hospital": request.doctor.hospital
            },
            upload_dir=str(session_dir),
            output_dir=str(output_dir)
        )
        
        # Also create patient data record
        await create_patient_data(
            user_id=current_user.id,
            session_id=session_id,
            patient_name=request.patient.name,
            patient_info=request.patient.model_dump()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")
    
    # Also keep in memory for backward compatibility during processing
    sessions[session_id] = {
        "patient": request.patient.model_dump(),
        "doctor": request.doctor.model_dump(),
        "doctor_id": current_user.id,  # Track who created it
        "status": "created",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "upload_dir": str(session_dir),
        "output_dir": str(output_dir),
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
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    current_user: UserResponse = Depends(get_current_user)  # REQUIRES LOGIN!
):
    """Upload MRI scan files (NIfTI format) - PROTECTED endpoint"""
    from crud import verify_session_ownership, update_session_status, create_audit_log
    
    # Verify session exists and belongs to this doctor
    owns_session = await verify_session_ownership(session_id, current_user.id)
    if not owns_session:
        raise HTTPException(
            status_code=403,
            detail="Session not found or you don't have permission to upload to this session"
        )
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Additional security check
    if session.get("doctor_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You can only upload to your own sessions")
    
    session_dir = Path(session["upload_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    uploaded_files = []
    
    for file in files:
        # Validate file extension
        if not (file.filename.endswith('.nii') or file.filename.endswith('.nii.gz')):
            continue
        
        # Clean filename - remove any path components
        clean_filename = Path(file.filename).name
        
        # Save file
        file_path = session_dir / clean_filename
        file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dir exists
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        uploaded_files.append(clean_filename)
    
    session["files"] = uploaded_files
    session["status"] = "uploaded"
    session["progress"] = 5
    
    # Update in DATABASE
    await update_session_status(
        session_id=session_id,
        status="uploaded",
        files_uploaded=uploaded_files
    )
    
    # Log the upload
    await create_audit_log(
        user_id=current_user.id,
        action="files_uploaded",
        details={
            "session_id": session_id,
            "file_count": len(uploaded_files),
            "files": uploaded_files
        }
    )
    
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
async def run_prediction(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: UserResponse = Depends(get_current_user)  # REQUIRES LOGIN!
):
    """Run tumor segmentation on uploaded MRI scans - PROTECTED endpoint"""
    from crud import verify_session_ownership, update_session_status, create_audit_log
    
    # Verify session exists and belongs to this doctor
    owns_session = await verify_session_ownership(session_id, current_user.id)
    if not owns_session:
        raise HTTPException(
            status_code=403,
            detail="Session not found or you don't have permission to run prediction"
        )
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Additional security check
    if session.get("doctor_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You can only run predictions on your own sessions")
    
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
    
    # Update in database
    from crud import update_session_status
    await update_session_status(session_id=session_id, status="processing")
    
    # Log the action
    from crud import create_audit_log
    await create_audit_log(
        user_id=current_user.id,
        action="prediction_started",
        details={"session_id": session_id}
    )
    
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
            # StableUNet3DInference returns (prediction, volumes, stats)
            prediction, volumes, stats = inference_engine.predict(session_dir)
            session["progress"] = 50
            session["volumes"] = volumes  # Store tumor volumes
            session["tumor_stats"] = stats  # Store detailed tumor statistics
            
            # Load input data for mesh generation (use T1CE as reference)
            t1ce_files = list(session_dir.glob("*t1ce*.nii*")) + list(session_dir.glob("*T1CE*.nii*"))
            if t1ce_files:
                img = sitk.ReadImage(str(t1ce_files[0]))
                input_data = sitk.GetArrayFromImage(img).astype(np.float32)
            else:
                # Use any available file
                nii_files = list(session_dir.glob("*.nii*"))
                if nii_files:
                    img = sitk.ReadImage(str(nii_files[0]))
                    input_data = sitk.GetArrayFromImage(img).astype(np.float32)
                else:
                    input_data = None
        else:
            # Demo mode - generate sample segmentation
            session["message"] = "Generating demo segmentation..."
            prediction, input_data = generate_demo_prediction(session_dir)
            session["progress"] = 50
            session["volumes"] = {"ncr": 0, "ed": 0, "et": 0, "total": 0}
            session["tumor_stats"] = {}
        
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
        
        # Build tumor stats for report generator
        tumor_stats_for_report = session.get("tumor_stats", {})
        if not tumor_stats_for_report and session.get("volumes"):
            # Convert volumes dict to the format expected by report generator
            volumes = session["volumes"]
            tumor_stats_for_report = {
                "NCR": {"volume_cm3": volumes.get("ncr", 0), "voxel_count": 0},
                "ED": {"volume_cm3": volumes.get("ed", 0), "voxel_count": 0},
                "ET": {"volume_cm3": volumes.get("et", 0), "voxel_count": 0},
            }
        
        # Generate report with tumor volumes and stats
        session["message"] = "Generating clinical report..."
        report = report_generator.generate_report(
            tumor_stats=tumor_stats_for_report if tumor_stats_for_report else mesh_data.get("tumor_stats", {}),
            patient_info=session["patient"],
            doctor_info=session["doctor"],
            output_dir=output_dir,
            session_id=session_id
        )
        session["report"] = report
        session["progress"] = 100
        
        session["status"] = "completed"
        session["message"] = "Analysis complete!"
        
        # Update DATABASE with completion status
        from crud import update_session_status
        import asyncio
        await update_session_status(
            session_id=session_id,
            status="completed",
            has_mesh=True,
            has_report=True,
            prediction_path=str(prediction_path)
        )
        
    except Exception as e:
        session["status"] = "failed"
        session["error"] = str(e)
        session["message"] = f"Error: {str(e)}"
        
        # Update database with error status
        from crud import update_session_status
        import asyncio
        await update_session_status(
            session_id=session_id,
            status="failed"
        )
        
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
    
    # Try in-memory first
    if session_id in sessions:
        session = sessions[session_id]
        
        if session["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Prediction not completed. Current status: {session['status']}"
            )
        
        if session.get("mesh_data"):
            return session["mesh_data"]
    
    # If not in memory, try loading from disk
    output_dir = OUTPUT_DIR / session_id
    mesh_json_path = output_dir / "mesh_data.json"
    
    if mesh_json_path.exists():
        import json
        with open(mesh_json_path, 'r') as f:
            mesh_data = json.load(f)
        return mesh_data
    
    raise HTTPException(status_code=404, detail="Session or mesh data not found")



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
    pdf_path = Path(session["output_dir"]) / "report.pdf"
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found. Make sure ReportLab is installed.")
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
