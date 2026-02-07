# =============================================================================
# DATABASE MODELS - MONGODB SCHEMAS
# 
# Defines data structures for MongoDB collections
# These are models for data stored IN the database
# =============================================================================

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId

# =============================================================================
# CUSTOM OBJECTID TYPE FOR PYDANTIC V2
# =============================================================================

class PyObjectId(str):
    """
    Custom ObjectId type for Pydantic v2 validation.
    MongoDB uses ObjectId for _id field, but Pydantic doesn't know about it by default.
    This class teaches Pydantic how to handle MongoDB ObjectIds.
    """
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: str(x)),
        )
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

# =============================================================================
# USER MODEL - Stored in "users" collection
# =============================================================================

class UserInDB(BaseModel):
    """
    User model as stored in MongoDB.
    This represents a doctor, radiologist, or admin who can login.
    """
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    email: EmailStr  # Validated email format (e.g., doctor@hospital.com)
    hashed_password: str  # NEVER store plain passwords! Always hashed
    full_name: str  # "Dr. John Smith"
    role: str = "doctor"  # "doctor", "radiologist", or "admin"
    hospital: Optional[str] = None  # Hospital or institution name
    is_active: bool = True  # Can deactivate users without deleting them
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True  # Allow using both "id" and "_id"
        arbitrary_types_allowed = True  # Allow ObjectId type
        json_encoders = {ObjectId: str}  # Convert ObjectId to string in JSON

# =============================================================================
# SESSION MODEL - Stored in "sessions" collection
# =============================================================================

class SessionInDB(BaseModel):
    """
    Analysis session model - represents one patient's scan analysis.
    Each time a user uploads scans, a new session is created.
    LINKED TO DOCTOR: Each session belongs to the doctor who created it.
    """
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    session_id: str  # Short ID like "a3f7b9c2" - easier to reference than ObjectId
    user_id: str  # Which user created this session (link to users collection)
    doctor_id: str  # The doctor who created this session (same as user_id for doctors)
    
    # Patient Information
    patient_name: str  # Patient's full name
    patient_age: Optional[str] = None
    patient_weight: Optional[str] = None
    patient_height: Optional[str] = None
    patient_disorder: Optional[str] = None  # Medical condition/disorder
    patient_description: Optional[str] = None  # Additional details
    
    # Doctor Information (who created this session)
    doctor_name: str
    doctor_email: str
    doctor_designation: Optional[str] = None
    doctor_hospital: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime  # Sessions expire after 24 hours
    status: str = "pending"  # "pending" → "processing" → "completed" or "error"
    files_uploaded: List[str] = []  # List of uploaded file names
    has_mesh: bool = False  # Has 3D mesh been generated?
    has_report: bool = False  # Has report been generated?
    
    # File paths
    upload_dir: Optional[str] = None  # Where files are stored
    output_dir: Optional[str] = None  # Where results are stored
    prediction_path: Optional[str] = None  # Segmentation file path
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# =============================================================================
# PATIENT DATA MODEL - Stored in "patient_data" collection
# =============================================================================

class PatientDataInDB(BaseModel):
    """
    Patient information for HIPAA compliance.
    Stores medical information linked to a session.
    DOCTOR-SPECIFIC: Each patient record belongs to the doctor who created it.
    """
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    session_id: str  # Link to sessions collection
    user_id: str  # Link to users collection (which doctor added this)
    doctor_id: str  # The doctor who created this record (same as user_id)
    patient_name: str
    patient_age: Optional[str] = None
    patient_weight: Optional[str] = None
    patient_height: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_disorder: Optional[str] = None  # Medical condition
    patient_description: Optional[str] = None  # Additional details
    scan_date: Optional[datetime] = None
    diagnosis: Optional[str] = None  # Pre-existing diagnosis
    notes: Optional[str] = None  # Doctor's notes
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# =============================================================================
# AUDIT LOG MODEL - Stored in "audit_logs" collection
# =============================================================================

class AuditLogInDB(BaseModel):
    """
    Audit log for HIPAA compliance and security.
    Tracks every action users take - required for medical applications!
    """
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: str  # Who performed the action
    action: str  # "login", "logout", "upload", "view_report", "download", etc.
    resource: Optional[str] = None  # What was accessed (session_id, report_id, etc.)
    ip_address: Optional[str] = None  # User's IP address
    user_agent: Optional[str] = None  # Browser/device information
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[dict] = None  # Additional context as needed
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
