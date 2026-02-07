# =============================================================================
# PYDANTIC SCHEMAS
# 
# Request/Response schemas for API endpoints
# These define what data comes IN and goes OUT of your API
# Think of them as the "contract" between frontend and backend
# =============================================================================

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

# =============================================================================
# AUTHENTICATION SCHEMAS
# =============================================================================

class UserCreate(BaseModel):
    """
    Schema for user registration (signup).
    This is what the frontend sends when someone creates an account.
    """
    email: EmailStr  # Must be valid email format
    password: str = Field(..., min_length=8)  # Minimum 8 characters
    full_name: str = Field(..., min_length=2)  # At least 2 characters
    role: str = "doctor"  # Default role
    hospital: Optional[str] = None  # Hospital or institution name
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "doctor@hospital.com",
                "password": "SecurePass123",
                "full_name": "Dr. John Smith",
                "role": "doctor",
                "hospital": "General Hospital"
            }
        }


class UserLogin(BaseModel):
    """
    Schema for user login.
    Simple: just email and password.
    """
    email: EmailStr
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "doctor@hospital.com",
                "password": "SecurePass123"
            }
        }


class UserResponse(BaseModel):
    """
    Schema for user response (NEVER include password!).
    This is what the API returns when you ask for user info.
    """
    id: str
    email: EmailStr
    full_name: str
    role: str
    hospital: Optional[str] = None
    is_active: bool
    created_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "email": "doctor@hospital.com",
                "full_name": "Dr. John Smith",
                "role": "doctor",
                "hospital": "General Hospital",
                "is_active": True,
                "created_at": "2024-01-15T10:30:00"
            }
        }


class Token(BaseModel):
    """
    Schema for JWT token response after login.
    The frontend stores this token and sends it with every request.
    """
    access_token: str  # The JWT token (long encrypted string)
    token_type: str = "bearer"  # Always "bearer" for JWT
    user: UserResponse  # Also return user info for convenience
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": "507f1f77bcf86cd799439011",
                    "email": "doctor@hospital.com",
                    "full_name": "Dr. John Smith",
                    "role": "doctor",
                    "is_active": True,
                    "created_at": "2024-01-15T10:30:00"
                }
            }
        }


class TokenData(BaseModel):
    """
    Schema for decoded token data (internal use).
    This is what we get after decoding a JWT token.
    """
    email: EmailStr
    user_id: str
                
            
    


class TokenData(BaseModel):
    """
    Schema for data stored inside JWT token.
    When we decode a token, this is what we get out.
    """
    email: Optional[str] = None
    user_id: Optional[str] = None

# =============================================================================
# SESSION SCHEMAS
# =============================================================================

class SessionCreate(BaseModel):
    """
    Schema for creating an analysis session.
    Optional patient info can be provided.
    """
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "patient_name": "John Doe"
            }
        }


class SessionResponse(BaseModel):
    """
    Schema for session response.
    This is what the API returns after creating/fetching a session.
    """
    session_id: str
    user_id: str
    patient_id: Optional[str]
    patient_name: Optional[str]
    created_at: str
    expires_at: str
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "a3f7b9c2",
                "user_id": "507f1f77bcf86cd799439011",
                "patient_id": "P12345",
                "patient_name": "John Doe",
                "created_at": "2024-01-15T10:30:00",
                "expires_at": "2024-01-16T10:30:00",
                "status": "pending"
            }
        }


class AuthSessionResponse(BaseModel):
    """
    Schema for authentication session (login session).
    This tracks user login sessions for security.
    """
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool

# =============================================================================
# PATIENT DATA SCHEMAS
# =============================================================================

class PatientDataCreate(BaseModel):
    """
    Schema for creating patient data record.
    Stores detailed patient information.
    """
    session_id: str
    patient_id: str
    patient_name: str
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    scan_date: Optional[datetime] = None
    diagnosis: Optional[str] = None
    notes: Optional[str] = None


class PatientDataResponse(BaseModel):
    """
    Schema for patient data response.
    """
    id: str
    session_id: str
    user_id: str
    patient_id: str
    patient_name: str
    patient_age: Optional[str]
    patient_gender: Optional[str]
    created_at: datetime
