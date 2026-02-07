# =============================================================================
# CRUD OPERATIONS (Create, Read, Update, Delete)
# 
# Database functions for user management, authentication, and data operations
# This is where we actually interact with MongoDB
# =============================================================================

from datetime import datetime
from typing import Optional, List
from bson import ObjectId

from database import get_collection, USERS_COLLECTION, SESSIONS_COLLECTION, PATIENT_DATA_COLLECTION, AUDIT_LOGS_COLLECTION
from auth import get_password_hash, verify_password, create_access_token
from schemas import UserCreate, UserResponse, Token, SessionCreate, SessionResponse, AuthSessionResponse
from models import UserInDB, SessionInDB, PatientDataInDB, AuditLogInDB

# =============================================================================
# USER OPERATIONS (For Signup/Profile Management)
# =============================================================================

async def create_user(user_data: UserCreate) -> UserResponse:
    """
    Create a new user in the database (SIGNUP).
    
    Steps:
    1. Check if email already exists (prevent duplicates)
    2. Hash the password (NEVER store plain text!)
    3. Create user document
    4. Insert into MongoDB
    5. Return user info
    
    Args:
        user_data: UserCreate schema with email, password, full_name, role
        
    Returns:
        UserResponse with user info (without password)
        
    Raises:
        ValueError: If email already exists
        
    Example:
        user = await create_user(UserCreate(
            email="doctor@hospital.com",
            password="SecurePass123",
            full_name="Dr. John Smith",
            role="doctor"
        ))
    """
    users_collection = get_collection(USERS_COLLECTION)
    
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise ValueError(f"User with email {user_data.email} already exists")
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user document
    user_dict = {
        "email": user_data.email,
        "hashed_password": hashed_password,
        "full_name": user_data.full_name,
        "role": user_data.role,
        "hospital": user_data.hospital,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Insert into database
    result = await users_collection.insert_one(user_dict)
    
    # Log the creation
    await create_audit_log(
        user_id=str(result.inserted_id),
        action="user_created",
        details={"email": user_data.email, "role": user_data.role}
    )
    
    # Return user info (without password)
    return UserResponse(
        id=str(result.inserted_id),
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        hospital=user_data.hospital,
        is_active=True,
        created_at=user_dict["created_at"]
    )


async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """
    Find a user by their email address.
    
    Args:
        email: User's email address
        
    Returns:
        UserInDB model if found, None otherwise
        
    Example:
        user = await get_user_by_email("doctor@hospital.com")
        if user:
            print(f"Found user: {user.full_name}")
    """
    users_collection = get_collection(USERS_COLLECTION)
    user_dict = await users_collection.find_one({"email": email})
    
    if user_dict:
        return UserInDB(**user_dict)
    return None


async def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    """
    Find a user by their ID.
    
    Args:
        user_id: User's MongoDB ObjectId as string
        
    Returns:
        UserInDB model if found, None otherwise
    """
    users_collection = get_collection(USERS_COLLECTION)
    user_dict = await users_collection.find_one({"_id": ObjectId(user_id)})
    
    if user_dict:
        return UserInDB(**user_dict)
    return None


async def update_user(user_id: str, updates: dict) -> Optional[UserResponse]:
    """
    Update user information.
    
    Args:
        user_id: User's MongoDB ObjectId as string
        updates: Dictionary of fields to update (e.g., {"full_name": "New Name"})
        
    Returns:
        Updated UserResponse if successful, None if user not found
        
    Example:
        updated_user = await update_user(
            user_id="507f1f77bcf86cd799439011",
            updates={"full_name": "Dr. Jane Doe"}
        )
    """
    users_collection = get_collection(USERS_COLLECTION)
    
    # Add updated_at timestamp
    updates["updated_at"] = datetime.utcnow()
    
    # Update in database
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": updates}
    )
    
    if result.modified_count > 0:
        # Get updated user
        user = await get_user_by_id(user_id)
        if user:
            await create_audit_log(
                user_id=user_id,
                action="user_updated",
                details={"updated_fields": list(updates.keys())}
            )
            return UserResponse(
                id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                is_active=user.is_active,
                created_at=user.created_at
            )
    return None


# =============================================================================
# AUTHENTICATION OPERATIONS (For Login)
# =============================================================================

async def authenticate_user(email: str, password: str) -> Optional[Token]:
    """
    Authenticate user credentials and return JWT token (LOGIN).
    
    This is the main function called when user tries to log in.
    
    Steps:
    1. Find user by email
    2. Check if account is active
    3. Verify password
    4. Create JWT token
    5. Create session record
    6. Log the login
    7. Return token + user info
    
    Args:
        email: User's email
        password: Plain text password
        
    Returns:
        Token object with access_token and user info if successful
        None if authentication fails
        
    Example:
        token = await authenticate_user("doctor@hospital.com", "SecurePass123")
        if token:
            print(f"Login successful! Token: {token.access_token}")
        else:
            print("Invalid credentials")
    """
    # Find user
    user = await get_user_by_email(email)
    if not user:
        return None
    
    # Check if active
    if not user.is_active:
        return None
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        # Log failed login attempt
        await create_audit_log(
            user_id=str(user.id),
            action="login_failed",
            details={"reason": "invalid_password"}
        )
        return None
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(user.id)}
    )
    
    # Create session
    await create_session(
        user_id=str(user.id),
        token=access_token
    )
    
    # Log successful login
    await create_audit_log(
        user_id=str(user.id),
        action="login_success",
        details={"email": user.email}
    )
    
    # Return token and user info
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at
        )
    )


async def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """
    Change user's password.
    
    Args:
        user_id: User's ID
        old_password: Current password (for verification)
        new_password: New password
        
    Returns:
        True if password changed successfully, False otherwise
    """
    user = await get_user_by_id(user_id)
    if not user:
        return False
    
    # Verify old password
    if not verify_password(old_password, user.hashed_password):
        return False
    
    # Hash new password
    new_hash = get_password_hash(new_password)
    
    # Update in database
    users_collection = get_collection(USERS_COLLECTION)
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"hashed_password": new_hash, "updated_at": datetime.utcnow()}}
    )
    
    if result.modified_count > 0:
        await create_audit_log(
            user_id=user_id,
            action="password_changed",
            details={}
        )
        return True
    return False


# =============================================================================
# SESSION MANAGEMENT (Track Active Logins)
# =============================================================================

async def create_session(user_id: str, token: str) -> AuthSessionResponse:
    """
    Create a new session record when user logs in.
    
    This tracks which devices/browsers the user is logged in from.
    
    Args:
        user_id: User's ID
        token: JWT access token
        
    Returns:
        AuthSessionResponse with session info
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    session_dict = {
        "user_id": ObjectId(user_id),
        "token": token,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow(),  # Will be calculated from JWT expiration
        "is_active": True
    }
    
    result = await sessions_collection.insert_one(session_dict)
    
    return AuthSessionResponse(
        id=str(result.inserted_id),
        user_id=user_id,
        created_at=session_dict["created_at"],
        expires_at=session_dict["expires_at"],
        is_active=True
    )


async def get_user_sessions(user_id: str) -> List[AuthSessionResponse]:
    """
    Get all active sessions for a user.
    
    Useful for:
    - "Where you're logged in" feature
    - Security: see all active devices
    - Logout from all devices
    
    Args:
        user_id: User's ID
        
    Returns:
        List of active sessions
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    cursor = sessions_collection.find({
        "user_id": ObjectId(user_id),
        "is_active": True
    })
    
    sessions = []
    async for session_dict in cursor:
        sessions.append(AuthSessionResponse(
            id=str(session_dict["_id"]),
            user_id=user_id,
            created_at=session_dict["created_at"],
            expires_at=session_dict["expires_at"],
            is_active=session_dict["is_active"]
        ))
    
    return sessions


async def invalidate_session(session_id: str) -> bool:
    """
    Invalidate a session (logout from specific device).
    
    Args:
        session_id: Session ID to invalidate
        
    Returns:
        True if session invalidated, False otherwise
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    result = await sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"is_active": False}}
    )
    
    return result.modified_count > 0


async def invalidate_all_sessions(user_id: str) -> int:
    """
    Invalidate all sessions for a user (logout from all devices).
    
    Args:
        user_id: User's ID
        
    Returns:
        Number of sessions invalidated
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    result = await sessions_collection.update_many(
        {"user_id": ObjectId(user_id), "is_active": True},
        {"$set": {"is_active": False}}
    )
    
    return result.modified_count


# =============================================================================
# AUDIT LOG (Security & Compliance)
# =============================================================================

async def create_audit_log(user_id: str, action: str, details: dict = None) -> str:
    """
    Create an audit log entry.
    
    Tracks all important actions for:
    - Security: detect suspicious activity
    - Compliance: HIPAA requires logging access to medical data
    - Debugging: trace what happened
    
    Args:
        user_id: User who performed the action
        action: Action name (e.g., "login_success", "patient_data_accessed")
        details: Additional details (optional)
        
    Returns:
        Audit log ID
        
    Example:
        await create_audit_log(
            user_id="507f1f77bcf86cd799439011",
            action="patient_data_accessed",
            details={"patient_id": "123", "data_type": "MRI_scan"}
        )
    """
    audit_collection = get_collection(AUDIT_LOGS_COLLECTION)
    
    log_dict = {
        "user_id": ObjectId(user_id),
        "action": action,
        "details": details or {},
        "timestamp": datetime.utcnow(),
        "ip_address": None  # TODO: Add IP tracking in API layer
    }
    
    result = await audit_collection.insert_one(log_dict)
    return str(result.inserted_id)


async def get_user_audit_logs(user_id: str, limit: int = 100) -> List[dict]:
    """
    Get audit logs for a specific user.
    
    Args:
        user_id: User's ID
        limit: Maximum number of logs to return
        
    Returns:
        List of audit log entries
    """
    audit_collection = get_collection(AUDIT_LOGS_COLLECTION)
    
    cursor = audit_collection.find(
        {"user_id": ObjectId(user_id)}
    ).sort("timestamp", -1).limit(limit)
    
    logs = []
    async for log_dict in cursor:
        logs.append({
            "id": str(log_dict["_id"]),
            "action": log_dict["action"],
            "details": log_dict.get("details", {}),
            "timestamp": log_dict["timestamp"]
        })
    
    return logs


# =============================================================================
# ANALYSIS SESSION OPERATIONS (Doctor-Specific Patient Sessions)
# =============================================================================

async def create_analysis_session(
    user_id: str,
    session_id: str,
    patient_info: dict,
    doctor_info: dict,
    upload_dir: str,
    output_dir: str
) -> str:
    """
    Create a new analysis session in the database.
    Links the session to the doctor who created it.
    
    Args:
        user_id: Doctor's user ID
        session_id: Unique session identifier
        patient_info: Patient details (name, age, disorder, etc.)
        doctor_info: Doctor details (name, email, hospital, etc.)
        upload_dir: Directory where files will be uploaded
        output_dir: Directory where results will be stored
        
    Returns:
        MongoDB document ID as string
    """
    from models import SessionInDB
    from datetime import timedelta
    
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    # Create session document
    session_data = SessionInDB(
        session_id=session_id,
        user_id=user_id,
        doctor_id=user_id,  # Same as user_id for doctors
        
        # Patient info
        patient_name=patient_info.get("name", "Unknown"),
        patient_age=patient_info.get("age"),
        patient_weight=patient_info.get("weight"),
        patient_height=patient_info.get("height"),
        patient_disorder=patient_info.get("disorder"),
        patient_description=patient_info.get("description"),
        
        # Doctor info
        doctor_name=doctor_info.get("name", "Unknown"),
        doctor_email=doctor_info.get("email", ""),
        doctor_designation=doctor_info.get("designation"),
        doctor_hospital=doctor_info.get("hospital"),
        
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=24),
        status="created",
        files_uploaded=[],
        has_mesh=False,
        has_report=False,
        upload_dir=upload_dir,
        output_dir=output_dir,
        prediction_path=None
    )
    
    # Convert to dict and insert
    session_dict = session_data.model_dump(by_alias=True, exclude={"id"})
    result = await sessions_collection.insert_one(session_dict)
    
    # Log the action
    await create_audit_log(
        user_id=user_id,
        action="session_created",
        details={
            "session_id": session_id,
            "patient_name": patient_info.get("name")
        }
    )
    
    return str(result.inserted_id)


async def get_session_by_id(session_id: str) -> Optional[dict]:
    """
    Get session details by session_id.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Session dictionary if found, None otherwise
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    session_dict = await sessions_collection.find_one({"session_id": session_id})
    
    if session_dict:
        session_dict["id"] = str(session_dict.pop("_id"))
        return session_dict
    return None


async def get_doctor_sessions(doctor_id: str, limit: int = 50) -> List[dict]:
    """
    Get all sessions created by a specific doctor.
    
    Args:
        doctor_id: Doctor's user ID
        limit: Maximum number of sessions to return
        
    Returns:
        List of session dictionaries
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    cursor = sessions_collection.find(
        {"doctor_id": doctor_id}
    ).sort("created_at", -1).limit(limit)
    
    sessions = []
    async for session_dict in cursor:
        session_dict["id"] = str(session_dict.pop("_id"))
        sessions.append(session_dict)
    
    return sessions


async def update_session_status(
    session_id: str,
    status: str,
    **additional_fields
) -> bool:
    """
    Update session status and other fields.
    
    Args:
        session_id: Session identifier
        status: New status value
        **additional_fields: Other fields to update (e.g., files_uploaded, has_mesh)
        
    Returns:
        True if updated successfully
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    update_data = {"status": status, **additional_fields}
    
    result = await sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": update_data}
    )
    
    return result.modified_count > 0


async def verify_session_ownership(session_id: str, user_id: str) -> bool:
    """
    Verify that a session belongs to the specified user.
    
    Args:
        session_id: Session identifier
        user_id: User ID to verify
        
    Returns:
        True if user owns the session, False otherwise
    """
    sessions_collection = get_collection(SESSIONS_COLLECTION)
    
    session = await sessions_collection.find_one({
        "session_id": session_id,
        "doctor_id": user_id
    })
    
    return session is not None


# =============================================================================
# PATIENT DATA OPERATIONS (Medical Records)
# =============================================================================

async def create_patient_data(
    user_id: str,
    session_id: str,
    patient_name: str,
    patient_info: dict
) -> str:
    """
    Create a patient data record linked to a session.
    
    Args:
        user_id: Doctor/user who created the record
        session_id: Associated session ID
        patient_name: Patient's name
        patient_info: Additional patient information
        
    Returns:
        Patient data ID
    """
    from models import PatientDataInDB
    
    patient_collection = get_collection(PATIENT_DATA_COLLECTION)
    
    patient_data = PatientDataInDB(
        session_id=session_id,
        user_id=user_id,
        doctor_id=user_id,
        patient_name=patient_name,
        patient_age=patient_info.get("age"),
        patient_weight=patient_info.get("weight"),
        patient_height=patient_info.get("height"),
        patient_disorder=patient_info.get("disorder"),
        patient_description=patient_info.get("description"),
        created_at=datetime.utcnow()
    )
    
    # Convert to dict and insert
    data_dict = patient_data.model_dump(by_alias=True, exclude={"id"})
    result = await patient_collection.insert_one(data_dict)
    
    # Log the action
    await create_audit_log(
        user_id=user_id,
        action="patient_data_created",
        details={"patient_data_id": str(result.inserted_id), "session_id": session_id}
    )
    
    return str(result.inserted_id)


async def get_patient_data(patient_data_id: str) -> Optional[dict]:
    """
    Get patient data by ID.
    
    Args:
        patient_data_id: Patient data ID
        
    Returns:
        Patient data dictionary if found
    """
    patient_collection = get_collection(PATIENT_DATA_COLLECTION)
    data_dict = await patient_collection.find_one({"_id": ObjectId(patient_data_id)})
    
    if data_dict:
        data_dict["id"] = str(data_dict.pop("_id"))
        data_dict["user_id"] = str(data_dict["user_id"])
        return data_dict
    return None


async def get_user_patient_data(user_id: str, limit: int = 50) -> List[dict]:
    """
    Get all patient data uploaded by a specific user.
    
    Args:
        user_id: User's ID
        limit: Maximum number of records to return
        
    Returns:
        List of patient data records
    """
    patient_collection = get_collection(PATIENT_DATA_COLLECTION)
    
    cursor = patient_collection.find(
        {"user_id": ObjectId(user_id)}
    ).sort("created_at", -1).limit(limit)
    
    data_list = []
    async for data_dict in cursor:
        data_dict["id"] = str(data_dict.pop("_id"))
        data_dict["user_id"] = str(data_dict["user_id"])
        data_list.append(data_dict)
    
    return data_list


async def update_patient_data_result(
    patient_data_id: str,
    segmentation_result: dict
) -> bool:
    """
    Update patient data with segmentation result.
    
    Args:
        patient_data_id: Patient data ID
        segmentation_result: Dictionary with segmentation results
        
    Returns:
        True if updated successfully
    """
    patient_collection = get_collection(PATIENT_DATA_COLLECTION)
    
    result = await patient_collection.update_one(
        {"_id": ObjectId(patient_data_id)},
        {
            "$set": {
                "segmentation_result": segmentation_result,
                "processed": True
            }
        }
    )
    
    return result.modified_count > 0
