# =============================================================================
# AUTHENTICATION & AUTHORIZATION
# 
# JWT token management, password hashing, and user authentication
# This is the security layer of your application
# =============================================================================

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

from schemas import TokenData, UserResponse
from database import get_collection, USERS_COLLECTION

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"  # Encryption algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Bearer token scheme for FastAPI
# This tells FastAPI to look for "Authorization: Bearer <token>" in requests
security = HTTPBearer()

# =============================================================================
# PASSWORD UTILITIES
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against its hashed version.
    
    Args:
        plain_password: The password the user entered (e.g., "MyPassword123")
        hashed_password: The hashed password from database
        
    Returns:
        True if password matches, False otherwise
        
    Example:
        verify_password("MyPass123", "$2b$12$abc...") → True or False
    """
    # bcrypt requires bytes, and hashed password is stored as string
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """
    Hash a plain text password for secure storage.
    NEVER store plain text passwords!
    
    Args:
        password: Plain text password (e.g., "MyPassword123")
        
    Returns:
        Hashed password (looks like: "$2b$12$abc...")
        
    Example:
        get_password_hash("MyPass123") → "$2b$12$abc..."
    """
    # Generate salt and hash password
    # 12 rounds is a good balance of security and performance
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    # Return as string for storage in MongoDB
    return hashed.decode('utf-8')

# =============================================================================
# JWT TOKEN UTILITIES
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    A JWT token is like a secure ticket that proves who you are.
    It's encrypted and contains user information.
    
    Args:
        data: Dictionary with user info (e.g., {"sub": "user@email.com", "user_id": "123"})
        expires_delta: Optional custom expiration time
        
    Returns:
        JWT token string (looks like: "eyJhbGciOiJIUzI1NiIs...")
        
    The token contains:
        - User information (email, user_id)
        - Expiration time
        - Encrypted signature (so it can't be tampered with)
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Create encrypted token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """
    Decode and verify a JWT token.
    
    Takes the token from the user and:
    1. Decrypts it
    2. Verifies it hasn't been tampered with
    3. Checks it hasn't expired
    4. Returns the user information inside
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData with email and user_id
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Decrypt the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Extract user information
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        
        # Validate we got the required information
        if email is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return TokenData(email=email, user_id=user_id)
        
    except JWTError:
        # Token is invalid or expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# =============================================================================
# AUTHENTICATION DEPENDENCY FOR FASTAPI
# =============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """
    FastAPI dependency to get the current authenticated user.
    
    This function is used in your API endpoints like this:
        @app.get("/protected")
        async def protected_route(user = Depends(get_current_user)):
            return {"user": user.email}
    
    What it does:
    1. Extracts the token from the "Authorization: Bearer <token>" header
    2. Decodes and validates the token
    3. Fetches the full user from database
    4. Returns the user info
    
    If any step fails, it raises 401 Unauthorized error.
    
    Args:
        credentials: Automatically extracted by FastAPI from Authorization header
        
    Returns:
        UserResponse with user information
        
    Raises:
        HTTPException: 401 if token invalid, 403 if user is inactive
    """
    # Get token from Authorization header
    token = credentials.credentials
    
    # Decode token to get user email and ID
    token_data = decode_access_token(token)
    
    # Fetch user from database
    users_collection = get_collection(USERS_COLLECTION)
    user = await users_collection.find_one({"email": token_data.email})
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user account is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Return user information (without password!)
    return UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        is_active=user["is_active"],
        created_at=user["created_at"]
    )


async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """
    Alternative dependency that explicitly checks user is active.
    
    Usage:
        @app.get("/protected")
        async def route(user = Depends(get_current_active_user)):
            # User is guaranteed to be active
            pass
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# =============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC)
# =============================================================================

def require_role(required_roles: list[str]):
    """
    Create a dependency that checks if user has required role.
    
    Usage:
        @app.get("/admin")
        async def admin_route(user = Depends(require_role(["admin"]))):
            # Only admins can access this
            pass
            
        @app.get("/medical")
        async def medical_route(user = Depends(require_role(["doctor", "radiologist"]))):
            # Doctors and radiologists can access this
            pass
    
    Args:
        required_roles: List of allowed roles
        
    Returns:
        Dependency function for FastAPI
    """
    async def role_checker(current_user: UserResponse = Depends(get_current_user)):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {', '.join(required_roles)}"
            )
        return current_user
    
    return role_checker
