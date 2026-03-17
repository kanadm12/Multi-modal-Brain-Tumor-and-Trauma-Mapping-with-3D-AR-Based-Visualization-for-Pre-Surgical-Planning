# =============================================================================
# DATABASE CONNECTION - MONGODB
# 
# Handles MongoDB connection and database operations
# =============================================================================

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# MongoDB Configuration from environment variables
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://Kanad%20M:Kanad1204@cluster0.rrnad.mongodb.net/?appName=Cluster0")
DATABASE_NAME = os.getenv("DATABASE_NAME", "brats_medical_db")

class MongoDB:
    """Singleton class to hold MongoDB client"""
    client: AsyncIOMotorClient = None
    
mongodb = MongoDB()  # Single instance

async def connect_to_mongo():
    """Connect to MongoDB - call this on app startup"""
    try:
        mongodb.client = AsyncIOMotorClient(MONGODB_URL)
        # Test connection
        await mongodb.client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {MONGODB_URL}")
    except ConnectionFailure as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close MongoDB connection - call this on app shutdown"""
    if mongodb.client:
        mongodb.client.close()
        print("🔌 Closed MongoDB connection")

def get_database():
    """Get database instance"""
    return mongodb.client[DATABASE_NAME]

def get_collection(collection_name: str):
    """Get a collection from the database"""
    db = get_database()
    return db[collection_name]

# Collection names - like table names in SQL
USERS_COLLECTION = "users"
SESSIONS_COLLECTION = "sessions"
PATIENT_DATA_COLLECTION = "patient_data"
AUDIT_LOGS_COLLECTION = "audit_logs"
