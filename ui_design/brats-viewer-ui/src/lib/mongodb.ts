// =============================================================================
// MONGODB CONNECTION FOR NEXT.JS API ROUTES
// =============================================================================

import { MongoClient, Db } from 'mongodb';

const MONGODB_URL = process.env.MONGODB_URL;
const DB_NAME = process.env.MONGODB_DB_NAME || 'brats_db';

if (!MONGODB_URL) {
  console.error('❌ MONGODB_URL environment variable is not set!');
}

// Global cache for MongoDB client (for serverless functions)
let cachedClient: MongoClient | null = null;
let cachedDb: Db | null = null;

export async function connectToDatabase(): Promise<{ client: MongoClient; db: Db }> {
  if (!MONGODB_URL) {
    throw new Error('MONGODB_URL environment variable is not configured. Please add it to your Vercel project settings.');
  }

  // Return cached connection if available
  if (cachedClient && cachedDb) {
    return { client: cachedClient, db: cachedDb };
  }

  try {
    // Create new connection
    const client = new MongoClient(MONGODB_URL);
    await client.connect();
    const db = client.db(DB_NAME);

    // Cache the connection
    cachedClient = client;
    cachedDb = db;

    console.log('✅ Connected to MongoDB');
    return { client, db };
  } catch (error) {
    console.error('❌ MongoDB connection error:', error);
    throw error;
  }
}

export async function getDb(): Promise<Db> {
  const { db } = await connectToDatabase();
  return db;
}
