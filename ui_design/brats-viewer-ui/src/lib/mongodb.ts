// =============================================================================
// MONGODB CONNECTION FOR NEXT.JS API ROUTES
// =============================================================================

import { MongoClient, Db } from 'mongodb';

const MONGODB_URL = process.env.MONGODB_URL || 'mongodb://localhost:27017';
const DB_NAME = process.env.MONGODB_DB_NAME || 'brats_db';

if (!MONGODB_URL) {
  throw new Error('Please define the MONGODB_URL environment variable');
}

// Global cache for MongoDB client (for serverless functions)
let cachedClient: MongoClient | null = null;
let cachedDb: Db | null = null;

export async function connectToDatabase(): Promise<{ client: MongoClient; db: Db }> {
  // Return cached connection if available
  if (cachedClient && cachedDb) {
    return { client: cachedClient, db: cachedDb };
  }

  // Create new connection
  const client = new MongoClient(MONGODB_URL);
  await client.connect();
  const db = client.db(DB_NAME);

  // Cache the connection
  cachedClient = client;
  cachedDb = db;

  console.log('✅ Connected to MongoDB');
  return { client, db };
}

export async function getDb(): Promise<Db> {
  const { db } = await connectToDatabase();
  return db;
}
