// =============================================================================
// GET CURRENT USER API ROUTE
// =============================================================================

import { NextRequest, NextResponse } from 'next/server';
import { ObjectId } from 'mongodb';
import { getDb } from '@/lib/mongodb';
import { getCurrentUser } from '@/lib/auth';

export async function GET(request: NextRequest) {
  try {
    // Get current user from token
    const tokenPayload = await getCurrentUser(request);
    if (!tokenPayload) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const db = await getDb();
    const usersCollection = db.collection('users');

    // Find user by ID
    const user = await usersCollection.findOne({ 
      _id: new ObjectId(tokenPayload.userId) 
    });

    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    // Return user data (without password)
    return NextResponse.json({
      id: user._id.toString(),
      email: user.email,
      full_name: user.full_name,
      role: user.role,
      hospital: user.hospital || '',
      is_active: user.is_active,
      created_at: user.created_at,
    });

  } catch (error) {
    console.error('Get user error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
