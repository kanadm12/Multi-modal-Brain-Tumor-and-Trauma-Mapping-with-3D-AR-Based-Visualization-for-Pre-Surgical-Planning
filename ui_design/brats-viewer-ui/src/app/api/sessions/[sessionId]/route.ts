// =============================================================================
// SESSION BY ID API ROUTE
// =============================================================================

import { NextRequest, NextResponse } from 'next/server';
import { ObjectId } from 'mongodb';
import { getDb } from '@/lib/mongodb';
import { getCurrentUser } from '@/lib/auth';

interface RouteParams {
  params: Promise<{ sessionId: string }>;
}

// GET - Get session by ID
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { sessionId } = await params;
    
    // Get current user from token
    const tokenPayload = await getCurrentUser(request);
    if (!tokenPayload) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const db = await getDb();
    const sessionsCollection = db.collection('sessions');

    // Find session by ID
    let session;
    try {
      session = await sessionsCollection.findOne({ 
        _id: new ObjectId(sessionId),
        user_id: tokenPayload.userId 
      });
    } catch {
      return NextResponse.json(
        { error: 'Invalid session ID' },
        { status: 400 }
      );
    }

    if (!session) {
      return NextResponse.json(
        { error: 'Session not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      session_id: session._id.toString(),
      created_at: session.created_at,
      patient_name: session.patient_name,
      patient_age: session.patient_age,
      patient_info: session.patient_info,
      doctor: session.doctor,
      status: session.status,
      has_report: session.has_report,
      has_mesh: session.has_mesh,
      mesh_data: session.mesh_data,
      report: session.report,
    });

  } catch (error) {
    console.error('Get session error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// PATCH - Update session
export async function PATCH(request: NextRequest, { params }: RouteParams) {
  try {
    const { sessionId } = await params;
    
    // Get current user from token
    const tokenPayload = await getCurrentUser(request);
    if (!tokenPayload) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const body = await request.json();
    
    const db = await getDb();
    const sessionsCollection = db.collection('sessions');

    // Update session
    const updateData = {
      ...body,
      updated_at: new Date().toISOString(),
    };

    // Remove fields that shouldn't be updated directly
    delete updateData.user_id;
    delete updateData._id;
    delete updateData.session_id;

    let result;
    try {
      result = await sessionsCollection.updateOne(
        { 
          _id: new ObjectId(sessionId),
          user_id: tokenPayload.userId 
        },
        { $set: updateData }
      );
    } catch {
      return NextResponse.json(
        { error: 'Invalid session ID' },
        { status: 400 }
      );
    }

    if (result.matchedCount === 0) {
      return NextResponse.json(
        { error: 'Session not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      message: 'Session updated successfully',
      session_id: sessionId,
    });

  } catch (error) {
    console.error('Update session error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// DELETE - Delete session
export async function DELETE(request: NextRequest, { params }: RouteParams) {
  try {
    const { sessionId } = await params;
    
    // Get current user from token
    const tokenPayload = await getCurrentUser(request);
    if (!tokenPayload) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const db = await getDb();
    const sessionsCollection = db.collection('sessions');

    let result;
    try {
      result = await sessionsCollection.deleteOne({ 
        _id: new ObjectId(sessionId),
        user_id: tokenPayload.userId 
      });
    } catch {
      return NextResponse.json(
        { error: 'Invalid session ID' },
        { status: 400 }
      );
    }

    if (result.deletedCount === 0) {
      return NextResponse.json(
        { error: 'Session not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      message: 'Session deleted successfully',
    });

  } catch (error) {
    console.error('Delete session error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
