// =============================================================================
// SESSIONS API ROUTE
// =============================================================================

import { NextRequest, NextResponse } from 'next/server';
import { getDb } from '@/lib/mongodb';
import { getCurrentUser } from '@/lib/auth';
import { createAuditLog, AuditActions } from '@/lib/audit';

// GET - Get all sessions for the current user
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
    const sessionsCollection = db.collection('sessions');

    // Find all sessions for this user
    const sessions = await sessionsCollection
      .find({ user_id: tokenPayload.userId })
      .sort({ created_at: -1 })
      .toArray();

    // Format sessions for response
    const formattedSessions = sessions.map(session => ({
      session_id: session._id.toString(),
      created_at: session.created_at,
      patient_name: session.patient_name || 'Anonymous',
      patient_age: session.patient_age,
      status: session.status || 'pending',
      has_report: session.has_report || false,
    }));

    return NextResponse.json(formattedSessions);

  } catch (error) {
    console.error('Get sessions error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST - Create a new session
export async function POST(request: NextRequest) {
  try {
    // Get current user from token
    const tokenPayload = await getCurrentUser(request);
    if (!tokenPayload) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    const body = await request.json();
    const { patient_name, patient_age, patient_info, doctor } = body;

    const db = await getDb();
    const sessionsCollection = db.collection('sessions');

    // Create new session
    const newSession = {
      user_id: tokenPayload.userId,
      patient_name: patient_name || patient_info?.name || 'Anonymous',
      patient_age: patient_age || patient_info?.age,
      patient_info: patient_info || {},
      doctor: doctor || {},
      status: 'pending',
      has_report: false,
      has_mesh: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    const result = await sessionsCollection.insertOne(newSession);

    // Also create patient_data entry
    const patientDataCollection = db.collection('patient_data');
    await patientDataCollection.insertOne({
      session_id: result.insertedId.toString(),
      user_id: tokenPayload.userId,
      patient_name: newSession.patient_name,
      patient_age: newSession.patient_age,
      patient_info: newSession.patient_info,
      created_at: new Date().toISOString(),
    });

    // Create audit log
    await createAuditLog(
      tokenPayload.userId,
      AuditActions.SESSION_CREATED,
      { 
        session_id: result.insertedId.toString(),
        patient_name: newSession.patient_name 
      }
    );

    await createAuditLog(
      tokenPayload.userId,
      AuditActions.PATIENT_DATA_CREATED,
      { 
        session_id: result.insertedId.toString(),
        patient_name: newSession.patient_name 
      }
    );

    return NextResponse.json({
      session_id: result.insertedId.toString(),
      status: 'pending',
      created_at: newSession.created_at,
      message: 'Session created successfully',
    }, { status: 201 });

  } catch (error) {
    console.error('Create session error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
