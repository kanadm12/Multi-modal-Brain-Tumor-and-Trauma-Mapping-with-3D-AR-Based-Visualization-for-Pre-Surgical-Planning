// =============================================================================
// AUDIT LOG UTILITY FOR NEXT.JS API ROUTES
// =============================================================================

import { getDb } from './mongodb';

export interface AuditLogEntry {
  user_id: string;
  action: string;
  details?: Record<string, unknown>;
  timestamp: string;
  ip_address?: string;
}

/**
 * Create an audit log entry
 * Tracks all important actions for security and compliance
 */
export async function createAuditLog(
  userId: string,
  action: string,
  details?: Record<string, unknown>,
  ipAddress?: string
): Promise<string> {
  try {
    const db = await getDb();
    const auditCollection = db.collection('audit_logs');

    const logEntry = {
      user_id: userId,
      action,
      details: details || {},
      timestamp: new Date().toISOString(),
      ip_address: ipAddress || null,
    };

    const result = await auditCollection.insertOne(logEntry);
    return result.insertedId.toString();
  } catch (error) {
    console.error('Failed to create audit log:', error);
    // Don't throw - audit logging shouldn't break the main operation
    return '';
  }
}

/**
 * Common audit actions
 */
export const AuditActions = {
  USER_CREATED: 'user_created',
  LOGIN_SUCCESS: 'login_success',
  LOGIN_FAILED: 'login_failed',
  LOGOUT: 'logout',
  SESSION_CREATED: 'session_created',
  SESSION_ACCESSED: 'session_accessed',
  SESSION_UPDATED: 'session_updated',
  SESSION_DELETED: 'session_deleted',
  PATIENT_DATA_CREATED: 'patient_data_created',
  PATIENT_DATA_ACCESSED: 'patient_data_accessed',
  INFERENCE_STARTED: 'inference_started',
  INFERENCE_COMPLETED: 'inference_completed',
  REPORT_GENERATED: 'report_generated',
  REPORT_DOWNLOADED: 'report_downloaded',
} as const;
