import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID 
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : '';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const { sessionId } = await params;
    
    if (!RUNPOD_ENDPOINT_ID || !RUNPOD_API_KEY) {
      return NextResponse.json(
        { error: 'RunPod not configured' },
        { status: 500 }
      );
    }

    // Get job info from global cache
    const jobInfo = (global as Record<string, unknown>)[`job_${sessionId}`] as {
      jobId: string;
      status: string;
      result?: unknown;
    } | undefined;

    if (!jobInfo || !jobInfo.jobId) {
      return NextResponse.json({
        session_id: sessionId,
        status: 'pending',
        progress: 0,
        message: 'Waiting for processing to start...',
        has_mesh: false,
        has_report: false,
      });
    }

    // Check job status on RunPod
    const statusResponse = await fetch(`${RUNPOD_BASE_URL}/status/${jobInfo.jobId}`, {
      headers: {
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      },
    });

    if (!statusResponse.ok) {
      console.error('RunPod status error:', statusResponse.statusText);
      return NextResponse.json({
        session_id: sessionId,
        status: 'processing',
        progress: 50,
        message: 'Processing...',
        has_mesh: false,
        has_report: false,
      });
    }

    const runpodStatus = await statusResponse.json();

    // Map RunPod status to our status format
    let status: 'pending' | 'processing' | 'completed' | 'error' = 'processing';
    let progress = 50;
    let message = 'Processing MRI scans...';

    switch (runpodStatus.status) {
      case 'IN_QUEUE':
        status = 'pending';
        progress = 10;
        message = 'Job queued, waiting for GPU worker...';
        break;
      case 'IN_PROGRESS':
        status = 'processing';
        progress = 50;
        message = 'AI is analyzing MRI scans...';
        break;
      case 'COMPLETED':
        status = 'completed';
        progress = 100;
        message = 'Analysis complete!';
        // Store result
        (global as Record<string, unknown>)[`result_${sessionId}`] = runpodStatus.output;
        break;
      case 'FAILED':
        status = 'error';
        progress = 0;
        message = runpodStatus.error || 'Processing failed';
        break;
      case 'CANCELLED':
        status = 'error';
        progress = 0;
        message = 'Job was cancelled';
        break;
    }

    const result = runpodStatus.output;

    return NextResponse.json({
      session_id: sessionId,
      status,
      progress,
      message,
      has_mesh: status === 'completed' && result?.mesh_data,
      has_report: status === 'completed' && result?.report_html,
      result: status === 'completed' ? result : undefined,
    });

  } catch (error) {
    console.error('Status error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Status check failed' },
      { status: 500 }
    );
  }
}
