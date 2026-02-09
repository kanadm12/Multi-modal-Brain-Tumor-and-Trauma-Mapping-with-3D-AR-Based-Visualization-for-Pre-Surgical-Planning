import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID 
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : '';

// Store running jobs
const jobStore: Record<string, { jobId: string; status: string; result?: unknown }> = {};

export async function POST(
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

    // Get request body
    const body = await request.json();
    const { patient_info, doctor_info } = body;

    // Get uploaded files from cache
    const uploadData = (global as Record<string, unknown>)[`upload_${sessionId}`] as {
      files: Array<{ filename: string; content: string }>;
    } | undefined;

    if (!uploadData || !uploadData.files) {
      return NextResponse.json(
        { error: 'No files found for this session. Please upload files first.' },
        { status: 400 }
      );
    }

    // Send to RunPod
    const runpodResponse = await fetch(`${RUNPOD_BASE_URL}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({
        input: {
          files: uploadData.files,
          patient_info: patient_info,
          doctor_info: doctor_info,
          options: {
            generate_report: true,
            return_gltf_base64: true,
          },
        },
      }),
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      console.error('RunPod error:', errorText);
      return NextResponse.json(
        { error: `RunPod error: ${runpodResponse.statusText}` },
        { status: runpodResponse.status }
      );
    }

    const runpodResult = await runpodResponse.json();
    
    // Store job info
    jobStore[sessionId] = {
      jobId: runpodResult.id,
      status: runpodResult.status,
    };

    // Also store in global for status endpoint
    (global as Record<string, unknown>)[`job_${sessionId}`] = {
      jobId: runpodResult.id,
      status: 'processing',
      startedAt: new Date().toISOString(),
    };

    return NextResponse.json({
      status: 'success',
      session_id: sessionId,
      message: 'Prediction started',
      task_id: runpodResult.id,
    });

  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Prediction failed' },
      { status: 500 }
    );
  }
}
