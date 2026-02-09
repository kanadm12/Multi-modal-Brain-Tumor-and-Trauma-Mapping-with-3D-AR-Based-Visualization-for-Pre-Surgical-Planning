import { NextRequest, NextResponse } from 'next/server';

// Use server-side env vars (not NEXT_PUBLIC)
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID 
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : '';

export const maxDuration = 300; // 5 minutes max for file processing
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    console.log('RunPod submit called');
    console.log('Endpoint ID:', RUNPOD_ENDPOINT_ID ? 'SET' : 'NOT SET');
    console.log('API Key:', RUNPOD_API_KEY ? 'SET' : 'NOT SET');

    if (!RUNPOD_ENDPOINT_ID || !RUNPOD_API_KEY) {
      return NextResponse.json(
        { error: 'RunPod not configured. Missing RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY' },
        { status: 500 }
      );
    }

    const body = await request.json();
    const { files, patient_info, options } = body;

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: 'No files provided' },
        { status: 400 }
      );
    }

    console.log(`Submitting ${files.length} files to RunPod...`);

    // Submit to RunPod
    const runpodResponse = await fetch(`${RUNPOD_BASE_URL}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({
        input: {
          files,
          patient_info,
          options: {
            generate_report: options?.generate_report ?? true,
            return_gltf_base64: true,
            tta_enabled: options?.tta_enabled ?? false,
          },
        },
      }),
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      console.error('RunPod error:', runpodResponse.status, errorText);
      return NextResponse.json(
        { error: `RunPod error: ${runpodResponse.status} - ${errorText}` },
        { status: runpodResponse.status }
      );
    }

    const result = await runpodResponse.json();
    console.log('RunPod job submitted:', result.id);

    return NextResponse.json({
      jobId: result.id,
      status: result.status,
    });

  } catch (error) {
    console.error('Submit error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to submit job' },
      { status: 500 }
    );
  }
}
