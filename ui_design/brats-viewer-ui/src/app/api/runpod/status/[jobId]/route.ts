import { NextRequest, NextResponse } from 'next/server';

const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID 
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : '';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  try {
    const { jobId } = await params;

    if (!RUNPOD_ENDPOINT_ID || !RUNPOD_API_KEY) {
      return NextResponse.json(
        { error: 'RunPod not configured' },
        { status: 500 }
      );
    }

    const response = await fetch(`${RUNPOD_BASE_URL}/status/${jobId}`, {
      headers: {
        'Authorization': `Bearer ${RUNPOD_API_KEY}`,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('RunPod status error:', response.status, errorText);
      return NextResponse.json(
        { error: `Failed to get status: ${response.status}` },
        { status: response.status }
      );
    }

    const status = await response.json();
    return NextResponse.json(status);

  } catch (error) {
    console.error('Status error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to get status' },
      { status: 500 }
    );
  }
}
