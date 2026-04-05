import { NextRequest, NextResponse } from 'next/server';

// Use server-side env vars (fallback to NEXT_PUBLIC for backwards compatibility)
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';

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

    // Get form data with files
    const formData = await request.formData();
    const files = formData.getAll('files') as File[];

    if (files.length === 0) {
      return NextResponse.json(
        { error: 'No files uploaded' },
        { status: 400 }
      );
    }

    // Convert files to base64 for RunPod
    const fileDataPromises = files.map(async (file) => {
      const buffer = await file.arrayBuffer();
      const base64 = Buffer.from(buffer).toString('base64');
      return {
        filename: file.name,
        content: base64,
      };
    });

    const fileData = await Promise.all(fileDataPromises);

    // Store in session storage (in production, use a database)
    // For now, we'll store the file data temporarily
    const uploadData = {
      session_id: sessionId,
      files: fileData,
      uploaded_at: new Date().toISOString(),
    };

    // Store in global cache (temporary solution)
    if (typeof global !== 'undefined') {
      (global as Record<string, unknown>)[`upload_${sessionId}`] = uploadData;
    }

    return NextResponse.json({
      status: 'success',
      session_id: sessionId,
      files_uploaded: files.length,
      modalities_found: files.map(f => f.name),
      message: `Successfully uploaded ${files.length} files`,
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Upload failed' },
      { status: 500 }
    );
  }
}
