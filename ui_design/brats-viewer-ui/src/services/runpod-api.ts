// =============================================================================
// RUNPOD SERVERLESS API SERVICE
// 
// Handles communication with RunPod GPU serverless endpoint for inference
// Falls back to local API if RunPod is not configured
// =============================================================================

// RunPod configuration from environment
const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';

// RunPod API base URL
const RUNPOD_BASE_URL = RUNPOD_ENDPOINT_ID 
  ? `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`
  : '';

// =============================================================================
// INTERFACES
// =============================================================================

export interface RunPodJobInput {
  files: Array<{
    filename: string;
    content: string; // base64 encoded
  }>;
  patient_info?: {
    name?: string;
    age?: string;
    id?: string;
  };
  options?: {
    generate_report?: boolean;
    return_gltf_base64?: boolean;
    tta_enabled?: boolean;
  };
}

export interface RunPodJobResponse {
  id: string;
  status: 'IN_QUEUE' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  delayTime?: number;
  executionTime?: number;
  output?: RunPodInferenceOutput;
  error?: string;
}

export interface RunPodInferenceOutput {
  status: 'success' | 'error';
  error?: string;
  traceback?: string;
  
  // Tumor data
  tumor_stats: Record<string, {
    class_id: number;
    label: string;
    volume_voxels: number;
    volume_mm3: number;
    volume_cm3: number;
    centroid: number[];
    bounding_box: {
      min: number[];
      max: number[];
      dimensions: number[];
    };
    color: string;
  }>;
  volumes: {
    ncr: number;
    ed: number;
    et: number;
    total: number;
  };
  center_of_mass: number[] | null;
  bounding_box: {
    min: number[];
    max: number[];
  } | null;
  
  // Mesh data for Three.js
  mesh_data: {
    brain_mesh: MeshData | null;
    tumor_meshes: Record<string, MeshData>;
  };
  
  // GLTF model (base64)
  gltf_base64?: string;
  gltf_bin_base64?: string;
  
  // Report
  report_html?: string;
  report_error?: string;
}

export interface MeshData {
  vertices: number[][];
  faces: number[][];
  normals?: number[][];
  color: number[];
  hex_color?: string;
  opacity: number;
  vertex_count: number;
  face_count: number;
}

// =============================================================================
// RUNPOD API SERVICE CLASS
// =============================================================================

class RunPodApiService {
  private endpointId: string;
  private apiKey: string;
  private baseUrl: string;

  constructor() {
    this.endpointId = RUNPOD_ENDPOINT_ID;
    this.apiKey = RUNPOD_API_KEY;
    this.baseUrl = RUNPOD_BASE_URL;
  }

  /**
   * Check if RunPod is configured
   */
  isConfigured(): boolean {
    return !!(this.endpointId && this.apiKey);
  }

  /**
   * Get headers for RunPod API
   */
  private getHeaders(): HeadersInit {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
    };
  }

  /**
   * Convert File to base64
   */
  private async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix if present
        const base64 = result.includes(',') ? result.split(',')[1] : result;
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  /**
   * Submit inference job to RunPod
   */
  async submitJob(
    files: File[], 
    patientInfo?: { name?: string; age?: string; id?: string },
    options?: { generate_report?: boolean; tta_enabled?: boolean }
  ): Promise<{ jobId: string }> {
    if (!this.isConfigured()) {
      throw new Error('RunPod is not configured. Set NEXT_PUBLIC_RUNPOD_ENDPOINT_ID and NEXT_PUBLIC_RUNPOD_API_KEY');
    }

    // Convert files to base64
    const fileData = await Promise.all(
      files.map(async (file) => ({
        filename: file.name,
        content: await this.fileToBase64(file),
      }))
    );

    const input: RunPodJobInput = {
      files: fileData,
      patient_info: patientInfo,
      options: {
        generate_report: options?.generate_report ?? true,
        return_gltf_base64: true,
        tta_enabled: options?.tta_enabled ?? true,
      },
    };

    const response = await fetch(`${this.baseUrl}/run`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(error.error || error.message || 'Failed to submit job');
    }

    const data = await response.json();
    return { jobId: data.id };
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<RunPodJobResponse> {
    if (!this.isConfigured()) {
      throw new Error('RunPod is not configured');
    }

    const response = await fetch(`${this.baseUrl}/status/${jobId}`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Poll job until completion
   */
  async pollJob(
    jobId: string,
    onProgress?: (status: RunPodJobResponse) => void,
    intervalMs: number = 2000,
    timeoutMs: number = 300000 // 5 minutes
  ): Promise<RunPodInferenceOutput> {
    const startTime = Date.now();

    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          // Check timeout
          if (Date.now() - startTime > timeoutMs) {
            reject(new Error('Job timed out'));
            return;
          }

          const status = await this.getJobStatus(jobId);

          if (onProgress) {
            onProgress(status);
          }

          switch (status.status) {
            case 'COMPLETED':
              if (status.output?.status === 'error') {
                reject(new Error(status.output.error || 'Inference failed'));
              } else {
                resolve(status.output!);
              }
              break;
            
            case 'FAILED':
            case 'CANCELLED':
              reject(new Error(status.error || `Job ${status.status.toLowerCase()}`));
              break;
            
            case 'IN_QUEUE':
            case 'IN_PROGRESS':
            default:
              setTimeout(poll, intervalMs);
              break;
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }

  /**
   * Run full inference pipeline
   */
  async runInference(
    files: File[],
    patientInfo?: { name?: string; age?: string; id?: string },
    options?: { generate_report?: boolean; tta_enabled?: boolean },
    onProgress?: (status: RunPodJobResponse) => void
  ): Promise<RunPodInferenceOutput> {
    // Submit job
    const { jobId } = await this.submitJob(files, patientInfo, options);
    
    // Poll until complete
    return this.pollJob(jobId, onProgress);
  }

  /**
   * Convert base64 GLTF to Blob URL for loading
   */
  createGLTFBlobUrl(gltfBase64: string, binBase64?: string): { gltfUrl: string; binUrl?: string } {
    // Decode GLTF JSON
    const gltfBytes = Uint8Array.from(atob(gltfBase64), c => c.charCodeAt(0));
    const gltfBlob = new Blob([gltfBytes], { type: 'model/gltf+json' });
    const gltfUrl = URL.createObjectURL(gltfBlob);

    let binUrl: string | undefined;
    if (binBase64) {
      const binBytes = Uint8Array.from(atob(binBase64), c => c.charCodeAt(0));
      const binBlob = new Blob([binBytes], { type: 'application/octet-stream' });
      binUrl = URL.createObjectURL(binBlob);
    }

    return { gltfUrl, binUrl };
  }

  /**
   * Health check for RunPod endpoint
   */
  async healthCheck(): Promise<{ status: string }> {
    if (!this.isConfigured()) {
      return { status: 'not_configured' };
    }

    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        headers: this.getHeaders(),
      });
      
      if (response.ok) {
        return { status: 'healthy' };
      }
      return { status: 'unhealthy' };
    } catch {
      return { status: 'unreachable' };
    }
  }
}

// Export singleton
export const runpodApi = new RunPodApiService();
export default runpodApi;
