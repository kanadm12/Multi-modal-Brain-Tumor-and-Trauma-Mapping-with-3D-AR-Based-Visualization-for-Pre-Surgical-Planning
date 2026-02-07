// =============================================================================
// API SERVICE
// 
// Handles communication with the BraTS inference backend
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// =============================================================================
// AUTHENTICATION INTERFACES
// =============================================================================

export interface UserData {
  id: string;
  email: string;
  full_name: string;
  role: string;
  hospital?: string;
  is_active: boolean;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserData;
}

export interface SignupData {
  email: string;
  password: string;
  full_name: string;
  role: string;
  hospital?: string;
}

export interface LoginData {
  email: string;
  password: string;
}

// =============================================================================
// SESSION INTERFACES
// =============================================================================

export interface SessionResponse {
  session_id: string;
  created_at: string;
  expires_at: string;
}

export interface UploadResponse {
  status: string;
  session_id: string;
  files_uploaded: number;
  modalities_found: string[];
  message: string;
}

export interface PredictionResponse {
  status: string;
  session_id: string;
  message: string;
  task_id?: string;
}

export interface StatusResponse {
  session_id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: number;
  message: string;
  has_mesh: boolean;
  has_report: boolean;
}

export interface TumorStats {
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
}

export interface MeshData {
  vertices: number[][];
  faces: number[][];
  normals?: number[][];
  color: number[];
  opacity: number;
  vertex_count: number;
  face_count: number;
}

export interface MeshResponse {
  session_id: string;
  brain_mesh: MeshData | null;
  tumor_meshes: Record<string, MeshData>;
  tumor_stats: Record<string, TumorStats>;
  center_of_mass: number[] | null;
  bounding_box: {
    min: number[];
    max: number[];
  } | null;
}

export interface ReportResponse {
  report_id: string;
  generated_at: string;
  hospital_name: string;
  patient: {
    name: string;
    id: string;
    age: string;
    gender: string;
  };
  doctor: {
    name: string;
    department: string;
    credentials: string;
  };
  tumor_analysis: {
    classes: Record<string, TumorStats & { full_name: string; description: string; clinical_significance: string }>;
    whole_tumor_volume_cm3: number;
    tumor_core_volume_cm3: number;
    enhancing_tumor_volume_cm3: number;
    estimated_location: {
      region: string;
      hemisphere: string;
      coordinates: {
        axial_slice: number;
        coronal_slice: number;
        sagittal_slice: number;
      };
    };
    estimated_grade: {
      grade: string;
      confidence: string;
      description: string;
    };
  };
  clinical_findings: string[];
  recommendations: string[];
  disclaimer: string;
}

class ApiService {
  private baseUrl: string;
  private token: string | null = null;

  constructor() {
    this.baseUrl = API_BASE_URL;
    // Load token from localStorage on initialization
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('auth_token');
    }
  }

  /**
   * Set authentication token
   */
  setToken(token: string | null) {
    this.token = token;
    if (typeof window !== 'undefined') {
      if (token) {
        localStorage.setItem('auth_token', token);
      } else {
        localStorage.removeItem('auth_token');
      }
    }
  }

  /**
   * Get current token
   */
  getToken(): string | null {
    return this.token;
  }

  /**
   * Get authentication headers
   */
  private getAuthHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  /**
   * Get auth headers for FormData (no Content-Type)
   */
  private getAuthHeadersOnly(): HeadersInit {
    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    return headers;
  }

  // =============================================================================
  // AUTHENTICATION METHODS
  // =============================================================================

  /**
   * Sign up a new user
   */
  async signup(data: SignupData): Promise<UserData> {
    const response = await fetch(`${this.baseUrl}/api/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    return response.json();
  }

  /**
   * Login with email and password
   */
  async login(data: LoginData): Promise<AuthResponse> {
    const response = await fetch(`${this.baseUrl}/api/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const authData: AuthResponse = await response.json();
    this.setToken(authData.access_token);
    return authData;
  }

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<UserData> {
    const response = await fetch(`${this.baseUrl}/api/me`, {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to get user info');
    }

    return response.json();
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/api/logout`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
      });
    } finally {
      this.setToken(null);
    }
  }

  // =============================================================================
  // SESSION METHODS
  // =============================================================================

  /**
   * Create a new session for inference
   */
  async createSession(data: {
    patient: {
      name: string;
      age?: string;
      weight?: string;
      height?: string;
      disorder?: string;
      description?: string;
    };
    doctor: {
      name: string;
      email: string;
      designation: string;
      hospital: string;
    };
  }): Promise<SessionResponse> {
    const response = await fetch(`${this.baseUrl}/api/session/create`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Upload MRI files to the session
   */
  async uploadFiles(sessionId: string, files: File[]): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${this.baseUrl}/api/upload/${sessionId}`, {
      method: 'POST',
      headers: this.getAuthHeadersOnly(),  // Don't set Content-Type for FormData
      body: formData,
    });

    if (!response.ok) {
      console.error('Upload response status:', response.status, response.statusText);
      const responseText = await response.text();
      console.error('Upload response text:', responseText);
      
      let error;
      try {
        error = JSON.parse(responseText);
      } catch {
        error = { detail: responseText || response.statusText };
      }
      
      console.error('Upload error:', error);
      const errorMessage = typeof error.detail === 'string' 
        ? error.detail 
        : (Array.isArray(error.detail) ? error.detail.map((e: { msg: string }) => e.msg).join(', ') : JSON.stringify(error.detail)) || error.message || 'Upload failed';
      throw new Error(errorMessage);
    }

    return response.json();
  }

  /**
   * Start inference prediction
   */
  async startPrediction(
    sessionId: string, 
    patientInfo?: { name?: string; id?: string; age?: string; gender?: string },
    doctorInfo?: { name?: string; department?: string; credentials?: string }
  ): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/api/predict/${sessionId}`, {
      method: 'POST',
      headers: {
        ...this.getAuthHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patient_info: patientInfo,
        doctor_info: doctorInfo,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
  }

  /**
   * Get processing status
   */
  async getStatus(sessionId: string): Promise<StatusResponse> {
    const response = await fetch(`${this.baseUrl}/api/status/${sessionId}`, {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Poll status until complete
   */
  async pollStatus(
    sessionId: string, 
    onProgress?: (status: StatusResponse) => void,
    intervalMs: number = 2000
  ): Promise<StatusResponse> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getStatus(sessionId);
          
          if (onProgress) {
            onProgress(status);
          }

          if (status.status === 'completed') {
            resolve(status);
          } else if (status.status === 'error') {
            reject(new Error(status.message));
          } else {
            setTimeout(poll, intervalMs);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }

  /**
   * Get mesh data for 3D visualization
   */
  async getMeshData(sessionId: string): Promise<MeshResponse> {
    const response = await fetch(`${this.baseUrl}/api/mesh/${sessionId}`);

    if (!response.ok) {
      throw new Error(`Failed to get mesh data: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get clinical report
   */
  async getReport(sessionId: string): Promise<ReportResponse> {
    const response = await fetch(`${this.baseUrl}/api/report/${sessionId}`);

    if (!response.ok) {
      throw new Error(`Failed to get report: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Download PDF report
   */
  async downloadReportPDF(sessionId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/report/${sessionId}/pdf`, {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to download report: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * Download GLTF model for AR
   */
  async downloadGLTF(sessionId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/download/gltf/${sessionId}`);

    if (!response.ok) {
      throw new Error(`Failed to download GLTF: ${response.statusText}`);
    }

    return response.blob();
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; gpu_available: boolean }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  /**
   * Get all sessions for the logged-in doctor
   */
  async getUserSessions(): Promise<SessionResponse[]> {
    const response = await fetch(
      `${this.baseUrl}/api/sessions`,
      {
        method: 'GET',
        headers: this.getAuthHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch sessions: ${response.statusText}`);
    }

    return response.json();
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
