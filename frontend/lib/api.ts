/**
 * API Client for AdWhisper Backend
 * 
 * Centralized API communication layer with type-safe methods
 */

import type {
  AdSubmissionResponse,
  AnalysisStatus,
  BiasReport,
  HealthResponse,
  APIError
} from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Base fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error: APIError = {
        message: `API Error: ${response.statusText}`,
        status: response.status,
      };

      try {
        const errorData = await response.json();
        error.detail = errorData.detail || errorData.message;
      } catch {
        // Ignore JSON parse errors
      }

      throw error;
    }

    return await response.json();
  } catch (error) {
    if (error && typeof error === 'object' && 'message' in error) {
      throw error;
    }

    throw {
      message: 'Network error or server unreachable',
      detail: String(error),
    } as APIError;
  }
}

/**
 * Submit an ad for bias analysis
 * 
 * @param textContent - Ad text/copy
 * @param imageFile - Image file (optional)
 * @param videoFile - Video file (optional)
 * @param imageUrl - Image URL (optional)
 * @param videoUrl - Video URL (optional)
 * @param metadata - Additional metadata (optional)
 * @returns Submission response with request_id
 */
export async function submitAd(params: {
  textContent?: string;
  imageFile?: File;
  videoFile?: File;
  imageUrl?: string;
  videoUrl?: string;
  metadata?: Record<string, any>;
}): Promise<AdSubmissionResponse> {
  const formData = new FormData();

  if (params.textContent) {
    formData.append('text_content', params.textContent);
  }

  if (params.imageFile) {
    formData.append('image_file', params.imageFile);
  }

  if (params.videoFile) {
    formData.append('video_file', params.videoFile);
  }

  if (params.imageUrl) {
    formData.append('image_url', params.imageUrl);
  }

  if (params.videoUrl) {
    formData.append('video_url', params.videoUrl);
  }

  if (params.metadata) {
    formData.append('metadata', JSON.stringify(params.metadata));
  }

  return fetchAPI<AdSubmissionResponse>('/api/analyze-ad', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Check the status of an analysis request
 * 
 * @param requestId - The request ID from submission
 * @returns Current analysis status
 */
export async function checkStatus(requestId: string): Promise<AnalysisStatus> {
  return fetchAPI<AnalysisStatus>(`/api/status/${requestId}`);
}

/**
 * Get the complete analysis results
 * 
 * @param requestId - The request ID from submission
 * @returns Complete bias report
 */
export async function getResults(requestId: string): Promise<BiasReport> {
  return fetchAPI<BiasReport>(`/api/results/${requestId}`);
}

/**
 * Check backend health status
 * 
 * @returns Health status and database info
 */
export async function getHealth(): Promise<HealthResponse> {
  return fetchAPI<HealthResponse>('/health');
}

/**
 * Poll for analysis completion
 * 
 * Polls the status endpoint at regular intervals until analysis is complete
 * 
 * @param requestId - The request ID from submission
 * @param onUpdate - Callback for status updates
 * @param interval - Polling interval in milliseconds (default: 5000)
 * @returns Promise that resolves when analysis is complete
 */
export async function pollUntilComplete(
  requestId: string,
  onUpdate: (status: AnalysisStatus) => void,
  interval: number = 5000
): Promise<BiasReport> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await checkStatus(requestId);
        onUpdate(status);

        if (status.status === 'complete') {
          // Fetch final results
          const results = await getResults(requestId);
          resolve(results);
        } else if (status.status === 'error') {
          reject(new Error(status.message));
        } else {
          // Continue polling
          setTimeout(poll, interval);
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
}

/**
 * Validate file before upload
 * 
 * @param file - File to validate
 * @param maxSizeMB - Maximum file size in MB (default: 50MB)
 * @returns Validation result
 */
export function validateFile(file: File, maxSizeMB: number = 50): { valid: boolean; error?: string } {
  const maxSize = maxSizeMB * 1024 * 1024; // Convert to bytes

  if (file.size > maxSize) {
    return {
      valid: false,
      error: `File size exceeds ${maxSizeMB}MB limit`
    };
  }

  const imageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
  const videoTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
  const validTypes = [...imageTypes, ...videoTypes];

  if (!validTypes.includes(file.type)) {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload an image (JPG, PNG, GIF, WEBP) or video (MP4, MOV, AVI, WEBM)'
    };
  }

  return { valid: true };
}

/**
 * Format file size for display
 * 
 * @param bytes - File size in bytes
 * @returns Formatted string (e.g., "2.5 MB")
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

