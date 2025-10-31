/**
 * API Type Definitions for AdWhisper
 * 
 * Type-safe interfaces for backend API communication
 */

// ==================== Request Types ====================

export interface AdSubmissionRequest {
  text_content?: string;
  image_url?: string;
  video_url?: string;
  image_file?: File;
  video_file?: File;
  metadata?: Record<string, any>;
}

// ==================== Response Types ====================

export interface AdSubmissionResponse {
  request_id: string;
  message: string;
  status: string;
  timestamp: string;
}

export interface AgentStatus {
  name: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
  message?: string;
  timestamp?: string;
}

export interface AnalysisStatus {
  request_id: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
  current_stage: 'submitted' | 'ingestion' | 'analyzing_text' | 'analyzing_visual' | 'scoring' | 'complete';
  message: string;
  timestamp: string;
  agents?: {
    ingestion_agent?: AgentStatus;
    text_bias_agent?: AgentStatus;
    visual_bias_agent?: AgentStatus;
    scoring_agent?: AgentStatus;
  };
}

export interface BiasIssue {
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  source: string;
  description: string;
  examples?: string[];
  confidence: number;
}

export interface Recommendation {
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  action: string;
  impact: string;
}

export interface ScoreBreakdown {
  text_score: number;
  visual_score: number;
  intersectional_penalty: number;
  weighted_score: number;
}

export interface BiasReport {
  request_id: string;
  status: string;
  overall_score?: number;
  assessment?: string;
  score_breakdown?: ScoreBreakdown;
  total_issues?: number;
  high_severity_count?: number;
  medium_severity_count?: number;
  low_severity_count?: number;
  bias_issues?: BiasIssue[];
  top_concerns?: string[];
  recommendations?: Recommendation[];
  similar_cases?: string[];
  benchmark_comparison?: {
    average_score: number;
    percentile: number;
  };
  confidence?: number;
  processing_time_ms?: number;
  timestamp: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  database?: string;
  collections?: number;
  data?: {
    text_patterns: number;
    visual_patterns: number;
    case_studies: number;
  };
  error?: string;
  timestamp: string;
}

// ==================== Error Types ====================

export interface APIError {
  message: string;
  status?: number;
  detail?: string;
}

// ==================== Score Interpretation ====================

export type ScoreLevel = 'critical' | 'warning' | 'caution' | 'good' | 'excellent';

export interface ScoreInterpretation {
  level: ScoreLevel;
  color: string;
  icon: string;
  message: string;
  action: string;
}

export function getScoreInterpretation(score: number): ScoreInterpretation {
  if (score <= 3) {
    return {
      level: 'critical',
      color: 'red',
      icon: '❌',
      message: 'Significant Bias Detected',
      action: 'Major revision required'
    };
  } else if (score <= 6) {
    return {
      level: 'warning',
      color: 'orange',
      icon: '⚠️',
      message: 'Moderate Bias Detected',
      action: 'Revision recommended'
    };
  } else if (score <= 8) {
    return {
      level: 'caution',
      color: 'yellow',
      icon: '⚡',
      message: 'Minor Bias Detected',
      action: 'Minor improvements suggested'
    };
  } else if (score <= 9.5) {
    return {
      level: 'good',
      color: 'green',
      icon: '✅',
      message: 'Minimal Bias',
      action: 'Good to go with minor tweaks'
    };
  } else {
    return {
      level: 'excellent',
      color: 'blue',
      icon: '🎯',
      message: 'Excellent - No Significant Bias',
      action: 'Approved for publication'
    };
  }
}

