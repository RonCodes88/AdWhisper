"""
Simplified Shared Models for AdWhisper Multi-Agent System

Following Fetch.ai uAgents standards:
- Uses uagents.Model (not BaseModel)
- Simple, clean message passing
- Minimal Field usage (avoids serialization issues)
- Defaults in __init__ only
"""

from uagents import Model
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ContentType(str, Enum):
    """Type of content being analyzed"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class BiasType(str, Enum):
    """Types of bias that can be detected"""
    GENDER = "gender_bias"
    RACIAL = "racial_bias"
    AGE = "age_bias"
    SOCIOECONOMIC = "socioeconomic_bias"
    DISABILITY = "disability_bias"
    LGBTQ = "lgbtq_bias"
    REPRESENTATION = "representation_bias"
    CONTEXTUAL = "contextual_bias"


class SeverityLevel(str, Enum):
    """Severity levels for bias instances"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# INGESTION AGENT MESSAGES
# ============================================================================

class TextAnalysisRequest(Model):
    """Request from Ingestion Agent to Text Bias Agent"""
    request_id: str
    text_content: str
    metadata: Optional[Dict[str, Any]]
    timestamp: str

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        if 'metadata' not in data:
            data['metadata'] = {}
        super().__init__(**data)


class VisualAnalysisRequest(Model):
    """Request from Ingestion Agent to Visual Bias Agent"""
    request_id: str
    frames_base64: List[str]
    num_frames: int
    metadata: Optional[Dict[str, Any]]
    timestamp: str

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        if 'metadata' not in data:
            data['metadata'] = {}
        super().__init__(**data)


# ============================================================================
# BIAS AGENT TO SCORING AGENT MESSAGES
# ============================================================================

class TextBiasReport(Model):
    """Report from Text Bias Agent to Scoring Agent"""
    request_id: str
    agent_name: str
    bias_detected: bool
    bias_instances: List[Dict[str, Any]]
    text_score: float
    recommendations: List[str]
    timestamp: str

    def __init__(self, **data):
        if 'agent_name' not in data:
            data['agent_name'] = "text_bias_agent"
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        if 'bias_instances' not in data:
            data['bias_instances'] = []
        if 'recommendations' not in data:
            data['recommendations'] = []
        super().__init__(**data)


class VisualBiasReport(Model):
    """Report from Visual Bias Agent to Scoring Agent"""
    request_id: str
    agent_name: str
    bias_detected: bool
    bias_instances: List[Dict[str, Any]]
    visual_score: float
    diversity_metrics: Optional[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str

    def __init__(self, **data):
        if 'agent_name' not in data:
            data['agent_name'] = "visual_bias_agent"
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        if 'bias_instances' not in data:
            data['bias_instances'] = []
        if 'recommendations' not in data:
            data['recommendations'] = []
        if 'diversity_metrics' not in data:
            data['diversity_metrics'] = None
        super().__init__(**data)


# ============================================================================
# FINAL REPORT (SCORING AGENT OUTPUT)
# ============================================================================

class FinalBiasReport(Model):
    """Final aggregated report from Scoring Agent"""
    request_id: str
    overall_score: float
    assessment: str
    text_score: float
    visual_score: float
    total_issues: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    text_analysis: Dict[str, Any]
    visual_analysis: Dict[str, Any]
    recommendations: List[str]
    benchmark: Optional[Dict[str, Any]]
    confidence: float
    timestamp: str

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        if 'recommendations' not in data:
            data['recommendations'] = []
        if 'benchmark' not in data:
            data['benchmark'] = None
        super().__init__(**data)


# ============================================================================
# ERROR HANDLING
# ============================================================================

class AgentError(Model):
    """Error message from any agent"""
    request_id: str
    agent_name: str
    error_type: str
    error_message: str
    timestamp: str

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_bias_instance_dict(
    bias_type: str,
    severity: str,
    examples: List[str],
    context: str,
    confidence: float
) -> Dict[str, Any]:
    """Helper to create bias instance dictionary"""
    return {
        "bias_type": bias_type,
        "severity": severity,
        "examples": examples,
        "context": context,
        "confidence": confidence
    }
