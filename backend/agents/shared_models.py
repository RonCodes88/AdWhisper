"""
Shared Message Models for Agent Communication

All agents use these models for consistent message passing.
"""

from uagents import Model
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from enum import Enum


# Content type enumeration
class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MIXED = "mixed"


class BiasCategory(str, Enum):
    GENDER = "gender_bias"
    RACIAL = "racial_bias"
    AGE = "age_bias"
    SOCIOECONOMIC = "socioeconomic_bias"
    DISABILITY = "disability_bias"
    LGBTQ = "lgbtq_bias"
    REPRESENTATION = "representation_bias"


# ============================================================================
# INGESTION AGENT MESSAGES
# ============================================================================

class AdContentRequest(Model):
    """Incoming ad content for analysis"""
    request_id: str
    content_type: ContentType
    text_content: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class EmbeddingPackage(Model):
    """Processed content with embeddings - sent to analysis agents"""
    request_id: str
    text_content: Optional[str] = None  # Original text for context
    text_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    frames_base64: Optional[List[str]] = None  # Base64-encoded video frames
    chromadb_collection_id: Optional[str] = None  # Optional - agents can use request_id if None
    content_type: ContentType
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class IngestionAcknowledgement(Model):
    """Acknowledgement of content ingestion"""
    request_id: str
    status: str
    message: str
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# ============================================================================
# BIAS DETECTION MESSAGES
# ============================================================================

class BiasInstance(Model):
    """Individual bias detection instance"""
    bias_type: BiasCategory
    severity: str  # "low", "medium", "high"
    examples: List[str]
    context: str
    confidence: float  # 0.0 to 1.0


class TextBiasReport(Model):
    """Text Bias Agent report"""
    request_id: str
    agent_name: str = "text_bias_agent"
    bias_detected: bool
    bias_instances: List[Dict[str, Any]] = []  # List of BiasInstance dicts
    overall_text_score: float  # 0-10 scale (0=high bias, 10=no bias)
    recommendations: List[str] = []
    rag_similar_cases: List[str] = []  # Similar case IDs from ChromaDB
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class VisualBiasReport(Model):
    """Visual Bias Agent report"""
    request_id: str
    agent_name: str = "visual_bias_agent"
    bias_detected: bool
    bias_instances: List[Dict[str, Any]] = []  # List of BiasInstance dicts
    overall_visual_score: float  # 0-10 scale (0=high bias, 10=no bias)
    diversity_metrics: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []
    rag_similar_cases: List[str] = []  # Similar case IDs from ChromaDB
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# ============================================================================
# SCORING AGENT MESSAGES
# ============================================================================

class BiasAnalysisComplete(Model):
    """Message from bias agents to scoring agent indicating analysis complete"""
    request_id: str
    sender_agent: str  # "text_bias_agent" or "visual_bias_agent"
    report: Dict[str, Any]  # The actual report data
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class FinalBiasReport(Model):
    """Final aggregated bias assessment from Scoring Agent"""
    request_id: str
    agent_name: str = "scoring_agent"
    overall_bias_score: float  # 0-10 scale (0=high bias, 10=no bias)
    score_breakdown: Dict[str, float]  # text_score, visual_score, intersectional_penalty
    bias_summary: Dict[str, int]  # total_issues, high_severity, medium_severity, low_severity
    top_concerns: List[str]
    recommendations: List[str]
    confidence: float  # 0.0 to 1.0
    similar_cases: List[str]  # Benchmark case IDs from ChromaDB
    text_report: Optional[Dict[str, Any]] = None  # Include original text report
    visual_report: Optional[Dict[str, Any]] = None  # Include original visual report
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
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
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_bias_instance_dict(
    bias_type: BiasCategory,
    severity: str,
    examples: List[str],
    context: str,
    confidence: float
) -> Dict[str, Any]:
    """Helper to create bias instance dictionary"""
    return {
        "bias_type": bias_type.value,
        "severity": severity,
        "examples": examples,
        "context": context,
        "confidence": confidence
    }
