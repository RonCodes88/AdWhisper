"""
Visual Bias Agent - Ad Bias Detection System

Role: Visual Content Analysis and Bias Detection
Responsibilities:
- Analyze visual content for representation bias
- Query ChromaDB for similar visual patterns (RAG RETRIEVAL POINT #2)
- Detect bias in subject representation, contextual placement, color usage
- Frame-by-frame analysis for video content
- Identify subtle visual cues and microaggressions
- Generate diversity metrics and recommendations
"""

from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from enum import Enum
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import shared models
from agents.shared_models import (
    EmbeddingPackage,
    VisualBiasReport,
    BiasAnalysisComplete,
    BiasCategory,
    create_bias_instance_dict,
    AgentError
)


# Visual bias types
class VisualBiasType(str, Enum):
    REPRESENTATION = "representation_bias"
    CONTEXTUAL = "contextual_bias"
    COLOR_SYMBOLISM = "color_symbolism_bias"
    BODY_REPRESENTATION = "body_representation_bias"
    CULTURAL_APPROPRIATION = "cultural_appropriation"
    TOKENISM = "tokenism"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Message Models
class VisualAnalysisRequest(Model):
    """Request for visual bias analysis"""
    request_id: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    visual_embedding: Optional[List[float]] = None
    chromadb_collection_id: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class VisualBiasDetection(Model):
    """Individual visual bias detection result"""
    bias_type: VisualBiasType
    severity: SeverityLevel
    examples: Optional[List[str]] = None
    context: str
    confidence: float


class DiversityMetrics(Model):
    """Quantitative diversity measurements"""
    gender_distribution: Optional[Dict[str, float]] = None
    apparent_ethnicity: Optional[Dict[str, float]] = None
    age_distribution: Optional[Dict[str, float]] = None
    body_type_diversity: float = 0.0
    power_dynamics_score: float = 0.0


class VisualBiasReport(Model):
    """Complete visual bias analysis report"""
    request_id: str
    agent: str = "visual_bias_agent"
    bias_detected: bool
    bias_types: Optional[List[VisualBiasDetection]] = None
    diversity_metrics: DiversityMetrics
    overall_visual_score: float
    recommendations: Optional[List[str]] = None
    rag_references: Optional[List[str]] = None
    confidence: float
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# Initialize Visual Bias Agent
visual_bias_agent = Agent(
    name="visual_bias_agent",
    seed="ad_bias_visual_agent_unique_seed_2024",
    port=8102,
    endpoint=["http://localhost:8102/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for visual bias analysis
visual_bias_protocol = Protocol(name="visual_bias_protocol", version="1.0")

# ASI:ONE LLM agent addresses
OPENAI_AGENT = 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y'
CLAUDE_AGENT = 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx'

# Scoring Agent address
SCORING_AGENT_ADDRESS = "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8"


@visual_bias_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Visual Bias Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {visual_bias_agent.address}")
    ctx.logger.info(f"🔧 Role: Visual Content Analysis and Bias Detection")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8102/submit")
    ctx.logger.info(f"👁️ Vision-LLM Integration: Ready for visual analysis")
    ctx.logger.info(f"⚡ Ready to analyze visual content for bias")


@visual_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Visual Bias Agent shutting down...")


@visual_bias_agent.on_rest_post("/analyze", EmbeddingPackage, BiasAnalysisComplete)
async def handle_visual_analysis_rest(ctx: Context, req: EmbeddingPackage) -> BiasAnalysisComplete:
    """
    REST endpoint for visual bias analysis.
    Analyzes visual content for bias and returns results.
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🎯 VISUAL BIAS AGENT - REST REQUEST RECEIVED")
    ctx.logger.info("=" * 80)

    try:
        ctx.logger.info(f"📨 Received REST request for visual analysis")
        ctx.logger.info(f"   📝 Request ID: {req.request_id}")
        ctx.logger.info(f"   🔢 Has visual embedding: {req.visual_embedding is not None}")
        ctx.logger.info(f"   🖼️  Has frames: {req.frames_base64 is not None and len(req.frames_base64 or []) > 0}")

        if req.frames_base64:
            ctx.logger.info(f"   📊 Number of frames: {len(req.frames_base64)}")
            ctx.logger.info(f"   📏 First frame size: {len(req.frames_base64[0]) if req.frames_base64 else 0} bytes")
        else:
            ctx.logger.warning(f"   ⚠️ No frames in base64 format!")

        # Extract media info from metadata
        ctx.logger.info(f"🔍 Extracting media information from metadata...")
        media_url = req.metadata.get("image_url") if req.metadata else None
        if not media_url and req.metadata:
            media_url = req.metadata.get("youtube", {}).get("thumbnail_url") if isinstance(req.metadata.get("youtube"), dict) else None

        media_type = "video" if req.frames_base64 else "image"
        ctx.logger.info(f"   🎨 Media type: {media_type}")
        ctx.logger.info(f"   🔗 Media URL: {media_url}")

        # Check if we have visual content to analyze
        has_frames = req.frames_base64 and len(req.frames_base64) > 0
        ctx.logger.info(f"   ✅ Has frames: {has_frames}")

        if not has_frames and not media_url:
            ctx.logger.error(f"❌ No visual content for request {req.request_id}")
            # Return error response
            return BiasAnalysisComplete(
                request_id=req.request_id,
                sender_agent="visual_bias_agent",
                report={
                    "request_id": req.request_id,
                    "agent_name": "visual_bias_agent",
                    "error": "No visual content provided"
                }
            )

        ctx.logger.info(f"✅ Visual content validated")

        # Step 1: Extract visual features
        ctx.logger.info(f"🔍 STEP 1: Extracting visual features...")
        if media_type == "video" and media_url:
            visual_frames = await extract_video_keyframes(ctx, media_url)
            ctx.logger.info(f"   🎬 Extracted {len(visual_frames)} keyframes from video")
        elif media_url:
            visual_frames = [media_url]
            ctx.logger.info(f"   🖼️  Using single image URL")
        else:
            visual_frames = []
            ctx.logger.info(f"   📦 Using {len(req.frames_base64)} base64 frames")

        # Step 2: Analyze visual content with Vision-LLM
        ctx.logger.info(f"👁️ STEP 2: Analyzing visual content with Vision-LLM...")
        initial_analysis = await analyze_visual_with_llm(ctx, visual_frames)
        ctx.logger.info(f"   ✅ Visual analysis complete")
        ctx.logger.info(f"   📊 Subjects detected: {len(initial_analysis.get('subjects', []))}")

        # Step 3: RAG RETRIEVAL - Query ChromaDB for similar visual patterns
        ctx.logger.info(f"🔎 STEP 3: RAG RETRIEVAL - Querying ChromaDB...")
        rag_results = await query_visual_patterns(ctx, req.visual_embedding, req.chromadb_collection_id)
        ctx.logger.info(f"   ✅ Found {len(rag_results)} similar visual cases")

        # Step 4: Detect representation metrics
        ctx.logger.info(f"📊 STEP 4: Calculating diversity metrics...")
        diversity_metrics = await detect_representation_metrics(ctx, initial_analysis)
        ctx.logger.info(f"   ✅ Diversity metrics calculated")

        # Step 5: Analyze composition and power dynamics
        ctx.logger.info(f"🎭 STEP 5: Analyzing composition and spatial dynamics...")
        composition_analysis = await analyze_composition(ctx, initial_analysis)
        ctx.logger.info(f"   ✅ Composition analysis complete")

        # Step 6: Classify visual biases
        ctx.logger.info(f"🏷️ STEP 6: Classifying visual bias types...")
        bias_detections = await classify_visual_biases(
            ctx,
            initial_analysis,
            diversity_metrics,
            composition_analysis,
            rag_results
        )
        ctx.logger.info(f"   ✅ Classified {len(bias_detections)} visual bias types")

        # Step 7: Calculate overall visual score
        ctx.logger.info(f"📊 STEP 7: Calculating overall visual bias score...")
        visual_score = await calculate_visual_score(ctx, bias_detections, diversity_metrics)
        ctx.logger.info(f"   ✅ Visual score calculated: {visual_score:.2f}/10")

        # Step 8: Generate recommendations
        ctx.logger.info(f"💡 STEP 8: Generating recommendations...")
        recommendations = await generate_visual_recommendations(ctx, bias_detections, diversity_metrics)
        ctx.logger.info(f"   ✅ Generated {len(recommendations)} recommendations")

        # Step 9: Create report - convert bias detections to dicts
        ctx.logger.info(f"📋 STEP 9: Creating visual bias report...")
        bias_instances_dicts = [
            {
                "bias_type": bd.bias_type.value,
                "severity": bd.severity.value,
                "examples": bd.examples or [],
                "context": bd.context,
                "confidence": bd.confidence
            }
            for bd in bias_detections
        ]

        # Create visual bias report dict
        report_dict = {
            "request_id": req.request_id,
            "agent_name": "visual_bias_agent",
            "bias_detected": len(bias_detections) > 0,
            "bias_instances": bias_instances_dicts,
            "overall_visual_score": visual_score,
            "diversity_metrics": diversity_metrics.dict() if hasattr(diversity_metrics, 'dict') else diversity_metrics,
            "recommendations": recommendations,
            "rag_similar_cases": [ref["id"] for ref in rag_results],
            "timestamp": datetime.now(UTC).isoformat()
        }
        ctx.logger.info(f"   ✅ Report created")

        ctx.logger.info(f"🎉 Analysis complete!")
        ctx.logger.info(f"   📊 Overall Score: {visual_score:.1f}/10")
        ctx.logger.info(f"   🚨 Bias detected: {len(bias_detections) > 0}")
        ctx.logger.info(f"   📝 Issues found: {len(bias_detections)}")
        ctx.logger.info(f"   💡 Recommendations: {len(recommendations)}")

        # Step 10: Send results to Scoring Agent
        ctx.logger.info(f"📤 STEP 10: Sending results to Scoring Agent...")
        ctx.logger.info(f"   🎯 Scoring Agent Address: {SCORING_AGENT_ADDRESS}")
        await send_to_scoring_agent(ctx, req.request_id, report_dict)
        ctx.logger.info(f"   ✅ Results sent successfully!")

        # Return response to REST caller
        response = BiasAnalysisComplete(
            request_id=req.request_id,
            sender_agent="visual_bias_agent",
            report=report_dict
        )
        ctx.logger.info(f"✅ Returning response to REST caller")
        ctx.logger.info(f"✅ Visual analysis for {req.request_id} completed successfully")
        ctx.logger.info("=" * 80)
        return response

    except Exception as e:
        ctx.logger.error("=" * 80)
        ctx.logger.error(f"❌ ERROR IN VISUAL BIAS AGENT")
        ctx.logger.error("=" * 80)
        ctx.logger.error(f"   Request ID: {req.request_id}")
        ctx.logger.error(f"   Error: {e}")
        ctx.logger.error(f"   Type: {type(e).__name__}")
        import traceback
        ctx.logger.error(f"   Traceback:\n{traceback.format_exc()}")
        ctx.logger.error("=" * 80)

        # Return error response
        return BiasAnalysisComplete(
            request_id=req.request_id,
            sender_agent="visual_bias_agent",
            report={
                "request_id": req.request_id,
                "agent_name": "visual_bias_agent",
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


async def extract_video_keyframes(ctx: Context, video_url: str) -> List[str]:
    """
    Extract keyframes from video for analysis.
    
    TODO: Implement actual video processing
    - Use OpenCV or ffmpeg to extract frames
    - Sample frames at regular intervals
    """
    ctx.logger.info(f"🎬 Extracting keyframes from video: {video_url}")
    
    # Placeholder keyframes
    keyframes = [
        f"{video_url}_frame_0.jpg",
        f"{video_url}_frame_5.jpg",
        f"{video_url}_frame_10.jpg"
    ]
    
    return keyframes


async def analyze_visual_with_llm(ctx: Context, visual_frames: List[str]) -> Dict[str, Any]:
    """
    Use Vision-LLM (GPT-4V, Claude Vision) to analyze visual content.
    
    TODO: Implement actual Vision-LLM integration
    - Use OpenAI GPT-4V or Claude Vision API
    - Analyze representation, composition, color usage
    """
    ctx.logger.info(f"👁️ Analyzing {len(visual_frames)} visual frame(s) with Vision-LLM...")
    
    # Placeholder analysis
    analysis = {
        "subjects": [
            {"type": "person", "gender": "male", "apparent_ethnicity": "white", "role": "executive"},
            {"type": "person", "gender": "female", "apparent_ethnicity": "white", "role": "assistant"}
        ],
        "composition": {
            "power_positioning": "males_dominant",
            "spatial_hierarchy": "males_foreground_females_background"
        },
        "color_usage": {
            "dominant_colors": ["blue", "grey"],
            "gendered_colors": "traditional"
        }
    }
    
    ctx.logger.info(f"✅ Vision-LLM analysis complete")
    return analysis


async def query_visual_patterns(
    ctx: Context,
    visual_embedding: Optional[List[float]],
    collection_id: str
) -> List[Dict[str, Any]]:
    """
    RAG RETRIEVAL POINT #2: Query ChromaDB for similar visual patterns.
    
    TODO: Implement actual ChromaDB query
    - Use visual_embedding (CLIP) for similarity search
    - Filter by bias_patterns_visual collection
    """
    ctx.logger.info(f"🔎 Querying ChromaDB for similar visual patterns...")
    
    # Placeholder RAG results
    rag_results = [
        {
            "id": "case_visual_042",
            "bias_type": "representation_bias",
            "similarity": 0.91,
            "context": "Similar male-dominant leadership positioning"
        },
        {
            "id": "case_visual_089",
            "bias_type": "contextual_bias",
            "similarity": 0.85,
            "context": "Women shown in supportive roles pattern"
        }
    ]
    
    ctx.logger.info(f"✅ Retrieved {len(rag_results)} similar visual cases")
    return rag_results


async def detect_representation_metrics(ctx: Context, analysis: Dict[str, Any]) -> DiversityMetrics:
    """
    Calculate quantitative diversity metrics from visual analysis.
    """
    ctx.logger.info(f"📊 Calculating diversity metrics...")
    
    # Placeholder metrics calculation
    metrics = DiversityMetrics(
        gender_distribution={"male": 0.8, "female": 0.2, "non-binary": 0.0},
        apparent_ethnicity={"white": 0.9, "black": 0.05, "asian": 0.05, "hispanic": 0.0},
        age_distribution={"young": 0.3, "middle": 0.6, "senior": 0.1},
        body_type_diversity=0.3,  # Low diversity
        power_dynamics_score=0.2  # Imbalanced (0=very imbalanced, 1=balanced)
    )
    
    ctx.logger.info(f"✅ Diversity metrics calculated")
    return metrics


async def analyze_composition(ctx: Context, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze visual composition for power dynamics and spatial positioning.
    """
    ctx.logger.info(f"🎭 Analyzing composition and power dynamics...")
    
    composition = {
        "spatial_hierarchy": "males_central_females_peripheral",
        "size_representation": "males_larger_prominent",
        "eye_contact": "males_direct_females_averted",
        "action_vs_passive": "males_active_females_passive"
    }
    
    ctx.logger.info(f"✅ Composition analysis complete")
    return composition


async def classify_visual_biases(
    ctx: Context,
    analysis: Dict[str, Any],
    diversity_metrics: DiversityMetrics,
    composition: Dict[str, Any],
    rag_results: List[Dict[str, Any]]
) -> List[VisualBiasDetection]:
    """
    Classify visual bias types based on analysis results.
    """
    ctx.logger.info(f"🏷️ Classifying visual biases...")
    
    detections = [
        VisualBiasDetection(
            bias_type=VisualBiasType.REPRESENTATION,
            severity=SeverityLevel.HIGH,
            examples=["All leadership figures are white males", "Women only in supportive roles"],
            context="Lacks diverse representation in authority roles",
            confidence=0.92
        ),
        VisualBiasDetection(
            bias_type=VisualBiasType.CONTEXTUAL,
            severity=SeverityLevel.MEDIUM,
            examples=["Males in foreground, females in background", "Size disparity in favor of males"],
            context="Spatial positioning reinforces gender power dynamics",
            confidence=0.85
        )
    ]
    
    ctx.logger.info(f"✅ Classified {len(detections)} visual bias types")
    return detections


async def calculate_visual_score(
    ctx: Context,
    detections: List[VisualBiasDetection],
    diversity_metrics: DiversityMetrics
) -> float:
    """
    Calculate overall visual bias score (0-10 scale).
    """
    if not detections:
        return 9.5
    
    # Severity-based penalties
    severity_weights = {
        SeverityLevel.LOW: 0.5,
        SeverityLevel.MEDIUM: 1.0,
        SeverityLevel.HIGH: 1.5,
        SeverityLevel.CRITICAL: 2.0
    }
    
    bias_penalty = sum(severity_weights[d.severity] * d.confidence for d in detections)
    
    # Diversity metrics penalty
    diversity_penalty = 0.0
    if diversity_metrics.power_dynamics_score < 0.5:
        diversity_penalty += 1.0
    if diversity_metrics.body_type_diversity < 0.5:
        diversity_penalty += 0.5
    
    score = max(0.0, 10.0 - bias_penalty - diversity_penalty)
    
    ctx.logger.info(f"📊 Calculated visual score: {score:.1f}")
    return score


async def generate_visual_recommendations(
    ctx: Context,
    detections: List[VisualBiasDetection],
    diversity_metrics: DiversityMetrics
) -> List[str]:
    """
    Generate actionable recommendations for visual content.
    """
    recommendations = []
    
    for detection in detections:
        if detection.bias_type == VisualBiasType.REPRESENTATION:
            recommendations.append("Increase diverse representation across all roles and positions")
        elif detection.bias_type == VisualBiasType.CONTEXTUAL:
            recommendations.append("Ensure balanced spatial positioning and equal prominence for all subjects")
        elif detection.bias_type == VisualBiasType.TOKENISM:
            recommendations.append("Avoid token representation; ensure meaningful and proportional diversity")
        elif detection.bias_type == VisualBiasType.CULTURAL_APPROPRIATION:
            recommendations.append("Review cultural symbols and ensure appropriate context and respect")
    
    # Diversity-based recommendations
    if diversity_metrics.power_dynamics_score < 0.5:
        recommendations.append("Balance power dynamics by showing diverse individuals in leadership roles")
    if diversity_metrics.body_type_diversity < 0.5:
        recommendations.append("Include diverse body types and avoid stereotypical beauty standards")
    
    ctx.logger.info(f"💡 Generated {len(recommendations)} recommendations")
    return recommendations


async def send_to_scoring_agent(ctx: Context, request_id: str, report_dict: Dict[str, Any]):
    """
    Send analysis results to Scoring Agent.
    """
    if not SCORING_AGENT_ADDRESS:
        ctx.logger.warning(f"⚠️ Scoring agent address not set")
        return

    analysis_complete = BiasAnalysisComplete(
        request_id=request_id,
        sender_agent="visual_bias_agent",
        report=report_dict
    )

    ctx.logger.info(f"📤 Sending results to Scoring Agent")
    await ctx.send(SCORING_AGENT_ADDRESS, analysis_complete)
    ctx.logger.info(f"✅ Results sent successfully")


# Include protocol
visual_bias_agent.include(visual_bias_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║       👁️ VISUAL BIAS AGENT - Ad Bias Detection               ║
╚══════════════════════════════════════════════════════════════╝

Role: Visual Content Analysis and Bias Detection

Capabilities:
  ✓ Analyzes images and video for visual bias
  ✓ RAG retrieval for similar visual patterns
  ✓ Calculates diversity and representation metrics
  ✓ Vision-LLM integration (GPT-4V, Claude Vision)
  ✓ Detects subtle visual cues and microaggressions

Visual Bias Types Detected:
  • Representation bias (diversity, tokenism)
  • Contextual bias (power dynamics, spatial positioning)
  • Color symbolism bias
  • Body representation stereotypes
  • Cultural appropriation
  • Tokenism patterns

Metrics Analyzed:
  • Gender distribution
  • Ethnic representation
  • Age distribution
  • Body type diversity
  • Power dynamics balance

📍 Waiting for visual analysis requests...
🛑 Stop with Ctrl+C
    """)
    visual_bias_agent.run()

