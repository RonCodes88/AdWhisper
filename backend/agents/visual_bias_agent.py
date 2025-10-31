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
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chroma import ChromaDB

from dotenv import load_dotenv
load_dotenv()


# Visual bias types
class VisualBiasType(str, Enum):
    REPRESENTATION = "representation_bias"
    CONTEXTUAL = "contextual_bias"
    COLOR_SYMBOLISM = "color_symbolism_bias"
    BODY_REPRESENTATION = "body_representation_bias"
    CULTURAL_APPROPRIATION = "cultural_appropriation"
    TOKENISM = "tokenism"
    ACCESSIBILITY = "accessibility_bias"


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


# REST API Models (for HTTP endpoint)
class RESTVisualAnalysisRequest(Model):
    """REST API request for visual bias analysis"""
    request_id: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    visual_embedding: Optional[List[float]] = None
    chromadb_collection_id: str
    metadata: Optional[Dict[str, Any]] = None


class RESTAcknowledgement(Model):
    """REST API acknowledgement response"""
    status: str
    request_id: str
    message: str


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
    mailbox=False  # Local development - direct agent-to-agent communication
)

# Protocol for visual bias analysis
visual_bias_protocol = Protocol(name="visual_bias_protocol", version="1.0")

# Global ChromaDB instance
chroma_db = None

# ASI:ONE LLM agent addresses
OPENAI_AGENT = os.getenv('ASI_OPENAI_AGENT', 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y')
CLAUDE_AGENT = os.getenv('ASI_CLAUDE_AGENT', 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx')

# Scoring Agent address
SCORING_AGENT_ADDRESS = os.getenv("SCORING_AGENT_ADDRESS", "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8")


@visual_bias_agent.on_event("startup")
async def startup(ctx: Context):
    global chroma_db
    
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"🚀 Visual Bias Agent starting up...")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📍 Agent address: {visual_bias_agent.address}")
    ctx.logger.info(f"📍 Agent name: {visual_bias_agent.name}")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8102/submit")
    ctx.logger.info(f"🔌 Port: 8102")
    ctx.logger.info(f"📬 Mailbox: {visual_bias_agent._use_mailbox}")
    
    # Initialize ChromaDB
    ctx.logger.info("💾 Initializing ChromaDB...")
    chroma_db = ChromaDB()
    ctx.logger.info(f"✅ ChromaDB initialized")
    
    ctx.logger.info(f"🔧 Role: Visual Content Analysis and Bias Detection")
    ctx.logger.info(f"👁️ Vision-LLM Integration: Ready for visual analysis")
    ctx.logger.info(f"⚡ Ready to analyze visual content for bias")
    ctx.logger.info(f"📨 Listening for VisualAnalysisRequest messages...")
    ctx.logger.info(f"📨 Message handlers registered: {len(visual_bias_protocol._models)}")
    ctx.logger.info("=" * 80)


# Shared analysis logic used by both REST and message handlers
async def process_visual_analysis(ctx: Context, msg: VisualAnalysisRequest):
    """
    Core visual analysis logic that can be called from multiple entry points.
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🎯 PROCESSING VISUAL ANALYSIS REQUEST")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 Request ID: {msg.request_id}")
    
    media_type = "image" if msg.image_url else "video"
    media_url = msg.image_url or msg.video_url
    ctx.logger.info(f"🎨 Media type: {media_type}, URL: {media_url}")
    
    # Step 1: Extract visual features
    ctx.logger.info(f"🔍 Extracting visual features...")
    if media_type == "video":
        visual_frames = await extract_video_keyframes(ctx, msg.video_url)
        ctx.logger.info(f"🎬 Extracted {len(visual_frames)} keyframes from video")
    else:
        visual_frames = [msg.image_url]
    
    # Step 2: Analyze visual content with Vision-LLM
    ctx.logger.info(f"👁️ Analyzing visual content with Vision-LLM...")
    initial_analysis = await analyze_visual_with_llm(ctx, visual_frames)
    
    # Step 3: RAG RETRIEVAL - Query ChromaDB for similar visual patterns
    ctx.logger.info(f"🔎 RAG RETRIEVAL: Querying ChromaDB for similar visual patterns...")
    rag_results = await query_visual_patterns(ctx, msg.visual_embedding, msg.chromadb_collection_id)
    ctx.logger.info(f"✅ Found {len(rag_results)} similar visual cases")
    
    # Step 4: Detect representation metrics
    ctx.logger.info(f"📊 Calculating diversity metrics...")
    diversity_metrics = await detect_representation_metrics(ctx, initial_analysis)
    
    # Step 5: Analyze composition and power dynamics
    ctx.logger.info(f"🎭 Analyzing composition and spatial dynamics...")
    composition_analysis = await analyze_composition(ctx, initial_analysis)
    
    # Step 6: Classify visual biases
    ctx.logger.info(f"🏷️ Classifying visual bias types...")
    bias_detections = await classify_visual_biases(
        ctx, 
        initial_analysis, 
        diversity_metrics,
        composition_analysis,
        rag_results
    )
    
    # Step 7: Calculate overall visual score
    ctx.logger.info(f"📊 Calculating overall visual bias score...")
    visual_score = await calculate_visual_score(ctx, bias_detections, diversity_metrics)
    
    # Step 8: Generate recommendations
    ctx.logger.info(f"💡 Generating recommendations...")
    recommendations = await generate_visual_recommendations(ctx, bias_detections, diversity_metrics)
    
    # Step 9: Create report
    report = VisualBiasReport(
        request_id=msg.request_id,
        agent="visual_bias_agent",
        bias_detected=len(bias_detections) > 0,
        bias_types=bias_detections,
        diversity_metrics=diversity_metrics,
        overall_visual_score=visual_score,
        recommendations=recommendations,
        rag_references=[ref["id"] for ref in rag_results],
        confidence=sum(bd.confidence for bd in bias_detections) / len(bias_detections) if bias_detections else 1.0
    )
    
    ctx.logger.info(f"✅ Analysis complete: Score={visual_score:.1f}, Biases detected={len(bias_detections)}")
    ctx.logger.info("=" * 80)
    
    return report


async def send_visual_report_to_scoring_agent(ctx: Context, report: VisualBiasReport):
    """Send visual bias report to Scoring Agent via HTTP."""
    try:
        import httpx
        payload = report.dict()
        
        ctx.logger.info(f"📤 Sending report to Scoring Agent...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8103/api/aggregate",
                json={"source": "visual_bias_agent", "report": payload}
            )
            ctx.logger.info(f"✅ Sent report to Scoring Agent")
    except Exception as e:
        ctx.logger.error(f"❌ Error sending to Scoring Agent: {e}")


# REST API Endpoint - HTTP interface for Ingestion Agent
@visual_bias_agent.on_rest_post("/api/analyze", RESTVisualAnalysisRequest, RESTAcknowledgement)
async def handle_rest_visual_analysis(ctx: Context, req: RESTVisualAnalysisRequest) -> RESTAcknowledgement:
    """
    REST API endpoint for receiving visual analysis requests via HTTP.
    This bypasses uAgent messaging and uses direct HTTP communication.
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🌐 REST API: Received visual analysis request via HTTP")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 Request ID: {req.request_id}")
    ctx.logger.info(f"🎨 Image URL: {req.image_url}")
    ctx.logger.info(f"🎬 Video URL: {req.video_url}")
    ctx.logger.info(f"📦 Embedding dimension: {len(req.visual_embedding) if req.visual_embedding else 'None'}")
    
    try:
        # Convert REST request to internal VisualAnalysisRequest
        visual_request = VisualAnalysisRequest(
            request_id=req.request_id,
            image_url=req.image_url,
            video_url=req.video_url,
            visual_embedding=req.visual_embedding,
            chromadb_collection_id=req.chromadb_collection_id,
            metadata=req.metadata
        )
        
        ctx.logger.info(f"✅ Converted REST request to internal format")
        ctx.logger.info(f"🔄 Starting visual bias analysis...")
        
        # Process the analysis (call the same logic as the message handler)
        report = await process_visual_analysis(ctx, visual_request)
        
        # Send results to Scoring Agent via HTTP
        await send_visual_report_to_scoring_agent(ctx, report)
        
        ctx.logger.info(f"✅ Visual analysis completed successfully")
        ctx.logger.info("=" * 80)
        
        return RESTAcknowledgement(
            status="accepted",
            request_id=req.request_id,
            message="Visual analysis request accepted and processing"
        )
        
    except Exception as e:
        ctx.logger.error("=" * 80)
        ctx.logger.error(f"❌ Error processing REST visual analysis request")
        ctx.logger.error(f"Error: {e}")
        import traceback
        ctx.logger.error(traceback.format_exc())
        ctx.logger.error("=" * 80)
        
        return RESTAcknowledgement(
            status="error",
            request_id=req.request_id,
            message=f"Error processing request: {str(e)}"
        )


# Periodic heartbeat to show agent is alive
@visual_bias_agent.on_interval(period=30.0)
async def heartbeat(ctx: Context):
    ctx.logger.info(f"💓 Visual Bias Agent heartbeat - listening on {ctx.agent.address}")


@visual_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Visual Bias Agent shutting down...")


# DEBUG: Log all incoming messages
@visual_bias_agent.on_message(model=Model)
async def debug_all_messages(ctx: Context, sender: str, msg: Model):
    """Debug handler to log ALL incoming messages"""
    ctx.logger.info("=" * 80)
    ctx.logger.info("🟢 VISUAL BIAS AGENT - MESSAGE RECEIVED!")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 From sender: {sender}")
    ctx.logger.info(f"📦 Message type: {type(msg).__name__}")
    ctx.logger.info(f"📦 Message module: {type(msg).__module__}")
    ctx.logger.info(f"📄 Message content: {msg}")
    ctx.logger.info("=" * 80)


@visual_bias_protocol.on_message(model=VisualAnalysisRequest, replies=VisualBiasReport)
async def handle_visual_analysis(ctx: Context, sender: str, msg: VisualAnalysisRequest):
    """
    Analyze visual content for bias using Vision-LLM and RAG retrieval.
    """
    try:
        ctx.logger.info("=" * 80)
        ctx.logger.info("🎯 PROCESSING VISUAL ANALYSIS REQUEST")
        ctx.logger.info("=" * 80)
        ctx.logger.info(f"📨 Received visual analysis request: {msg.request_id}")
        ctx.logger.info(f"📨 From sender: {sender}")
        media_type = "image" if msg.image_url else "video"
        media_url = msg.image_url or msg.video_url
        ctx.logger.info(f"🎨 Media type: {media_type}, URL: {media_url}")
        
        # Step 1: Extract visual features
        ctx.logger.info(f"🔍 Extracting visual features...")
        if media_type == "video":
            visual_frames = await extract_video_keyframes(ctx, msg.video_url)
            ctx.logger.info(f"🎬 Extracted {len(visual_frames)} keyframes from video")
        else:
            visual_frames = [msg.image_url]
        
        # Step 2: Analyze visual content with Vision-LLM
        ctx.logger.info(f"👁️ Analyzing visual content with Vision-LLM...")
        initial_analysis = await analyze_visual_with_llm(ctx, visual_frames)
        
        # Step 3: RAG RETRIEVAL - Query ChromaDB for similar visual patterns
        ctx.logger.info(f"🔎 RAG RETRIEVAL: Querying ChromaDB for similar visual patterns...")
        rag_results = await query_visual_patterns(ctx, msg.visual_embedding, msg.chromadb_collection_id)
        ctx.logger.info(f"✅ Found {len(rag_results)} similar visual cases")
        
        # Step 4: Detect representation metrics
        ctx.logger.info(f"📊 Calculating diversity metrics...")
        diversity_metrics = await detect_representation_metrics(ctx, initial_analysis)
        
        # Step 5: Analyze composition and power dynamics
        ctx.logger.info(f"🎭 Analyzing composition and spatial dynamics...")
        composition_analysis = await analyze_composition(ctx, initial_analysis)
        
        # Step 6: Classify visual biases
        ctx.logger.info(f"🏷️ Classifying visual bias types...")
        bias_detections = await classify_visual_biases(
            ctx, 
            initial_analysis, 
            diversity_metrics,
            composition_analysis,
            rag_results
        )
        
        # Step 7: Calculate overall visual score
        ctx.logger.info(f"📊 Calculating overall visual bias score...")
        visual_score = await calculate_visual_score(ctx, bias_detections, diversity_metrics)
        
        # Step 8: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = await generate_visual_recommendations(ctx, bias_detections, diversity_metrics)
        
        # Step 9: Create report
        report = VisualBiasReport(
            request_id=msg.request_id,
            agent="visual_bias_agent",
            bias_detected=len(bias_detections) > 0,
            bias_types=bias_detections,
            diversity_metrics=diversity_metrics,
            overall_visual_score=visual_score,
            recommendations=recommendations,
            rag_references=[ref["id"] for ref in rag_results],
            confidence=sum(bd.confidence for bd in bias_detections) / len(bias_detections) if bias_detections else 1.0
        )
        
        ctx.logger.info(f"✅ Analysis complete: Score={visual_score:.1f}, Biases detected={len(bias_detections)}")
        
        # Step 10: Send report back to sender and to Scoring Agent
        await ctx.send(sender, report)
        
        if SCORING_AGENT_ADDRESS:
            ctx.logger.info(f"📤 Forwarding report to Scoring Agent: {SCORING_AGENT_ADDRESS}")
            # await ctx.send(SCORING_AGENT_ADDRESS, report)
        
        ctx.logger.info(f"✅ Visual analysis for {msg.request_id} completed successfully")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing visual content for {msg.request_id}: {e}")
        # Send error report
        error_report = VisualBiasReport(
            request_id=msg.request_id,
            agent="visual_bias_agent",
            bias_detected=False,
            bias_types=[],
            diversity_metrics=DiversityMetrics(),
            overall_visual_score=5.0,
            recommendations=["Error occurred during visual analysis"],
            rag_references=[],
            confidence=0.0
        )
        await ctx.send(sender, error_report)


async def extract_video_keyframes(ctx: Context, video_url: str, max_frames: int = None) -> List[str]:
    """
    Extract keyframes from video for analysis using OpenCV with smart sampling.
    
    Args:
        ctx: Agent context
        video_url: Path to the video file
        max_frames: Maximum number of frames to extract (if None, uses dynamic calculation)
    
    Returns:
        List of paths to extracted frame images
    """
    ctx.logger.info(f"🎬 Extracting keyframes from video: {video_url}")
    
    try:
        import cv2
        import os
        
        # Check if video file exists
        if not os.path.exists(video_url):
            ctx.logger.error(f"❌ Video file not found: {video_url}")
            return []
        
        # Open the video
        video = cv2.VideoCapture(video_url)
        
        if not video.isOpened():
            ctx.logger.error(f"❌ Could not open video file: {video_url}")
            return []
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        ctx.logger.info(f"📊 Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
        
        # Smart frame calculation based on video length
        if max_frames is None:
            if duration <= 15:
                # Very short ads (Instagram, TikTok): capture every second
                calculated_frames = int(duration)
                ctx.logger.info(f"📐 Short video (<15s): extracting 1 frame/second = {calculated_frames} frames")
            elif duration <= 30:
                # Standard ads (YouTube pre-roll): 1 frame every 2 seconds
                calculated_frames = int(duration / 2)
                ctx.logger.info(f"📐 Medium video (15-30s): extracting 1 frame/2s = {calculated_frames} frames")
            elif duration <= 60:
                # Long-form ads: 1 frame every 3 seconds
                calculated_frames = 20
                ctx.logger.info(f"📐 Long video (30-60s): extracting ~20 frames")
            else:
                # Extended content: cap at 30 frames
                calculated_frames = 30
                ctx.logger.info(f"📐 Very long video (>60s): capping at 30 frames")
            
            max_frames = min(calculated_frames, 30)  # Hard cap at 30 frames
        
        ctx.logger.info(f"🎯 Target frame count: {max_frames}")
        
        # Calculate frame indices to extract (evenly distributed)
        if total_frames <= max_frames:
            # Extract all frames if video is very short
            frame_indices = list(range(total_frames))
        else:
            # Extract evenly distributed frames
            step = total_frames // max_frames
            frame_indices = [i * step for i in range(max_frames)]
        
        ctx.logger.info(f"🎯 Extracting {len(frame_indices)} frames at indices: {frame_indices}")
        
        # Extract frames
        keyframe_paths = []
        for idx in frame_indices:
            # Set video position
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            
            if ret:
                # Create frame filename
                frame_filename = f"{video_url}_frame_{idx}.jpg"
                
                # Save frame as JPG
                cv2.imwrite(frame_filename, frame)
                keyframe_paths.append(frame_filename)
                ctx.logger.info(f"   ✅ Extracted frame {idx} → {frame_filename}")
            else:
                ctx.logger.warning(f"   ⚠️  Could not read frame {idx}")
        
        # Release video
        video.release()
        
        ctx.logger.info(f"✅ Successfully extracted {len(keyframe_paths)} keyframes from video")
        return keyframe_paths
        
    except ImportError:
        ctx.logger.error("❌ OpenCV (cv2) not installed. Install with: pip install opencv-python")
        return []
    except Exception as e:
        ctx.logger.error(f"❌ Error extracting video frames: {e}")
        import traceback
        ctx.logger.error(traceback.format_exc())
        return []


async def analyze_visual_with_llm(ctx: Context, visual_frames: List[str]) -> Dict[str, Any]:
    """
    Analyze visual content for bias using Claude Vision API.
    
    Uses Anthropic's Claude Vision model to detect demographic representation,
    power dynamics, stereotyping, and other visual biases.
    
    Based on Fetch.ai's image analysis example:
    https://innovationlab.fetch.ai/resources/docs/next/examples/chat-protocol/image-analysis-agent
    """
    ctx.logger.info(f"👁️ Analyzing {len(visual_frames)} visual frame(s) with Claude Vision...")
    
    try:
        from vision_analysis import analyze_image_for_bias, analyze_multiple_frames
        
        if not visual_frames:
            return {"error": "No frames provided"}
        
        # Use Claude Vision API for real bias detection
        if len(visual_frames) == 1:
            # Single image analysis
            analysis_result = analyze_image_for_bias(visual_frames[0])
        else:
            # Multiple frames (video) - aggregate analysis
            ctx.logger.info(f"🎬 Analyzing video with {len(visual_frames)} frames...")
            analysis_result = analyze_multiple_frames(visual_frames)
        
        # Check if we got an error or fallback
        if analysis_result.get("error") or analysis_result.get("fallback"):
            ctx.logger.warning(f"⚠️  Vision API issue: {analysis_result.get('error', 'Fallback mode')}")
            if analysis_result.get("message"):
                ctx.logger.warning(f"   {analysis_result['message']}")
        else:
            ctx.logger.info(f"✅ Claude Vision analysis complete")
            
            # Log key findings
            if "text_detected" in analysis_result:
                text_info = analysis_result["text_detected"]
                if text_info.get("has_text"):
                    ctx.logger.info(f"   📝 Text in image: '{text_info.get('text_content', 'N/A')}'")
                    if text_info.get("racial_color_associations"):
                        ctx.logger.info(f"   🚨 Racial associations: {text_info['racial_color_associations']}")
            
            if "people_detected" in analysis_result:
                people = analysis_result["people_detected"]
                ctx.logger.info(f"   👥 People detected: {people.get('total_count', 0)}")
            
            if "frames_analyzed" in analysis_result:
                ctx.logger.info(f"   🎬 Frames analyzed: {analysis_result['frames_analyzed']}")
            
            if "bias_detections" in analysis_result:
                bias_count = len(analysis_result["bias_detections"])
                ctx.logger.info(f"   🔍 Bias detections from Claude: {bias_count}")
                for i, bias in enumerate(analysis_result["bias_detections"][:3], 1):  # Show first 3
                    if isinstance(bias, dict):
                        ctx.logger.info(f"      {i}. {bias.get('type', 'unknown')} ({bias.get('severity', 'unknown')}): {bias.get('description', '')[:80]}...")
                    else:
                        ctx.logger.warning(f"      {i}. ⚠️  String format (expected object): {str(bias)[:80]}...")
            
            if "overall_assessment" in analysis_result:
                assessment = analysis_result["overall_assessment"]
                ctx.logger.info(f"   📊 Bias score: {assessment.get('bias_score', 'N/A')}/10")
                ctx.logger.info(f"   🎨 Diversity score: {assessment.get('diversity_score', 'N/A')}/10")
        
        return analysis_result
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in visual analysis: {e}")
        import traceback
        ctx.logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "fallback": True
        }


async def query_visual_patterns(
    ctx: Context,
    visual_embedding: Optional[List[float]],
    collection_id: str
) -> List[Dict[str, Any]]:
    """
    RAG RETRIEVAL POINT #2: Query ChromaDB for similar visual patterns.
    Uses CLIP embeddings for semantic visual similarity search.
    """
    global chroma_db
    
    if chroma_db is None or visual_embedding is None:
        ctx.logger.warning("⚠️ ChromaDB or visual embedding not available for RAG retrieval")
        return []
    
    try:
        ctx.logger.info(f"🔎 RAG: Querying ChromaDB for similar visual bias patterns...")
        
        # Query the visual bias patterns collection
        results = chroma_db.query_by_embedding(
            collection_name=ChromaDB.COLLECTION_VISUAL_PATTERNS,
            embedding=visual_embedding,
            n_results=5
        )
        
        # Parse results
        rag_results = []
        if results and 'ids' in results and len(results['ids']) > 0:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results else []
            distances = results['distances'][0] if 'distances' in results else []
            
            for i, case_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                rag_results.append({
                    "id": case_id,
                    "bias_type": metadata.get("bias_type", "unknown"),
                    "similarity": similarity,
                    "context": metadata.get("context", "No context available"),
                    "severity": metadata.get("severity", "medium"),
                    "visual_features": metadata.get("visual_features", "")
                })
        
        ctx.logger.info(f"✅ RAG: Retrieved {len(rag_results)} similar visual bias cases")
        return rag_results
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in RAG retrieval: {e}")
        return []


async def detect_representation_metrics(ctx: Context, analysis: Dict[str, Any]) -> DiversityMetrics:
    """
    Calculate quantitative diversity metrics from Claude Vision analysis.
    """
    ctx.logger.info(f"📊 Calculating diversity metrics...")
    
    # Check if we have real Vision API data
    if analysis.get("fallback") or analysis.get("error"):
        # Fallback to neutral metrics
        metrics = DiversityMetrics(
            gender_distribution={"unknown": 1.0},
            apparent_ethnicity={"unknown": 1.0},
            age_distribution={"unknown": 1.0},
            body_type_diversity=0.5,
            power_dynamics_score=0.5
        )
        ctx.logger.info(f"⚠️  Using neutral metrics (Vision API not available)")
        return metrics
    
    # Extract real data from Claude Vision analysis
    people_data = analysis.get("people_detected", {})
    demographics = people_data.get("visible_demographics", {})
    spatial = analysis.get("spatial_analysis", {})
    
    # Calculate gender distribution (as proportions)
    gender_data = demographics.get("gender", {})
    total_gender = sum(gender_data.values()) or 1
    gender_dist = {k: v/total_gender for k, v in gender_data.items() if v > 0}
    
    # Calculate ethnicity distribution
    ethnicity_data = demographics.get("ethnicity", {})
    total_ethnicity = sum(ethnicity_data.values()) or 1
    ethnicity_dist = {k: v/total_ethnicity for k, v in ethnicity_data.items() if v > 0}
    
    # Calculate age distribution
    age_data = demographics.get("age_groups", {})
    total_age = sum(age_data.values()) or 1
    age_dist = {k: v/total_age for k, v in age_data.items() if v > 0}
    
    # Calculate body type diversity score (0-1 scale)
    body_types = demographics.get("body_types", {})
    body_variety = len([v for v in body_types.values() if v > 0])
    body_diversity = min(body_variety / 4.0, 1.0)  # Normalize to 0-1
    
    # Estimate power dynamics score from spatial analysis
    # Lower score = more imbalanced power dynamics
    power_score = 0.5  # Default neutral
    if "power_positioning" in spatial:
        positioning = spatial["power_positioning"].lower()
        if "balanced" in positioning or "equal" in positioning:
            power_score = 0.8
        elif "imbalanced" in positioning or "dominant" in positioning:
            power_score = 0.3
    
    metrics = DiversityMetrics(
        gender_distribution=gender_dist if gender_dist else {"unknown": 1.0},
        apparent_ethnicity=ethnicity_dist if ethnicity_dist else {"unknown": 1.0},
        age_distribution=age_dist if age_dist else {"unknown": 1.0},
        body_type_diversity=body_diversity,
        power_dynamics_score=power_score
    )
    
    ctx.logger.info(f"✅ Diversity metrics calculated from Claude Vision data")
    return metrics


async def analyze_composition(ctx: Context, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze visual composition using Claude Vision spatial analysis data.
    """
    ctx.logger.info(f"🎭 Analyzing composition and power dynamics...")
    
    # Extract spatial analysis from Claude Vision
    spatial = analysis.get("spatial_analysis", {})
    
    composition_analysis = {
        "power_positioning": spatial.get("power_positioning", "unknown"),
        "central_subjects": spatial.get("central_subjects", []),
        "peripheral_subjects": spatial.get("peripheral_subjects", []),
        "prominence_analysis": spatial.get("prominence_analysis", ""),
        "data_source": "claude_vision" if not analysis.get("fallback") else "fallback"
    }
    
    ctx.logger.info(f"✅ Composition analysis complete")
    return composition_analysis


async def classify_visual_biases(
    ctx: Context,
    analysis: Dict[str, Any],
    diversity_metrics: DiversityMetrics,
    composition: Dict[str, Any],
    rag_results: List[Dict[str, Any]]
) -> List[VisualBiasDetection]:
    """
    Classify visual bias types using Claude Vision API detections.
    """
    ctx.logger.info(f"🏷️ Classifying visual biases...")
    
    detections = []
    
    # Check if we have real Vision API data
    if analysis.get("fallback") or analysis.get("error"):
        # API not available - return minimal detection
        detections.append(
            VisualBiasDetection(
                bias_type=VisualBiasType.REPRESENTATION,
                severity=SeverityLevel.LOW,
                examples=["Vision API not configured"],
                context=analysis.get("message", "Set ANTHROPIC_API_KEY to enable real bias detection"),
                confidence=0.3
            )
        )
        ctx.logger.info(f"⚠️  Vision API not available - returning limited results")
        return detections
    
    # Extract bias detections from Claude Vision analysis
    bias_detections_data = analysis.get("bias_detections", [])
    accessibility_issues = analysis.get("accessibility_issues", [])
    text_data = analysis.get("text_detected", {})
    
    # Check for text-based racial bias (highest priority)
    if text_data.get("has_text"):
        text_content = text_data.get("text_content", "").lower()
        text_analysis_result = text_data.get("text_analysis", "")
        racial_associations = text_data.get("racial_color_associations", "")
        
        # Log the text content for debugging
        ctx.logger.info(f"📝 Text detected in image: '{text_content}'")
        ctx.logger.info(f"📝 Text analysis: {text_analysis_result}")
        
        # Check for problematic text (racial color associations, etc.)
        if racial_associations and "bias" in racial_associations.lower():
            # Add as CRITICAL bias
            detections.append(
                VisualBiasDetection(
                    bias_type=VisualBiasType.CONTEXTUAL,
                    severity=SeverityLevel.CRITICAL,
                    examples=[f"Text: '{text_content}'", racial_associations],
                    context=f"Racial color association detected in text: {racial_associations}",
                    confidence=0.95
                )
            )
            ctx.logger.info(f"🚨 CRITICAL: Racial bias detected in text")
    
    # Process gender dynamics from Claude Vision's analysis
    gender_dynamics = analysis.get("gender_dynamics", {})
    if gender_dynamics.get("has_gender_bias"):
        problematic_msg = gender_dynamics.get("problematic_messaging", "")
        stereotypes = gender_dynamics.get("gender_stereotypes", [])
        
        ctx.logger.info(f"🚨 Gender bias detected by Claude Vision")
        ctx.logger.info(f"   Problematic messaging: {problematic_msg}")
        
        # Add this as a separate detection from Claude's structured output
        detections.append(
            VisualBiasDetection(
                bias_type=VisualBiasType.CONTEXTUAL,
                severity=SeverityLevel.HIGH,
                examples=stereotypes if stereotypes else [problematic_msg],
                context=f"Gender stereotype detected: {problematic_msg}",
                confidence=0.9
            )
        )
    
    # Convert Claude's bias detections to our format
    bias_type_map = {
        "representation": VisualBiasType.REPRESENTATION,
        "contextual": VisualBiasType.CONTEXTUAL,
        "tokenism": VisualBiasType.TOKENISM,
        "stereotyping": VisualBiasType.BODY_REPRESENTATION,
        "cultural_appropriation": VisualBiasType.CULTURAL_APPROPRIATION,
        "gender": VisualBiasType.CONTEXTUAL,  # Gender bias is contextual
    }
    
    severity_map = {
        "low": SeverityLevel.LOW,
        "medium": SeverityLevel.MEDIUM,
        "high": SeverityLevel.HIGH,
        "critical": SeverityLevel.CRITICAL
    }
    
    for bias in bias_detections_data:
        bias_type_str = bias.get("type", "representation")
        bias_type = bias_type_map.get(bias_type_str, VisualBiasType.REPRESENTATION)
        
        severity_str = bias.get("severity", "medium")
        severity = severity_map.get(severity_str, SeverityLevel.MEDIUM)
        
        description = bias.get("description", "")
        evidence = bias.get("evidence", [])
        affected_groups = bias.get("affected_groups", [])
        
        # Log each detected bias
        ctx.logger.info(f"   🔍 Bias detected: {bias_type_str} ({severity_str})")
        ctx.logger.info(f"      Description: {description[:100]}...")
        
        # Construct context from description and affected groups
        context_parts = [description]
        if affected_groups:
            context_parts.append(f"Affects: {', '.join(affected_groups)}")
        context = " | ".join(context_parts)
        
        detections.append(
            VisualBiasDetection(
                bias_type=bias_type,
                severity=severity,
                examples=evidence if evidence else [description],
                context=context,
                confidence=0.85  # High confidence from Vision-LLM
            )
        )
    
    # Add accessibility issues
    for issue in accessibility_issues:
        issue_type = issue.get("type", "contrast")
        severity_str = issue.get("severity", "medium")
        severity = severity_map.get(severity_str, SeverityLevel.MEDIUM)
        description = issue.get("description", "")
        
        detections.append(
            VisualBiasDetection(
                bias_type=VisualBiasType.ACCESSIBILITY,
                severity=severity,
                examples=[f"{issue_type}: {description}"],
                context=f"Accessibility concern: {description}",
                confidence=0.9
            )
        )
    
    ctx.logger.info(f"✅ Classified {len(detections)} visual bias types from Claude Vision")
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
    
    ctx.logger.info(f"📊 Calculated visual score: {score:.1f}/10")
    return score


async def generate_visual_recommendations(
    ctx: Context,
    detections: List[VisualBiasDetection],
    diversity_metrics: DiversityMetrics
) -> List[str]:
    """
    Generate actionable recommendations based on Claude Vision analysis.
    """
    recommendations = []
    
    # Generate specific recommendations based on detected bias types
    for detection in detections:
        if detection.bias_type == VisualBiasType.REPRESENTATION:
            if detection.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                recommendations.append("Increase diverse representation across all roles, especially in prominent positions")
                recommendations.append("Ensure marginalized groups are shown in leadership and active roles, not just supportive positions")
            else:
                recommendations.append("Review representation balance to ensure inclusive diversity")
                
        elif detection.bias_type == VisualBiasType.CONTEXTUAL:
            recommendations.append("Balance spatial positioning - ensure diverse individuals are shown centrally and prominently")
            recommendations.append("Avoid placing certain demographics consistently in background or passive roles")
            
        elif detection.bias_type == VisualBiasType.TOKENISM:
            recommendations.append("Avoid token representation - include meaningful and proportional diversity")
            recommendations.append("Show diverse individuals in various roles and contexts, not as singular representatives")
            
        elif detection.bias_type == VisualBiasType.BODY_REPRESENTATION:
            recommendations.append("Include diverse body types and avoid perpetuating narrow beauty standards")
            recommendations.append("Show people of all body types in positive, active, and professional contexts")
            
        elif detection.bias_type == VisualBiasType.CULTURAL_APPROPRIATION:
            recommendations.append("Review cultural symbols and ensure they're used with appropriate context and respect")
            recommendations.append("Consult with cultural representatives when using specific cultural elements")
            
        elif detection.bias_type == VisualBiasType.ACCESSIBILITY:
            recommendations.append("Adjust brightness and contrast to meet WCAG accessibility standards")
            recommendations.append("Ensure visual content is accessible to people with visual impairments")
    
    # Add diversity-specific recommendations
    gender_dist = diversity_metrics.gender_distribution
    if "unknown" not in gender_dist:
        male_ratio = gender_dist.get("male", 0)
        female_ratio = gender_dist.get("female", 0)
        if male_ratio > 0.7 or female_ratio > 0.7:
            recommendations.append("Balance gender representation - aim for more equal distribution across genders")
    
    if diversity_metrics.power_dynamics_score < 0.4:
        recommendations.append("Address power dynamics imbalance - show diverse individuals in positions of authority")
    
    if diversity_metrics.body_type_diversity < 0.3:
        recommendations.append("Increase body type diversity in visual content")
    
    # If no specific recommendations, provide general best practices
    if not recommendations:
        recommendations.append("Continue monitoring visual content for representation and bias")
        recommendations.append("Maintain diverse and inclusive imagery across all marketing materials")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    ctx.logger.info(f"💡 Generated {len(unique_recommendations)} recommendations")
    return unique_recommendations[:10]  # Limit to top 10 most relevant


# Include protocol
print(f"📋 Registering visual_bias_protocol with agent...")
print(f"📋 Protocol name: {visual_bias_protocol.name}")
print(f"📋 Protocol version: {visual_bias_protocol.version}")
visual_bias_agent.include(visual_bias_protocol, publish_manifest=True)
print(f"✅ Protocol registered successfully")


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

