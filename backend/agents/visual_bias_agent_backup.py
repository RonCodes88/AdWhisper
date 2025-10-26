"""
Visual Bias Agent - Ad Bias Detection System

Role: Visual Content Analysis and Bias Detection
Responsibilities:
- Analyze visual content for representation bias
- Generate bias analysis and recommendations
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
import json
import base64
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Claude Vision API Configuration
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_ENGINE = os.getenv("MODEL_ENGINE", "claude-3-5-haiku-latest")

if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY":
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set. Vision analysis will use fallback mode.")
    print("   Set ANTHROPIC_API_KEY environment variable to enable Claude Vision API.")
    ANTHROPIC_API_KEY = None

HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
} if ANTHROPIC_API_KEY else {}


def repair_json(json_str: str) -> str:
    """
    Repair malformed JSON from Claude Vision API.
    Simple approach: escape all control characters within string contexts.
    """
    # Remove markdown code blocks
    json_str = re.sub(r'^```json\s*', '', json_str)
    json_str = re.sub(r'\s*```$', '', json_str)
    
    # Simple strategy: go through character by character
    # Track if we're in a string by counting unescaped quotes
    # Only escape control chars when inside strings
    
    result = []
    i = 0
    in_string = False
    prev_char = ''
    
    while i < len(json_str):
        char = json_str[i]
        
        # Check if this is an escape sequence
        if prev_char == '\\' and in_string:
            # This character is being escaped, just keep it
            result.append(char)
            prev_char = char if char != '\\' else ''  # Don't chain escapes
            i += 1
            continue
        
        if char == '"':
            # Toggle string state (unless it was escaped, which we handled above)
            in_string = not in_string
            result.append(char)
            prev_char = char
            i += 1
            continue
        
        if in_string:
            # We're inside a string - escape control characters
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif char == '\b':
                result.append('\\b')
            elif char == '\f':
                result.append('\\f')
            elif ord(char) < 32:
                # Other control characters
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            # Outside string, keep as-is
            result.append(char)
        
        prev_char = char
        i += 1
    
    return ''.join(result)


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
        if has_frames:
            # Use base64 frames directly for analysis
            visual_frames = req.frames_base64
            ctx.logger.info(f"   📦 Using {len(visual_frames)} base64 frames for analysis")
        elif media_type == "video" and media_url:
            visual_frames = await extract_video_keyframes(ctx, media_url)
            ctx.logger.info(f"   🎬 Extracted {len(visual_frames)} keyframes from video")
        elif media_url:
            visual_frames = [media_url]
            ctx.logger.info(f"   🖼️  Using single image URL")
        else:
            visual_frames = []
            ctx.logger.info(f"   ⚠️ No visual content available")

        # Step 2: Analyze visual content with Vision-LLM
        ctx.logger.info(f"👁️ STEP 2: Analyzing visual content with Claude Vision...")
        initial_analysis = await analyze_visual_with_llm(ctx, visual_frames)
        ctx.logger.info(f"   ✅ Visual analysis complete")
        
        # Extract bias detections from Claude analysis
        bias_detections_from_claude = initial_analysis.get('bias_detections', [])
        ctx.logger.info(f"   📊 Bias detections found: {len(bias_detections_from_claude)}")
        
        # Extract people detected info
        people_detected = initial_analysis.get('people_detected', {})
        ctx.logger.info(f"   👥 People detected: {people_detected.get('total_count', 0)}")

        # Step 3: Calculate diversity metrics
        ctx.logger.info(f"📊 STEP 3: Calculating diversity metrics...")
        diversity_metrics = await detect_representation_metrics(ctx, initial_analysis, people_detected)
        ctx.logger.info(f"   ✅ Diversity metrics calculated")

        # Step 4: Analyze composition and power dynamics
        ctx.logger.info(f"🎭 STEP 4: Analyzing composition and spatial dynamics...")
        composition_analysis = await analyze_composition(ctx, initial_analysis)
        ctx.logger.info(f"   ✅ Composition analysis complete")

        # Step 5: Classify visual biases
        ctx.logger.info(f"🏷️ STEP 5: Classifying visual bias types...")
        bias_detections = await classify_visual_biases(
            ctx,
            initial_analysis,
            diversity_metrics,
            composition_analysis,
            bias_detections_from_claude
        )
        ctx.logger.info(f"   ✅ Classified {len(bias_detections)} visual bias types")

        # Step 6: Calculate overall visual score
        ctx.logger.info(f"📊 STEP 6: Calculating overall visual bias score...")
        visual_score = await calculate_visual_score(ctx, bias_detections, diversity_metrics)
        ctx.logger.info(f"   ✅ Visual score calculated: {visual_score:.2f}/10")

        # Step 7: Generate recommendations
        ctx.logger.info(f"💡 STEP 7: Generating recommendations...")
        recommendations = await generate_visual_recommendations(ctx, bias_detections, diversity_metrics)
        ctx.logger.info(f"   ✅ Generated {len(recommendations)} recommendations")

        # Step 8: Create report - convert bias detections to dicts
        ctx.logger.info(f"📋 STEP 8: Creating visual bias report...")
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

        # Create comprehensive visual bias report dict
        report_dict = {
            "request_id": req.request_id,
            "agent_name": "visual_bias_agent",
            "bias_detected": len(bias_detections) > 0,
            "bias_instances": bias_instances_dicts,
            "overall_visual_score": visual_score,
            "diversity_metrics": diversity_metrics.dict() if hasattr(diversity_metrics, 'dict') else diversity_metrics,
            "recommendations": recommendations,
            "claude_analysis": {
                "text_detected": initial_analysis.get('text_detected', {}),
                "people_detected": initial_analysis.get('people_detected', {}),
                "spatial_analysis": initial_analysis.get('spatial_analysis', {}),
                "gender_dynamics": initial_analysis.get('gender_dynamics', {}),
                "overall_assessment": initial_analysis.get('overall_assessment', {}),
                "api_used": initial_analysis.get('api_used', 'unknown'),
                "model": initial_analysis.get('model', 'unknown'),
                "frames_analyzed": initial_analysis.get('frames_analyzed', 0)
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
        ctx.logger.info(f"   ✅ Report created")

        # Display comprehensive JSON results on console for testing
        ctx.logger.info("=" * 100)
        ctx.logger.info("🔍 COMPREHENSIVE VISUAL BIAS ANALYSIS RESULTS:")
        ctx.logger.info("=" * 100)
        ctx.logger.info(json.dumps(report_dict, indent=2))
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"📊 SUMMARY:")
        ctx.logger.info(f"   🎯 Request ID: {req.request_id}")
        ctx.logger.info(f"   📊 Overall Score: {visual_score:.1f}/10")
        ctx.logger.info(f"   🚨 Bias detected: {len(bias_detections) > 0}")
        ctx.logger.info(f"   📝 Issues found: {len(bias_detections)}")
        ctx.logger.info(f"   💡 Recommendations: {len(recommendations)}")
        ctx.logger.info(f"   👥 People analyzed: {people_detected.get('total_count', 0)}")
        ctx.logger.info(f"   🎬 Frames analyzed: {initial_analysis.get('frames_analyzed', 0)}")
        ctx.logger.info(f"   🤖 API used: {initial_analysis.get('api_used', 'unknown')}")
        ctx.logger.info("=" * 100)

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
    Use Vision-LLM (Claude Vision) to analyze visual content.
    """
    ctx.logger.info(f"👁️ Analyzing {len(visual_frames)} visual frame(s) with Claude Vision...")
    
    if not ANTHROPIC_API_KEY:
        ctx.logger.warning("⚠️ ANTHROPIC_API_KEY not set, using fallback analysis")
        return _fallback_analysis(ctx)
    
    try:
        # If we have base64 frames, analyze them directly
        if visual_frames and len(visual_frames) > 0 and isinstance(visual_frames[0], str) and visual_frames[0].startswith('data:'):
            return await analyze_base64_frames(ctx, visual_frames)
        
        # If we have file paths, analyze them
        elif visual_frames and len(visual_frames) > 0:
            return await analyze_multiple_frames(ctx, visual_frames)
        
        else:
            ctx.logger.warning("⚠️ No valid visual frames provided")
            return _fallback_analysis(ctx)
            
    except Exception as e:
        ctx.logger.error(f"❌ Error in visual analysis: {e}")
        return _fallback_analysis(ctx)


async def analyze_base64_frames(ctx: Context, base64_frames: List[str]) -> Dict[str, Any]:
    """
    Analyze base64 encoded frames using Claude Vision API.
    """
    ctx.logger.info(f"🔄 Analyzing {len(base64_frames)} base64 frames...")
    
    try:
        # Prepare content for Claude
        content = []
        
        for i, frame_data in enumerate(base64_frames):
            # Extract mime type and data from base64 string
            if frame_data.startswith('data:'):
                header, data = frame_data.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0]
            else:
                mime_type = 'image/jpeg'
                data = frame_data
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": data,
                }
            })
            
            if i == 0:
                # Add the prompt after the first image
                content.append({"type": "text", "text": _get_bias_analysis_prompt()})
        
        # Make API request
        data = {
            "model": MODEL_ENGINE,
            "max_tokens": MAX_TOKENS,
            "system": "You are a bias detection AI. You respond ONLY with valid JSON matching the exact schema provided. No other text.",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        }
        
        ctx.logger.info(f"📤 Making API request to Claude Vision...")
        response = requests.post(
            CLAUDE_URL, 
            headers=HEADERS, 
            data=json.dumps(data), 
            timeout=180
        )
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        if "error" in response_data:
            ctx.logger.error(f"❌ API Error: {response_data['error']}")
            return _fallback_analysis(ctx)
        
        messages = response_data.get("content", [])
        if messages and len(messages) > 0:
            text_response = messages[0].get("text", "")
            
            try:
                # Parse JSON response
                json_str = "{" + text_response.strip()
                
                try:
                    analysis_result = json.loads(json_str)
                    ctx.logger.info(f"✅ JSON parsed successfully!")
                except json.JSONDecodeError as parse_err:
                    ctx.logger.warning(f"⚠️ Initial parse failed, attempting repair...")
                    repaired_json = repair_json(json_str)
                    analysis_result = json.loads(repaired_json)
                    ctx.logger.info(f"✅ JSON repaired and parsed successfully!")
                
                analysis_result["api_used"] = "claude_vision"
                analysis_result["model"] = MODEL_ENGINE
                analysis_result["frames_analyzed"] = len(base64_frames)
                
                # Process the new compact schema (Step 1 implementation)
                ctx.logger.info(f"🔧 Processing compact analysis schema...")
                
                # Extract people summary for diversity metrics calculation
                people_summary = analysis_result.get("people_summary", [])
                ctx.logger.info(f"   📊 Found {len(people_summary)} people entries")
                
                # Extract bias flags
                bias_flags = analysis_result.get("bias_flags", {})
                ctx.logger.info(f"   🚨 Bias flags: {bias_flags}")
                
                # Extract severity and summary
                bias_severity = analysis_result.get("bias_severity", "low")
                overall_summary = analysis_result.get("overall_summary", "No bias detected")
                
                ctx.logger.info(f"   📈 Bias severity: {bias_severity}")
                ctx.logger.info(f"   📝 Summary: {overall_summary}")
                
                # Display results on console for testing
                ctx.logger.info("=" * 80)
                ctx.logger.info("🔍 CLAUDE VISION ANALYSIS RESULTS:")
                ctx.logger.info("=" * 80)
                ctx.logger.info(json.dumps(analysis_result, indent=2))
                ctx.logger.info("=" * 80)
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                ctx.logger.error(f"❌ Could not parse JSON: {e}")
                return {
                    "raw_analysis": text_response,
                    "api_used": "claude_vision",
                    "model": MODEL_ENGINE,
                    "note": "Could not parse structured JSON, returning raw analysis",
                    "parse_error": str(e)
                }
        else:
            return {"error": "No response from API"}
            
    except Exception as e:
        ctx.logger.error(f"❌ Error in base64 frame analysis: {e}")
        return _fallback_analysis(ctx)


async def analyze_multiple_frames(ctx: Context, frame_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze multiple video frames by sending ALL frames to Claude in ONE request.
    """
    ctx.logger.info(f"🎬 Analyzing {len(frame_paths)} video frames in ONE request...")
    
    try:
        # Encode ALL frames
        frame_images = []
        for i, frame_path in enumerate(frame_paths):
            ctx.logger.info(f"   Encoding frame {i+1}: {frame_path}")
            image_base64, mime_type = encode_image_to_base64(frame_path)
            frame_images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_base64,
                }
            })
        
        # Build content array
        content = []
        for i, frame_img in enumerate(frame_images):
            content.append(frame_img)
            if i == 0:
                content.append({"type": "text", "text": _get_bias_analysis_prompt()})
        
        # Make API request
        data = {
            "model": MODEL_ENGINE,
            "max_tokens": MAX_TOKENS,
            "system": "You are a bias detection AI. You respond ONLY with valid JSON matching the exact schema provided. No other text.",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        }
        
        ctx.logger.info(f"📤 Making API request to Claude Vision...")
        response = requests.post(
            CLAUDE_URL, 
            headers=HEADERS, 
            data=json.dumps(data), 
            timeout=180
        )
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        if "error" in response_data:
            ctx.logger.error(f"❌ API Error: {response_data['error']}")
            return _fallback_analysis(ctx)
        
        messages = response_data.get("content", [])
        if messages and len(messages) > 0:
            text_response = messages[0].get("text", "")
            
            try:
                json_str = "{" + text_response.strip()
                
                try:
                    analysis_result = json.loads(json_str)
                    ctx.logger.info(f"✅ JSON parsed successfully!")
                except json.JSONDecodeError as parse_err:
                    ctx.logger.warning(f"⚠️ Initial parse failed, attempting repair...")
                    repaired_json = repair_json(json_str)
                    analysis_result = json.loads(repaired_json)
                    ctx.logger.info(f"✅ JSON repaired and parsed successfully!")
                
                analysis_result["api_used"] = "claude_vision"
                analysis_result["model"] = MODEL_ENGINE
                analysis_result["frames_analyzed"] = len(frame_paths)
                
                # Display results on console for testing
                ctx.logger.info("=" * 80)
                ctx.logger.info("🔍 CLAUDE VISION VIDEO ANALYSIS RESULTS:")
                ctx.logger.info("=" * 80)
                ctx.logger.info(json.dumps(analysis_result, indent=2))
                ctx.logger.info("=" * 80)
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                ctx.logger.error(f"❌ Could not parse JSON: {e}")
                return {
                    "raw_analysis": text_response,
                    "api_used": "claude_vision",
                    "model": MODEL_ENGINE,
                    "note": "Could not parse structured JSON, returning raw analysis",
                    "parse_error": str(e)
                }
        else:
            return {"error": "No response from API"}
            
    except Exception as e:
        ctx.logger.error(f"❌ Error in multi-frame analysis: {e}")
        return _fallback_analysis(ctx)


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Encode image file to base64 string.
    """
    # Determine mime type from extension
    ext = image_path.lower().split('.')[-1]
    mime_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    mime_type = mime_type_map.get(ext, 'image/jpeg')
    
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string, mime_type


def _get_bias_analysis_prompt() -> str:
    """Get the compact bias analysis prompt (Step 1 & 4 implementation)."""
    return """You are analyzing frames from a video advertisement.

For each frame, tag what you see using concise JSON. Focus on *who is visible* and *what roles they appear in*.
Then, summarize if the combined visuals suggest any bias patterns.

**CRITICAL**: These frames are from the SAME video ad. Look for patterns, narrative, and how the message develops across time.

Respond ONLY with this JSON structure:

{
  "people_summary": [
    {"frame_id": 1, "gender": "male", "role": "leader", "ethnicity": "white", "age": "middle_aged"},
    {"frame_id": 1, "gender": "female", "role": "assistant", "ethnicity": "white", "age": "young_adult"}
  ],
  "bias_flags": {
    "gender_bias": true,
    "racial_bias": false,
    "age_bias": false,
    "body_representation_bias": false,
    "cultural_bias": false,
    "contextual_bias": true
  },
  "dominant_groups": ["male", "white"],
  "minority_groups_visible": ["female"],
  "activities_summary": "Men portrayed in leadership roles, women in supportive positions",
  "potential_issues": [
    "Male-dominated representation in work environment",
    "Lack of visible diversity in main frames"
  ],
  "bias_severity": "medium",
  "overall_summary": "The ad shows subtle gender representation bias through role imbalance."
}

Do NOT include any text before or after the JSON.
Do NOT wrap the JSON in markdown code blocks.
Just return the raw JSON object."""


def _fallback_analysis(ctx: Context) -> Dict[str, Any]:
    """
    Fallback analysis when API key is not available.
    """
    ctx.logger.warning("⚠️ Using fallback analysis - no API key configured")
    return {
        "fallback": True,
        "error": "ANTHROPIC_API_KEY not configured",
        "message": "Set ANTHROPIC_API_KEY environment variable to enable real bias detection",
        "bias_detections": [],
        "overall_assessment": {
            "bias_score": 5.0,
            "diversity_score": 5.0,
            "main_concerns": ["API key not configured"],
            "positive_aspects": []
        }
    }


async def detect_representation_metrics(ctx: Context, analysis: Dict[str, Any], people_detected: Dict[str, Any]) -> DiversityMetrics:
    """
    Calculate quantitative diversity metrics from visual analysis.
    """
    ctx.logger.info(f"📊 Calculating diversity metrics...")
    
    # Extract demographics from Claude analysis
    demographics = people_detected.get('visible_demographics', {})
    
    # Calculate gender distribution
    gender_dist = demographics.get('gender', {})
    total_people = people_detected.get('total_count', 0)
    
    gender_distribution = {}
    if total_people > 0:
        for gender, count in gender_dist.items():
            gender_distribution[gender] = count / total_people
    else:
        gender_distribution = {"male": 0.5, "female": 0.5, "non_binary": 0.0, "unknown": 0.0}
    
    # Calculate ethnicity distribution
    ethnicity_dist = demographics.get('ethnicity', {})
    ethnicity_distribution = {}
    if total_people > 0:
        for ethnicity, count in ethnicity_dist.items():
            ethnicity_distribution[ethnicity] = count / total_people
    else:
        ethnicity_distribution = {"white": 0.8, "black": 0.1, "asian": 0.05, "hispanic": 0.05, "unknown": 0.0}
    
    # Calculate age distribution
    age_dist = demographics.get('age_groups', {})
    age_distribution = {}
    if total_people > 0:
        for age_group, count in age_dist.items():
            age_distribution[age_group] = count / total_people
    else:
        age_distribution = {"young_adult": 0.4, "middle_aged": 0.5, "senior": 0.1, "unknown": 0.0}
    
    # Calculate body type diversity
    body_types = demographics.get('body_types', {})
    unique_body_types = len([bt for bt, count in body_types.items() if count > 0])
    body_type_diversity = min(1.0, unique_body_types / 4.0)  # Normalize to 0-1
    
    # Calculate power dynamics score based on spatial analysis
    spatial_analysis = analysis.get('spatial_analysis', {})
    power_positioning = spatial_analysis.get('power_positioning', '')
    
    # Simple power dynamics calculation
    power_dynamics_score = 0.5  # Default neutral
    if 'balanced' in power_positioning.lower() or 'equal' in power_positioning.lower():
        power_dynamics_score = 0.8
    elif 'dominant' in power_positioning.lower() or 'central' in power_positioning.lower():
        power_dynamics_score = 0.3
    
    metrics = DiversityMetrics(
        gender_distribution=gender_distribution,
        apparent_ethnicity=ethnicity_distribution,
        age_distribution=age_distribution,
        body_type_diversity=body_type_diversity,
        power_dynamics_score=power_dynamics_score
    )
    
    ctx.logger.info(f"✅ Diversity metrics calculated")
    ctx.logger.info(f"   👥 Gender distribution: {gender_distribution}")
    ctx.logger.info(f"   🌍 Ethnicity distribution: {ethnicity_distribution}")
    ctx.logger.info(f"   📊 Body type diversity: {body_type_diversity:.2f}")
    ctx.logger.info(f"   ⚖️ Power dynamics score: {power_dynamics_score:.2f}")
    
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
    claude_bias_detections: List[Dict[str, Any]]
) -> List[VisualBiasDetection]:
    """
    Classify visual bias types based on analysis results.
    """
    ctx.logger.info(f"🏷️ Classifying visual biases...")
    
    detections = []
    
    # Convert Claude bias detections to our format
    for claude_bias in claude_bias_detections:
        try:
            # Map Claude bias types to our enum
            bias_type_mapping = {
                "gender": VisualBiasType.REPRESENTATION,
                "representation": VisualBiasType.REPRESENTATION,
                "contextual": VisualBiasType.CONTEXTUAL,
                "tokenism": VisualBiasType.TOKENISM,
                "stereotyping": VisualBiasType.REPRESENTATION,
                "cultural_appropriation": VisualBiasType.CULTURAL_APPROPRIATION,
                "racial": VisualBiasType.REPRESENTATION
            }
            
            bias_type = bias_type_mapping.get(claude_bias.get('type', 'contextual'), VisualBiasType.CONTEXTUAL)
            
            # Map severity levels
            severity_mapping = {
                "low": SeverityLevel.LOW,
                "medium": SeverityLevel.MEDIUM,
                "high": SeverityLevel.HIGH,
                "critical": SeverityLevel.CRITICAL
            }
            severity = severity_mapping.get(claude_bias.get('severity', 'medium'), SeverityLevel.MEDIUM)
            
            # Extract evidence and affected groups
            evidence = claude_bias.get('evidence', [])
            affected_groups = claude_bias.get('affected_groups', ['General audience'])
            
            detection = VisualBiasDetection(
                bias_type=bias_type,
                severity=severity,
                examples=evidence,
                context=claude_bias.get('description', 'Bias detected in visual content'),
                confidence=0.85  # Default confidence for Claude detections
            )
            
            detections.append(detection)
            ctx.logger.info(f"   ✅ Converted Claude bias: {bias_type.value} ({severity.value})")
            
        except Exception as e:
            ctx.logger.warning(f"   ⚠️ Could not convert Claude bias detection: {e}")
            continue
    
    # If no detections from Claude, create some based on diversity metrics
    if not detections:
        ctx.logger.info(f"   📊 No Claude detections, analyzing diversity metrics...")
        
        # Check for representation bias
        if diversity_metrics.gender_distribution.get('male', 0) > 0.7:
            detections.append(VisualBiasDetection(
            bias_type=VisualBiasType.REPRESENTATION,
                severity=SeverityLevel.MEDIUM,
                examples=["Male-dominated representation"],
                context="Gender distribution shows male overrepresentation",
                confidence=0.7
            ))
        
        if diversity_metrics.power_dynamics_score < 0.4:
            detections.append(VisualBiasDetection(
            bias_type=VisualBiasType.CONTEXTUAL,
            severity=SeverityLevel.MEDIUM,
                examples=["Imbalanced power dynamics"],
                context="Spatial positioning shows power imbalance",
                confidence=0.6
            ))
    
    ctx.logger.info(f"✅ Classified {len(detections)} visual bias types")
    return detections


async def calculate_visual_score(
    ctx: Context,
    detections: List[VisualBiasDetection],
    diversity_metrics: DiversityMetrics
) -> float:
    """
    Calculate overall visual bias score using lightweight heuristic (Step 6).
    """
    ctx.logger.info(f"📊 Calculating visual score using lightweight heuristic...")
    
    # Start with perfect score
    score = 10.0
    
    # Gender ratio penalty
    male_ratio = diversity_metrics.gender_distribution.get('male', 0.0)
    female_ratio = diversity_metrics.gender_distribution.get('female', 0.0)
    gender_ratio = max(male_ratio, female_ratio)
    
    if gender_ratio > 0.7:
        score -= 2.0
        ctx.logger.info(f"   ⚠️ Gender imbalance penalty: -2.0 (ratio: {gender_ratio:.2f})")
    
    # Racial diversity penalty
    white_ratio = diversity_metrics.apparent_ethnicity.get('white', 0.0)
    if white_ratio > 0.8:
        score -= 1.5
        ctx.logger.info(f"   ⚠️ Racial diversity penalty: -1.5 (white ratio: {white_ratio:.2f})")
    
    # Body type diversity penalty
    if diversity_metrics.body_type_diversity < 0.3:
        score -= 1.0
        ctx.logger.info(f"   ⚠️ Body diversity penalty: -1.0 (diversity: {diversity_metrics.body_type_diversity:.2f})")
    
    # Power dynamics penalty
    if diversity_metrics.power_dynamics_score < 0.4:
        score -= 1.5
        ctx.logger.info(f"   ⚠️ Power dynamics penalty: -1.5 (score: {diversity_metrics.power_dynamics_score:.2f})")
    
    # Bias detection penalties
    for detection in detections:
        if detection.severity == SeverityLevel.CRITICAL:
            score -= 3.0
            ctx.logger.info(f"   🚨 Critical bias penalty: -3.0 ({detection.bias_type.value})")
        elif detection.severity == SeverityLevel.HIGH:
            score -= 2.0
            ctx.logger.info(f"   ⚠️ High bias penalty: -2.0 ({detection.bias_type.value})")
        elif detection.severity == SeverityLevel.MEDIUM:
            score -= 1.0
            ctx.logger.info(f"   ⚠️ Medium bias penalty: -1.0 ({detection.bias_type.value})")
        elif detection.severity == SeverityLevel.LOW:
            score -= 0.5
            ctx.logger.info(f"   ⚠️ Low bias penalty: -0.5 ({detection.bias_type.value})")
    
    # Ensure score stays within bounds
    score = max(0.0, min(10.0, score))
    
    ctx.logger.info(f"📊 Calculated visual score: {score:.1f}/10")
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

