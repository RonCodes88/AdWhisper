"""
Simplified Ingestion Agent - AdWhisper

Role: YouTube Content Extraction and Agent Routing
Responsibilities:
- Receive YouTube URL from FastAPI
- Extract transcript and frames using Claude
- Route text to Text Bias Agent
- Route frames to Visual Bias Agent

Following Fetch.ai uAgents standards for clean, simple agent communication.
"""

from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Add utils to path for Claude YouTube processor
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.append(utils_path)
from claude_youtube_processor import extract_youtube_content_with_claude

# Import simplified models (relative import from agents directory)
from simple_shared_models import (
    TextAnalysisRequest,
    VisualAnalysisRequest,
    AgentError
)

# Environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

ingestion_agent = Agent(
    name="simple_ingestion_agent",
    seed="simple_ingestion_agent_seed_2024",
    port=8100,
    endpoint=["http://localhost:8100/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for ingestion
ingestion_protocol = Protocol(name="simple_ingestion_protocol", version="1.0")

# Agent addresses (from Agentverse/deterministic seeds)
TEXT_BIAS_AGENT_ADDRESS = os.getenv("TEXT_BIAS_AGENT_ADDRESS", "agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv")
VISUAL_BIAS_AGENT_ADDRESS = os.getenv("VISUAL_BIAS_AGENT_ADDRESS", "agent1qtnatq0rhrj2pauyg2a8dgf56uqkf6tw3757z806w6c57zkw9nry2my2933")


# ============================================================================
# REST REQUEST/RESPONSE MODELS
# ============================================================================

class YouTubeIngestionRequest(Model):
    """REST request from FastAPI"""
    request_id: str = Field(..., description="Unique request identifier")
    video_url: str = Field(..., description="YouTube video URL")
    timestamp: str = Field(default="", description="Request timestamp")

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class IngestionAcknowledgement(Model):
    """Acknowledgement response"""
    request_id: str = Field(..., description="Request identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(default="", description="Response timestamp")

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@ingestion_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Simple Ingestion Agent started!")
    ctx.logger.info(f"📍 Agent address: {ingestion_agent.address}")
    ctx.logger.info(f"🌐 REST endpoint: http://localhost:8100/analyze")
    ctx.logger.info(f"🔧 Role: YouTube extraction + routing")

    # Check API keys
    if ANTHROPIC_API_KEY:
        ctx.logger.info(f"✅ ANTHROPIC_API_KEY: Configured")
    else:
        ctx.logger.error(f"❌ ANTHROPIC_API_KEY: Not set!")

    ctx.logger.info(f"📤 Text Agent: {TEXT_BIAS_AGENT_ADDRESS}")
    ctx.logger.info(f"👁️ Visual Agent: {VISUAL_BIAS_AGENT_ADDRESS}")


@ingestion_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Simple Ingestion Agent shutting down...")


# ============================================================================
# REST ENDPOINT
# ============================================================================

@ingestion_agent.on_rest_post("/analyze", YouTubeIngestionRequest, IngestionAcknowledgement)
async def handle_youtube_ingestion(ctx: Context, req: YouTubeIngestionRequest) -> IngestionAcknowledgement:
    """
    REST endpoint for FastAPI to submit YouTube URLs

    Flow:
    1. Extract YouTube content with Claude (transcript + frames)
    2. Send text → Text Bias Agent
    3. Send frames → Visual Bias Agent
    4. Return acknowledgement immediately
    """
    ctx.logger.info(f"📨 REST request received: {req.request_id}")
    ctx.logger.info(f"🔗 YouTube URL: {req.video_url}")

    try:
        # Step 1: Extract YouTube content with Claude
        ctx.logger.info(f"🎬 Extracting YouTube content with Claude...")
        youtube_data = extract_youtube_content_with_claude(req.video_url)

        if not youtube_data["success"]:
            error_msg = youtube_data.get("error", "Unknown error")
            ctx.logger.error(f"❌ YouTube extraction failed: {error_msg}")
            return IngestionAcknowledgement(
                request_id=req.request_id,
                status="error",
                message=f"YouTube extraction failed: {error_msg}"
            )

        # Extract data
        transcript = youtube_data.get("transcript", "")
        frames_base64 = youtube_data.get("frames_base64", [])
        num_frames = youtube_data.get("num_frames", 0)
        metadata = youtube_data.get("metadata", {})
        transcript_analysis = youtube_data.get("transcript_analysis", {})

        ctx.logger.info(f"✅ Extraction complete:")
        ctx.logger.info(f"   📝 Transcript: {len(transcript)} chars")
        ctx.logger.info(f"   🎬 Frames: {num_frames}")
        ctx.logger.info(f"   🧠 Claude analysis: {'✅' if transcript_analysis else '❌'}")

        # Step 2: Send text to Text Bias Agent
        if transcript and TEXT_BIAS_AGENT_ADDRESS:
            ctx.logger.info(f"📤 Sending text to Text Bias Agent...")

            text_request = TextAnalysisRequest(
                request_id=req.request_id,
                text_content=transcript,
                metadata={
                    "video_url": req.video_url,
                    "youtube_metadata": metadata,
                    "claude_analysis": transcript_analysis
                }
            )

            await ctx.send(TEXT_BIAS_AGENT_ADDRESS, text_request)
            ctx.logger.info(f"✅ Text sent to Text Bias Agent")
        else:
            ctx.logger.warning(f"⚠️ No transcript or Text Agent address not set")

        # Step 3: Send frames to Visual Bias Agent
        if frames_base64 and VISUAL_BIAS_AGENT_ADDRESS:
            ctx.logger.info(f"📤 Sending {num_frames} frames to Visual Bias Agent...")

            visual_request = VisualAnalysisRequest(
                request_id=req.request_id,
                frames_base64=frames_base64,
                num_frames=num_frames,
                metadata={
                    "video_url": req.video_url,
                    "youtube_metadata": metadata
                }
            )

            await ctx.send(VISUAL_BIAS_AGENT_ADDRESS, visual_request)
            ctx.logger.info(f"✅ Frames sent to Visual Bias Agent")
        else:
            ctx.logger.warning(f"⚠️ No frames or Visual Agent address not set")

        # Step 4: Return acknowledgement
        ctx.logger.info(f"✅ Request {req.request_id} processed successfully")

        return IngestionAcknowledgement(
            request_id=req.request_id,
            status="success",
            message=f"YouTube content extracted and routed to bias analysis agents"
        )

    except Exception as e:
        ctx.logger.error(f"❌ Error processing request: {e}")
        return IngestionAcknowledgement(
            request_id=req.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )


# ============================================================================
# INCLUDE PROTOCOLS
# ============================================================================

ingestion_agent.include(ingestion_protocol, publish_manifest=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🚀 Simple Ingestion Agent - AdWhisper               ║
╚══════════════════════════════════════════════════════════════╝

Role: YouTube Content Extraction and Agent Routing

Flow:
  1. Receive YouTube URL from FastAPI
  2. Extract transcript + frames with Claude
  3. Send text → Text Bias Agent
  4. Send frames → Visual Bias Agent
  5. Return acknowledgement

Endpoints:
  • POST /analyze - YouTube ingestion endpoint

Running on: http://localhost:8100
🛑 Stop with Ctrl+C
    """)
    ingestion_agent.run()
