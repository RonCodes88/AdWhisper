"""
YouTube Ingestion Agent using uAgents Framework
Simplified version that just processes YouTube videos with Claude
"""

import os
import sys
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from uagents import Agent, Context, Model

# Load environment variables from .env file
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from claude_youtube_processor import extract_youtube_content_with_claude

# Environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")

# Pydantic models for communication
class YouTubeProcessingRequest(Model):
    """Request to process a YouTube video"""
    request_id: str
    video_url: str
    metadata: Optional[Dict[str, Any]] = None

class YouTubeProcessingResponse(Model):
    """Response from YouTube processing"""
    request_id: str
    success: bool
    transcript: str = ""
    transcript_analysis: Dict[str, Any] = {}
    frames_base64: list = []
    num_frames: int = 0
    metadata: Dict[str, Any] = {}
    error: str = ""
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)

# Initialize YouTube Ingestion Agent
youtube_agent = Agent(
    name="youtube_ingestion_agent",
    seed="youtube-ingestion-agent-seed-phrase-for-deterministic-address",
    port=8101,
    endpoint=["http://127.0.0.1:8101/submit"]
)

@youtube_agent.on_event("startup")
async def startup(ctx: Context):
    """Agent startup event"""
    ctx.logger.info("🚀 YouTube Ingestion Agent started!")
    ctx.logger.info(f"📍 Agent address: {youtube_agent.address}")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8100/submit")
    
    # Check API keys
    if ANTHROPIC_API_KEY:
        ctx.logger.info("✅ ANTHROPIC_API_KEY: Configured")
    else:
        ctx.logger.error("❌ ANTHROPIC_API_KEY: Not set - Claude processing will fail")
    
    if AGENTVERSE_API_KEY:
        ctx.logger.info("✅ AGENTVERSE_API_KEY: Configured")
    else:
        ctx.logger.warning("⚠️ AGENTVERSE_API_KEY: Not set - Agent communication may fail")
    
    ctx.logger.info("🎬 Ready to process YouTube videos with Claude!")

@youtube_agent.on_message(model=YouTubeProcessingRequest)
async def process_youtube_video(ctx: Context, sender: str, msg: YouTubeProcessingRequest):
    """
    Process YouTube video with Claude for bias analysis
    """
    ctx.logger.info(f"🎬 Processing YouTube video request: {msg.request_id}")
    ctx.logger.info(f"🔗 URL: {msg.video_url}")
    
    try:
        # Check if Anthropic API key is set
        if not ANTHROPIC_API_KEY:
            error_msg = "ANTHROPIC_API_KEY not configured. Claude processing requires Anthropic API key."
            ctx.logger.error(f"❌ {error_msg}")
            
            response = YouTubeProcessingResponse(
                request_id=msg.request_id,
                success=False,
                error=error_msg
            )
            await ctx.send(sender, response)
            return
        
        # Process the YouTube video with Claude
        ctx.logger.info("🧠 Processing with Claude...")
        result = extract_youtube_content_with_claude(msg.video_url)
        
        if result["success"]:
            ctx.logger.info("✅ Processing successful!")
            ctx.logger.info(f"   📝 Transcript: {len(result['transcript'])} chars")
            ctx.logger.info(f"   🎬 Frames: {result['num_frames']}")
            ctx.logger.info(f"   🧠 Claude analysis: {'✅' if 'transcript_analysis' in result else '❌'}")
            
            response = YouTubeProcessingResponse(
                request_id=msg.request_id,
                success=True,
                transcript=result.get("transcript", ""),
                transcript_analysis=result.get("transcript_analysis", {}),
                frames_base64=result.get("frames_base64", []),
                num_frames=result.get("num_frames", 0),
                metadata=result.get("metadata", {}),
                error=""
            )
        else:
            ctx.logger.error(f"❌ Processing failed: {result['error']}")
            response = YouTubeProcessingResponse(
                request_id=msg.request_id,
                success=False,
                error=result["error"]
            )
        
        # Send response back
        await ctx.send(sender, response)
        
    except Exception as e:
        ctx.logger.error(f"❌ Error processing YouTube video: {e}")
        response = YouTubeProcessingResponse(
            request_id=msg.request_id,
            success=False,
            error=str(e)
        )
        await ctx.send(sender, response)

# REST endpoint for HTTP requests
@youtube_agent.on_rest_post("/process-youtube", YouTubeProcessingRequest, YouTubeProcessingResponse)
async def process_youtube_rest(ctx: Context, request: YouTubeProcessingRequest):
    """
    REST endpoint for processing YouTube videos
    """
    ctx.logger.info(f"🌐 REST request received: {request.request_id}")
    
    # Process the request
    result = extract_youtube_content_with_claude(request.video_url)
    
    if result["success"]:
        return YouTubeProcessingResponse(
            request_id=request.request_id,
            success=True,
            transcript=result.get("transcript", ""),
            transcript_analysis=result.get("transcript_analysis", {}),
            frames_base64=result.get("frames_base64", []),
            num_frames=result.get("num_frames", 0),
            metadata=result.get("metadata", {}),
            error=""
        )
    else:
        return YouTubeProcessingResponse(
            request_id=request.request_id,
            success=False,
            error=result["error"]
        )

if __name__ == "__main__":
    print("🚀 Starting YouTube Ingestion Agent (uAgents Framework)")
    print("=" * 60)
    print("✅ Uses uAgents framework")
    print("✅ Requires AGENTVERSE_API_KEY for agent communication")
    print("✅ Requires ANTHROPIC_API_KEY for Claude processing")
    print("=" * 60)
    
    # Check API keys
    if not ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY not set!")
        print("📝 Set it with: export ANTHROPIC_API_KEY='your_key_here'")
        print("🔗 Get your key from: https://console.anthropic.com/")
        sys.exit(1)
    
    if not AGENTVERSE_API_KEY:
        print("❌ AGENTVERSE_API_KEY not set!")
        print("📝 Set it with: export AGENTVERSE_API_KEY='your_key_here'")
        print("🔗 Get your key from: https://agentverse.ai/")
        sys.exit(1)
    
    print("✅ All API keys are set!")
    print("🌐 Starting agent on http://localhost:8100")
    print("🛑 Stop with Ctrl+C")
    
    youtube_agent.run()
