"""
Ingestion Agent - Ad Bias Detection System

Role: Data Reception, Preprocessing, and Embedding Generation
Responsibilities:
- Receive and validate incoming ad content from frontend API
- Extract and separate multi-modal components (text, images, videos)
- Generate embeddings using appropriate models
- Store embeddings in ChromaDB with metadata
- Route content to specialized analysis agents
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

# Note: Embedding models are now handled by individual agents

# Import ChromaDB
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chroma import ChromaDB

# Import Claude YouTube processor from utils directory
utils_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)  # Insert at beginning to prioritize utils version
from claude_youtube_processor import get_claude_youtube_processor, extract_youtube_content_with_claude

# Import shared models
from agents.shared_models import (
    AdContentRequest,
    EmbeddingPackage,
    IngestionAcknowledgement,
    ContentType,
    BiasCategory
)

# Claude preprocessing models
class ClaudePreprocessRequest(Model):
    """Request for Claude to preprocess ad content"""
    request_id: str
    text_content: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    frames_count: Optional[int] = None
    content_type: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class ClaudePreprocessResponse(Model):
    """Claude's response after preprocessing ad content"""
    request_id: str
    processed_text: Optional[str] = None
    text_analysis: Optional[Dict[str, Any]] = None
    visual_analysis: Optional[Dict[str, Any]] = None
    preprocessing_notes: Optional[str] = None
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# Initialize Ingestion Agent
ingestion_agent = Agent(
    name="ingestion_agent",
    seed="ad_bias_ingestion_agent_unique_seed_2024",
    port=8100,
    endpoint=["http://localhost:8100/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for ingestion
ingestion_protocol = Protocol(name="ingestion_protocol", version="1.0")

# Protocol for Claude preprocessing
claude_preprocess_protocol = Protocol(name="claude_preprocess_protocol", version="1.0")


# ChromaDB instance (for agents to use)
_chroma_db = None

def get_chroma_db():
    """Get ChromaDB instance"""
    global _chroma_db
    if _chroma_db is None:
        _chroma_db = ChromaDB()
    return _chroma_db


# Agent addresses for routing
TEXT_BIAS_AGENT_ADDRESS = "agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv"
VISUAL_BIAS_AGENT_ADDRESS = "agent1qtnatq0rhrj2pauyg2a8dgf56uqkf6tw3757z806w6c57zkw9nry2my2933"

# ASI:ONE LLM agent addresses for Claude preprocessing
CLAUDE_AGENT = 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx'

# Environment variables for ASI:ONE integration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")


@ingestion_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Ingestion Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {ingestion_agent.address}")
    ctx.logger.info(f"🔧 Role: Content Extraction, Claude Preprocessing, and Agent Routing")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8100/submit")
    
    # Check API keys
    if ANTHROPIC_API_KEY:
        ctx.logger.info(f"✅ ANTHROPIC_API_KEY: Configured")
    else:
        ctx.logger.error(f"❌ ANTHROPIC_API_KEY: Not set - Claude preprocessing will fail")
    
    if AGENTVERSE_API_KEY:
        ctx.logger.info(f"✅ AGENTVERSE_API_KEY: Configured")
    else:
        ctx.logger.error(f"❌ AGENTVERSE_API_KEY: Not set - Agent communication will fail")
    
    ctx.logger.info(f"🧠 Claude Integration: {CLAUDE_AGENT}")
    ctx.logger.info(f"📤 Text Agent: {TEXT_BIAS_AGENT_ADDRESS}")
    ctx.logger.info(f"👁️ Visual Agent: {VISUAL_BIAS_AGENT_ADDRESS}")
    ctx.logger.info(f"✅ Claude YouTube Processor: Ready for intelligent video processing")
    ctx.logger.info(f"⚡ Ready to extract content and route to bias analysis agents")


@ingestion_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Ingestion Agent shutting down...")
    ctx.logger.info("🧹 Cleaning up resources...")


async def process_ad_content(ctx: Context, msg: AdContentRequest) -> IngestionAcknowledgement:
    """
    Shared processing logic for ad content (used by both protocol and REST)
    Following the correct architecture: Extract → Claude Preprocess → Route to Agents
    """
    try:
        ctx.logger.info(f"📨 Received ad content request: {msg.request_id}")
        ctx.logger.info(f"📊 Content type: {msg.content_type}")
        
        # Store request in context storage
        ctx.storage.set(msg.request_id, {
            "content_type": msg.content_type.value,
            "received_at": datetime.now(UTC).isoformat()
        })
        
        # Step 1: Extract content (YouTube transcript + frames)
        ctx.logger.info(f"🔄 Extracting content for request {msg.request_id}...")
        extracted_data = await extract_content(ctx, msg)
        
        # Step 2: Claude Preprocessing (as per architecture diagram)
        ctx.logger.info(f"🧠 Claude preprocessing for request {msg.request_id}...")
        preprocessed_data = await claude_preprocess_content(ctx, extracted_data, msg.request_id)
        
        # Step 3: Create embedding package with raw content (no embeddings yet)
        # Convert frames to base64 if they exist
        frames_base64 = None
        if "frames" in extracted_data and extracted_data["frames"]:
            ctx.logger.info(f"🖼️  Converting {len(extracted_data['frames'])} Claude-selected frames to base64...")
            claude_processor = get_claude_youtube_processor()
            frames_base64 = claude_processor.frames_to_base64(extracted_data["frames"])
            ctx.logger.info(f"✅ Claude-selected frames converted to base64")
        
        embedding_package = EmbeddingPackage(
            request_id=msg.request_id,
            text_content=preprocessed_data.get("text"),  # Claude-processed text
            text_embedding=None,  # Let Text Bias Agent generate this
            visual_embedding=None,  # Let Visual Bias Agent generate this
            frames_base64=frames_base64,  # Raw video frames
            chromadb_collection_id=None,  # Let agents handle ChromaDB storage
            content_type=msg.content_type,
            metadata=preprocessed_data.get("metadata")
        )
        
        # Step 4: Route to analysis agents (they'll handle embeddings + ChromaDB)
        ctx.logger.info(f"🔀 Routing content to analysis agents...")
        await route_to_analysis_agents(ctx, embedding_package, msg)
        
        ctx.logger.info(f"✅ Request {msg.request_id} processed successfully")
        
        return IngestionAcknowledgement(
            request_id=msg.request_id,
            status="success",
            message=f"Content extracted, preprocessed with Claude, and routed to analysis agents"
        )
        
    except Exception as e:
        ctx.logger.error(f"❌ Error processing request {msg.request_id}: {e}")
        return IngestionAcknowledgement(
            request_id=msg.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )


@ingestion_protocol.on_message(model=AdContentRequest, replies=IngestionAcknowledgement)
async def handle_ad_content(ctx: Context, sender: str, msg: AdContentRequest):
    """
    Handle incoming ad content from other agents via protocol
    """
    acknowledgement = await process_ad_content(ctx, msg)
    await ctx.send(sender, acknowledgement)


@claude_preprocess_protocol.on_message(model=ClaudePreprocessResponse, replies=None)
async def handle_claude_response(ctx: Context, sender: str, msg: ClaudePreprocessResponse):
    """
    Handle Claude preprocessing responses (if needed for async processing)
    """
    ctx.logger.info(f"📨 Received Claude preprocessing response for request {msg.request_id}")
    ctx.logger.info(f"   - Confidence: {msg.confidence_score}")
    ctx.logger.info(f"   - Processing time: {msg.processing_time_ms}ms")


@ingestion_agent.on_rest_post("/analyze", AdContentRequest, IngestionAcknowledgement)
async def handle_rest_analyze(ctx: Context, req: AdContentRequest) -> IngestionAcknowledgement:
    """
    REST endpoint for FastAPI to send content directly
    Usage: POST http://localhost:8100/analyze
    """
    ctx.logger.info(f"🌐 REST request received: {req.request_id}")
    return await process_ad_content(ctx, req)


def extract_youtube_content(youtube_url: str) -> Dict[str, Any]:
    """
    Extract transcript, frames, and metadata from YouTube video
    
    Returns:
        Dict with:
        - success: bool
        - video_id: str
        - transcript: dict with text and metadata
        - frames: list of numpy arrays
        - thumbnail_url: str
        - metadata: dict with video info
    """
    try:
        processor = get_youtube_processor()
        result = processor.process_youtube_video(youtube_url)
        
        return {
            "success": True,
            "video_id": result["video_id"],
            "transcript": {
                "success": True,
                "text": result["transcript"],
                "language": "en",  # YouTube API provides this
                "is_generated": False
            },
            "frames": result["frames"],
            "num_frames": result["num_frames"],
            "thumbnail_url": f"https://img.youtube.com/vi/{result['video_id']}/maxresdefault.jpg",
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "transcript": {
                "success": False,
                "error": str(e)
            }
        }


async def extract_content(ctx: Context, content: AdContentRequest) -> Dict[str, Any]:
    """
    Extract raw content from the request (YouTube transcript + frames).
    This is the first step before Claude preprocessing.
    """
    extracted = {
        "text": content.text_content.strip() if content.text_content else None,
        "image_url": content.image_url,
        "video_url": content.video_url,
        "frames": [],
        "metadata": content.metadata or {}
    }

    # If it's a YouTube video, extract content using Claude
    if content.video_url and ("youtube.com" in content.video_url or "youtu.be" in content.video_url):
        ctx.logger.info(f"🎬 Detected YouTube URL, extracting content with Claude...")

        try:
            youtube_data = extract_youtube_content_with_claude(content.video_url)

            if youtube_data["success"]:
                ctx.logger.info(f"✅ YouTube content extracted successfully with Claude")

                # Store raw transcript
                transcript_text = youtube_data.get("transcript", "")
                if transcript_text:
                    extracted["text"] = transcript_text
                    ctx.logger.info(f"   ✅ TRANSCRIPT EXTRACTED:")
                    ctx.logger.info(f"      Length: {len(transcript_text)} characters")
                    ctx.logger.info(f"      Preview: {transcript_text[:200]}...")
                    ctx.logger.info(f"   📄 FULL TRANSCRIPT:")
                    ctx.logger.info(f"   {'-' * 40}")
                    ctx.logger.info(f"   {transcript_text}")
                    ctx.logger.info(f"   {'-' * 40}")
                    
                    # Save transcript to extracted_frames folder
                    video_id = youtube_data.get("video_id", "unknown")
                    frames_dir = os.path.join(os.path.dirname(__file__), "..", "extracted_frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    transcript_filename = f"transcript_{video_id}.txt"
                    transcript_path = os.path.join(frames_dir, transcript_filename)
                    
                    try:
                        with open(transcript_path, 'w', encoding='utf-8') as f:
                            f.write(transcript_text)
                        ctx.logger.info(f"   💾 TRANSCRIPT SAVED to: {transcript_path}")
                    except Exception as e:
                        ctx.logger.error(f"   ❌ Failed to save transcript: {e}")
                else:
                    ctx.logger.warning(f"   - No transcript available")

                # Store Claude-analyzed transcript
                transcript_analysis = youtube_data.get("transcript_analysis", {})
                if transcript_analysis and "error" not in transcript_analysis:
                    extracted["metadata"]["claude_transcript_analysis"] = transcript_analysis
                    ctx.logger.info(f"   🧠 CLAUDE TRANSCRIPT ANALYSIS:")
                    ctx.logger.info(f"      Overall bias score: {transcript_analysis.get('overall_bias_score', 'N/A')}")
                    ctx.logger.info(f"      Summary: {transcript_analysis.get('summary', 'N/A')[:100]}...")

                # Store Claude-selected bias-relevant frames
                extracted["frames"] = youtube_data.get("frames", [])
                extracted["num_frames"] = youtube_data.get("num_frames", 0)
                ctx.logger.info(f"   ✅ CLAUDE-SELECTED BIAS-RELEVANT FRAMES:")
                ctx.logger.info(f"      Count: {youtube_data.get('num_frames', 0)} frames")
                if youtube_data.get("frames"):
                    ctx.logger.info(f"      Size: {youtube_data['frames'][0].shape if youtube_data['frames'] else 'N/A'}")
                    ctx.logger.info(f"      Format: numpy arrays (BGR, ready for analysis)")
                    ctx.logger.info(f"      Selection: Claude identified most bias-relevant frames")
                
                # Store thumbnail URL
                video_id = youtube_data.get("video_id", "unknown")
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                extracted["image_url"] = thumbnail_url
                ctx.logger.info(f"   - Thumbnail URL: {thumbnail_url}")

                # Add YouTube metadata
                metadata = youtube_data.get("metadata", {})
                extracted["metadata"]["youtube"] = {
                    "video_id": video_id,
                    "title": metadata.get("title"),
                    "author": metadata.get("author"),
                    "duration": metadata.get("duration"),
                    "views": metadata.get("views"),
                    "description": metadata.get("description", ""),
                    "has_transcript": bool(transcript_text),
                    "num_frames": youtube_data.get("num_frames", 0),
                    "claude_processed": True
                }

            else:
                ctx.logger.error(f"❌ Failed to extract YouTube content with Claude: {youtube_data['error']}")
                extracted["metadata"]["youtube_error"] = youtube_data["error"]

        except Exception as e:
            ctx.logger.error(f"❌ Error processing YouTube video with Claude: {e}")
            extracted["metadata"]["youtube_error"] = str(e)

    ctx.logger.info(f"✅ Content extracted for request {content.request_id}")
    return extracted


async def claude_preprocess_content(ctx: Context, extracted_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Use Claude to preprocess the extracted content (as per architecture diagram).
    This is where Claude analyzes and prepares the content for the bias agents.

    NOTE: For YouTube videos, Claude analysis is already done in extract_youtube_content_with_claude,
    so we skip additional preprocessing to avoid redundancy and REST context limitations.
    """
    ctx.logger.info(f"🧠 Claude preprocessing for request {request_id}...")

    # Check if we already have Claude analysis from YouTube extraction
    if extracted_data.get("metadata", {}).get("youtube", {}).get("claude_processed"):
        ctx.logger.info(f"✅ Claude preprocessing already done during YouTube extraction - using existing analysis")

        # Use the already-processed content
        preprocessed = {
            "text": extracted_data.get("text"),
            "image_url": extracted_data.get("image_url"),
            "video_url": extracted_data.get("video_url"),
            "frames": extracted_data.get("frames", []),
            "metadata": extracted_data.get("metadata", {})
        }

        # Add preprocessing status to metadata
        preprocessed["metadata"]["claude_preprocessing"] = {
            "preprocessing_status": "completed_with_youtube_extraction",
            "processed_at": datetime.now(UTC).isoformat()
        }

        return preprocessed

    # For non-YouTube content or if we need additional preprocessing via agent communication
    # Check if we have the send_and_wait capability (not available in REST context)
    if not hasattr(ctx, 'send_and_wait'):
        ctx.logger.warning(f"⚠️ Agent communication not available in REST context - skipping additional Claude preprocessing")
        ctx.logger.info(f"✅ Using extracted content as-is (Claude preprocessing skipped)")

        preprocessed = {
            "text": extracted_data.get("text"),
            "image_url": extracted_data.get("image_url"),
            "video_url": extracted_data.get("video_url"),
            "frames": extracted_data.get("frames", []),
            "metadata": extracted_data.get("metadata", {})
        }

        preprocessed["metadata"]["claude_preprocessing"] = {
            "preprocessing_status": "skipped_rest_context",
            "processed_at": datetime.now(UTC).isoformat()
        }

        return preprocessed

    # Full agent-based preprocessing (for protocol messages with full context)
    ctx.logger.info(f"📤 Sending content to Claude agent for preprocessing...")

    # Check if API keys are configured
    if not ANTHROPIC_API_KEY:
        raise Exception("ANTHROPIC_API_KEY not configured. Claude preprocessing requires Anthropic API key.")

    if not AGENTVERSE_API_KEY:
        raise Exception("AGENTVERSE_API_KEY not configured. Agent communication requires Agentverse API key.")

    # Prepare Claude request
    claude_request = ClaudePreprocessRequest(
        request_id=request_id,
        text_content=extracted_data.get("text"),
        image_url=extracted_data.get("image_url"),
        video_url=extracted_data.get("video_url"),
        frames_count=len(extracted_data.get("frames", [])),
        content_type="youtube_video" if extracted_data.get("video_url") else "text",
        metadata=extracted_data.get("metadata", {})
    )

    # Send to Claude agent and wait for response
    claude_response = await ctx.send_and_wait(
        CLAUDE_AGENT,
        claude_request,
        timeout=30.0,
        response_type=ClaudePreprocessResponse
    )

    if not claude_response:
        raise Exception(f"Claude preprocessing timed out after 30 seconds for request {request_id}")

    ctx.logger.info(f"✅ Claude agent preprocessing completed successfully")
    ctx.logger.info(f"   - Confidence score: {claude_response.confidence_score}")
    ctx.logger.info(f"   - Processing time: {claude_response.processing_time_ms}ms")

    # Use Claude's processed content
    preprocessed = {
        "text": claude_response.processed_text or extracted_data.get("text"),
        "image_url": extracted_data.get("image_url"),
        "video_url": extracted_data.get("video_url"),
        "frames": extracted_data.get("frames", []),
        "metadata": extracted_data.get("metadata", {})
    }

    # Add Claude preprocessing results to metadata
    preprocessed["metadata"]["claude_preprocessing"] = {
        "processed_at": claude_response.timestamp,
        "confidence_score": claude_response.confidence_score,
        "processing_time_ms": claude_response.processing_time_ms,
        "text_analysis": claude_response.text_analysis,
        "visual_analysis": claude_response.visual_analysis,
        "preprocessing_notes": claude_response.preprocessing_notes,
        "preprocessing_status": "completed_with_claude_agent"
    }

    return preprocessed




async def preprocess_content(ctx: Context, content: AdContentRequest) -> Dict[str, Any]:
    """
    Preprocess and clean incoming content.
    For YouTube videos, extract transcript and metadata.
    """
    preprocessed = {
        "text": content.text_content.strip() if content.text_content else None,
        "image_url": content.image_url,
        "video_url": content.video_url,
        "metadata": content.metadata or {}
    }

    # If it's a YouTube video, extract content
    if content.video_url and ("youtube.com" in content.video_url or "youtu.be" in content.video_url):
        ctx.logger.info(f"🎬 Detected YouTube URL, extracting content...")

        try:
            youtube_data = extract_youtube_content(content.video_url)

            if youtube_data["success"]:
                ctx.logger.info(f"✅ YouTube content extracted successfully")

                # Use transcript as text content if available
                if youtube_data["transcript"]["success"]:
                    transcript_text = youtube_data["transcript"]["text"]
                    preprocessed["text"] = transcript_text
                    ctx.logger.info(f"   ✅ TRANSCRIPT EXTRACTED:")
                    ctx.logger.info(f"      Length: {len(transcript_text)} characters")
                    ctx.logger.info(f"      Preview: {transcript_text[:200]}...")
                else:
                    ctx.logger.warning(f"   - No transcript available: {youtube_data['transcript']['error']}")

                # Store frames for visual analysis
                preprocessed["frames"] = youtube_data["frames"]
                preprocessed["num_frames"] = youtube_data["num_frames"]
                ctx.logger.info(f"   ✅ FRAMES EXTRACTED:")
                ctx.logger.info(f"      Count: {youtube_data['num_frames']} frames")
                ctx.logger.info(f"      Size: {youtube_data['frames'][0].shape if youtube_data['frames'] else 'N/A'}")
                ctx.logger.info(f"      Format: numpy arrays (BGR, ready for analysis)")
                
                # Use thumbnail as image URL
                preprocessed["image_url"] = youtube_data["thumbnail_url"]
                ctx.logger.info(f"   - Thumbnail URL: {youtube_data['thumbnail_url']}")

                # Add YouTube metadata
                preprocessed["metadata"]["youtube"] = {
                    "video_id": youtube_data["video_id"],
                    "title": youtube_data["metadata"].get("title"),
                    "author": youtube_data["metadata"].get("author"),
                    "duration": youtube_data["metadata"].get("duration"),
                    "views": youtube_data["metadata"].get("views"),
                    "description": youtube_data["metadata"].get("description", ""),
                    "has_transcript": youtube_data["transcript"]["success"],
                    "transcript_language": youtube_data["transcript"].get("language"),
                    "is_auto_generated": youtube_data["transcript"].get("is_generated"),
                    "num_frames": youtube_data["num_frames"]
                }

            else:
                ctx.logger.error(f"❌ Failed to extract YouTube content: {youtube_data['error']}")
                preprocessed["metadata"]["youtube_error"] = youtube_data["error"]

        except Exception as e:
            ctx.logger.error(f"❌ Error processing YouTube video: {e}")
            preprocessed["metadata"]["youtube_error"] = str(e)

    ctx.logger.info(f"✅ Content preprocessed for request {content.request_id}")
    return preprocessed


# Note: Embedding generation and ChromaDB storage are now handled by individual agents
# The Ingestion Agent only extracts content and routes it to specialized agents


async def route_to_analysis_agents(
    ctx: Context,
    embedding_package: EmbeddingPackage,
    original_content: AdContentRequest
):
    """
    Route content to Text and Visual Bias agents for parallel analysis.
    Each agent receives the raw content and handles their own embedding generation.
    """
    routed_count = 0

    # Route to Text Bias Agent if text content exists
    if embedding_package.text_content and TEXT_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing TEXT content to Text Bias Agent: {TEXT_BIAS_AGENT_ADDRESS}")
        ctx.logger.info(f"   - Text length: {len(embedding_package.text_content)} characters")
        await ctx.send(TEXT_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1

    # Route to Visual Bias Agent if visual content exists (frames or image)
    has_visual_content = (embedding_package.frames_base64 and len(embedding_package.frames_base64) > 0) or embedding_package.metadata.get("image_url")
    if has_visual_content and VISUAL_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing VISUAL content to Visual Bias Agent: {VISUAL_BIAS_AGENT_ADDRESS}")
        if embedding_package.frames_base64:
            ctx.logger.info(f"   - Frames count: {len(embedding_package.frames_base64)}")
        if embedding_package.metadata.get("image_url"):
            ctx.logger.info(f"   - Image URL: {embedding_package.metadata.get('image_url')}")
        await ctx.send(VISUAL_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1

    if routed_count == 0:
        ctx.logger.warning(f"⚠️ No analysis agents configured or no content to analyze.")
        ctx.logger.warning(f"   - Text content: {embedding_package.text_content is not None}")
        ctx.logger.warning(f"   - Visual content (frames): {embedding_package.frames_base64 is not None and len(embedding_package.frames_base64 or []) > 0}")
        ctx.logger.warning(f"   - Visual content (image): {embedding_package.metadata.get('image_url') is not None}")
        ctx.logger.warning(f"   - Text agent address: {TEXT_BIAS_AGENT_ADDRESS}")
        ctx.logger.warning(f"   - Visual agent address: {VISUAL_BIAS_AGENT_ADDRESS}")
    else:
        ctx.logger.info(f"✅ Content routed to {routed_count} analysis agent(s)")


# Flexible REST endpoint for FastAPI (handles string content_type)
class FlexibleAdContentRequest(Model):
    """Flexible request model that accepts string content_type"""
    request_id: str
    content_type: str  # Accept string instead of enum
    text_content: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


@ingestion_agent.on_rest_post("/submit", FlexibleAdContentRequest, IngestionAcknowledgement)
async def rest_submit_content(ctx: Context, req: FlexibleAdContentRequest) -> IngestionAcknowledgement:
    """
    REST endpoint for receiving ad content from FastAPI backend.
    This allows FastAPI to submit content via HTTP POST.
    Uses the same flow as the protocol handler: Extract → Claude Preprocess → Route to Agents
    """
    ctx.logger.info(f"📨 REST API: Received content request: {req.request_id}")
    ctx.logger.info(f"   - Content type: {req.content_type}")
    ctx.logger.info(f"   - Video URL: {req.video_url}")
    ctx.logger.info(f"   - Text content: {req.text_content is not None}")

    try:
        # Convert flexible request to proper AdContentRequest
        content_type_enum = ContentType(req.content_type) if req.content_type in [e.value for e in ContentType] else ContentType.VIDEO
        
        proper_request = AdContentRequest(
            request_id=req.request_id,
            content_type=content_type_enum,
            text_content=req.text_content,
            image_url=req.image_url,
            video_url=req.video_url,
            metadata=req.metadata,
            timestamp=req.timestamp
        )
        
        # Use the same processing logic as the protocol handler
        return await process_ad_content(ctx, proper_request)

    except Exception as e:
        ctx.logger.error(f"❌ Error processing REST request {req.request_id}: {e}")
        return IngestionAcknowledgement(
            request_id=req.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )


# Include protocols
ingestion_agent.include(ingestion_protocol, publish_manifest=True)
ingestion_agent.include(claude_preprocess_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🚀 INGESTION AGENT - Ad Bias Detection              ║
╚══════════════════════════════════════════════════════════════╝

Role: Content Extraction, Claude Preprocessing, and Agent Routing

Architecture Flow:
  1. Extract YouTube content (transcript + frames)
  2. Claude preprocessing and analysis
  3. Route raw content to specialized agents
  4. Agents handle their own embeddings + ChromaDB

Capabilities:
  ✓ Extracts YouTube transcripts and video frames
  ✓ Claude-powered content preprocessing
  ✓ Routes text content to Text Bias Agent
  ✓ Routes visual content to Visual Bias Agent
  ✓ 1-minute video duration limit enforcement

Endpoints:
  • Agent Protocol: http://localhost:8100/submit
  • REST API: http://localhost:8100/submit

📍 Waiting for YouTube URLs...
🛑 Stop with Ctrl+C
    """)
    ingestion_agent.run()

