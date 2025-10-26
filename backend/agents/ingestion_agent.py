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
import aiohttp
import asyncio
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


@ingestion_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Ingestion Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {ingestion_agent.address}")
    ctx.logger.info(f"🔧 Role: Content Extraction and Agent Routing")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8100/submit")
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
        ctx.logger.info("=" * 100)
        ctx.logger.info("🎬 BACKGROUND TASK: PROCESSING AD CONTENT")
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"📨 Received ad content request: {msg.request_id}")
        ctx.logger.info(f"📊 Content type: {msg.content_type}")
        ctx.logger.info(f"🔗 Video URL: {msg.video_url}")
        ctx.logger.info(f"📄 Text content: {msg.text_content is not None}")

        # Store request in context storage
        ctx.storage.set(msg.request_id, {
            "content_type": msg.content_type.value,
            "received_at": datetime.now(UTC).isoformat()
        })
        ctx.logger.info(f"✅ Request stored in context storage")

        # Step 1: Extract content (YouTube transcript + frames)
        ctx.logger.info("")
        ctx.logger.info("=" * 100)
        ctx.logger.info("STEP 1: EXTRACTING CONTENT")
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"🔄 Extracting content for request {msg.request_id}...")
        extracted_data = await extract_content(ctx, msg)
        ctx.logger.info(f"✅ Content extraction complete!")
        ctx.logger.info(f"   📊 Keys in extracted data: {list(extracted_data.keys())}")
        ctx.logger.info(f"   📄 Has text: {extracted_data.get('text') is not None}")
        ctx.logger.info(f"   🖼️  Has frames: {'frames' in extracted_data and len(extracted_data.get('frames', [])) > 0}")
        if 'frames' in extracted_data:
            ctx.logger.info(f"   🎬 Number of frames: {len(extracted_data.get('frames', []))}")

        # Step 2: Claude Preprocessing (as per architecture diagram)
        ctx.logger.info("")
        ctx.logger.info("=" * 100)
        ctx.logger.info("STEP 2: CLAUDE PREPROCESSING")
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"🧠 Claude preprocessing for request {msg.request_id}...")
        preprocessed_data = await claude_preprocess_content(ctx, extracted_data, msg.request_id)
        ctx.logger.info(f"✅ Claude preprocessing complete!")
        ctx.logger.info(f"   📊 Keys in preprocessed data: {list(preprocessed_data.keys())}")

        # Step 3: Create embedding package with raw content (no embeddings yet)
        ctx.logger.info("")
        ctx.logger.info("=" * 100)
        ctx.logger.info("STEP 3: CREATING EMBEDDING PACKAGE")
        ctx.logger.info("=" * 100)
        # Convert frames to base64 if they exist
        frames_base64 = None
        if "frames" in extracted_data and extracted_data["frames"]:
            ctx.logger.info(f"🖼️  Converting {len(extracted_data['frames'])} Claude-selected frames to base64...")
            claude_processor = get_claude_youtube_processor()
            frames_base64 = claude_processor.frames_to_base64(extracted_data["frames"])
            ctx.logger.info(f"✅ Claude-selected frames converted to base64")
            ctx.logger.info(f"   📏 Base64 frames count: {len(frames_base64)}")
            ctx.logger.info(f"   📦 First frame base64 length: {len(frames_base64[0]) if frames_base64 else 0}")
        else:
            ctx.logger.info(f"⚠️  No frames to convert to base64")

        ctx.logger.info(f"📦 Creating EmbeddingPackage...")
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
        ctx.logger.info(f"✅ EmbeddingPackage created!")
        ctx.logger.info(f"   📝 Request ID: {embedding_package.request_id}")
        ctx.logger.info(f"   📄 Has text content: {embedding_package.text_content is not None}")
        if embedding_package.text_content:
            ctx.logger.info(f"   📏 Text length: {len(embedding_package.text_content)} chars")
        ctx.logger.info(f"   🖼️  Has frames: {embedding_package.frames_base64 is not None and len(embedding_package.frames_base64 or []) > 0}")
        if embedding_package.frames_base64:
            ctx.logger.info(f"   🎬 Frames count: {len(embedding_package.frames_base64)}")
        ctx.logger.info(f"   📊 Has metadata: {embedding_package.metadata is not None}")

        # Step 4: Route to analysis agents (they'll handle embeddings + ChromaDB)
        ctx.logger.info("")
        ctx.logger.info("=" * 100)
        ctx.logger.info("STEP 4: ROUTING TO ANALYSIS AGENTS")
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"🔀 About to call route_to_analysis_agents()...")
        ctx.logger.info(f"   🎯 This will send HTTP POST requests to Text and Visual Bias Agents")
        await route_to_analysis_agents(ctx, embedding_package, msg)
        ctx.logger.info(f"✅ Returned from route_to_analysis_agents()")

        ctx.logger.info("")
        ctx.logger.info("=" * 100)
        ctx.logger.info(f"✅✅✅ REQUEST {msg.request_id} FULLY PROCESSED ✅✅✅")
        ctx.logger.info("=" * 100)
        
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


@ingestion_agent.on_rest_post("/analyze", AdContentRequest, Dict[str, Any])
async def handle_rest_analyze(ctx: Context, req: AdContentRequest) -> Dict[str, Any]:
    """
    REST endpoint for FastAPI to send content directly.
    Returns the final scored bias report directly.
    Usage: POST http://localhost:8100/analyze
    """
    ctx.logger.info(f"🌐 REST request received: {req.request_id}")
    final_report = await process_ad_content(ctx, req)
    
    if final_report:
        # Return the complete scored report
        return final_report
    else:
        # Return error if scoring failed
        return {
            "request_id": req.request_id,
            "status": "error",
            "message": "Failed to generate final bias report"
        }


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
    Prepare extracted content for bias analysis.

    For YouTube videos, Claude analysis is already done in extract_youtube_content_with_claude.
    For other content, we pass it through as-is (agents will do their own analysis).
    """
    ctx.logger.info(f"🧠 Preparing content for bias analysis: {request_id}...")

    # Use the already-extracted content
    preprocessed = {
        "text": extracted_data.get("text"),
        "image_url": extracted_data.get("image_url"),
        "video_url": extracted_data.get("video_url"),
        "frames": extracted_data.get("frames", []),
        "metadata": extracted_data.get("metadata", {})
    }

    # Check if we already have Claude analysis from YouTube extraction
    if extracted_data.get("metadata", {}).get("youtube", {}).get("claude_processed"):
        ctx.logger.info(f"✅ Claude preprocessing already done during YouTube extraction")
        preprocessed["metadata"]["claude_preprocessing"] = {
            "preprocessing_status": "completed_with_youtube_extraction",
            "processed_at": datetime.now(UTC).isoformat()
        }
    else:
        ctx.logger.info(f"✅ Using extracted content as-is - agents will perform analysis")
        preprocessed["metadata"]["claude_preprocessing"] = {
            "preprocessing_status": "skipped_no_preprocessing_needed",
            "processed_at": datetime.now(UTC).isoformat()
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
    Route content to Text and Visual Bias agents via REST endpoints.
    Wait for both responses, then pass them to Scoring Agent.
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("📨 ROUTING CONTENT TO ANALYSIS AGENTS (REST)")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"🔍 Analyzing embedding package to route:")
    ctx.logger.info(f"   📝 Request ID: {embedding_package.request_id}")
    ctx.logger.info(f"   📄 Text content present: {embedding_package.text_content is not None}")
    if embedding_package.text_content:
        ctx.logger.info(f"   📏 Text length: {len(embedding_package.text_content)} chars")
        ctx.logger.info(f"   📖 Text preview: {embedding_package.text_content[:100]}...")
    ctx.logger.info(f"   🖼️  Frames present: {embedding_package.frames_base64 is not None and len(embedding_package.frames_base64 or []) > 0}")
    if embedding_package.frames_base64:
        ctx.logger.info(f"   🎬 Number of frames: {len(embedding_package.frames_base64)}")

    # Convert embedding_package to dict for JSON serialization
    ctx.logger.info(f"📦 Converting EmbeddingPackage to dict for JSON serialization...")
    package_dict = {
        "request_id": embedding_package.request_id,
        "text_content": embedding_package.text_content,
        "text_embedding": embedding_package.text_embedding,
        "frames_base64": embedding_package.frames_base64,
        "visual_embedding": embedding_package.visual_embedding,
        "chromadb_collection_id": embedding_package.chromadb_collection_id,
        "content_type": embedding_package.content_type.value if hasattr(embedding_package.content_type, 'value') else embedding_package.content_type,
        "metadata": embedding_package.metadata,
        "timestamp": embedding_package.timestamp
    }
    ctx.logger.info(f"✅ Package dict created with keys: {list(package_dict.keys())}")

    # Initialize variables to store reports
    text_report = None
    visual_report = None

    # Prepare URLs
    text_agent_url = "http://localhost:8101/analyze"
    visual_agent_url = "http://localhost:8102/analyze"

    # Flags to determine which calls to make
    should_call_text = bool(embedding_package.text_content)
    has_visual_content = (embedding_package.frames_base64 and len(embedding_package.frames_base64) > 0) or (embedding_package.metadata and embedding_package.metadata.get("image_url"))

    ctx.logger.info("")
    ctx.logger.info("🔍🔍🔍 STEP 1+2: CALLING TEXT & VISUAL BIAS AGENTS (in parallel) 🔍🔍🔍")

    async def call_text_agent(session: aiohttp.ClientSession):
        nonlocal text_report
        if not should_call_text:
            ctx.logger.warning(f"⚠️ SKIPPING TEXT BIAS AGENT - No text content to send")
            return
        ctx.logger.info("")
        ctx.logger.info("=" * 80)
        ctx.logger.info("📤 SENDING HTTP REQUEST TO TEXT BIAS AGENT")
        ctx.logger.info("=" * 80)
        ctx.logger.info(f"   🎯 Endpoint URL: {text_agent_url}")
        ctx.logger.info(f"   📏 Text content length: {len(embedding_package.text_content)} characters")
        ctx.logger.info(f"   📝 Request ID: {embedding_package.request_id}")
        ctx.logger.info(f"   ⏱️  Timeout: 120 seconds")
        try:
            ctx.logger.info(f"📡 Making POST request to {text_agent_url}...")
            async with session.post(text_agent_url, json=package_dict, timeout=aiohttp.ClientTimeout(total=120)) as response:
                ctx.logger.info(f"📥 Received response from Text Bias Agent!")
                ctx.logger.info(f"   📊 Status code: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    candidate = result.get('report', {})
                    # Require expected fields to consider it valid
                    if isinstance(candidate, dict) and ("overall_text_score" in candidate or "bias_instances" in candidate):
                        text_report = candidate
                        ctx.logger.info(f"✅ SUCCESS - Text Bias Agent analysis complete")
                        ctx.logger.info(f"   📊 Text Score: {text_report.get('overall_text_score', 'N/A')}")
                        ctx.logger.info(f"   📋 Issues Found: {len(text_report.get('bias_instances', []))}")
                    else:
                        ctx.logger.error(f"   ❌ Text report missing expected fields, will not trigger scoring from this report")
                else:
                    error_text = await response.text()
                    ctx.logger.error(f"   ❌ Text Bias Agent returned status: {response.status}")
                    ctx.logger.error(f"   📄 Error response: {error_text}")
        except Exception as e:
            ctx.logger.error(f"❌ ERROR calling Text Bias Agent: {e}")
            import traceback
            ctx.logger.error(traceback.format_exc())

    async def call_visual_agent(session: aiohttp.ClientSession):
        nonlocal visual_report
        if not has_visual_content:
            ctx.logger.warning(f"⚠️ SKIPPING VISUAL BIAS AGENT - No visual content to send")
            return
        ctx.logger.info("")
        ctx.logger.info("=" * 80)
        ctx.logger.info("📤 SENDING HTTP REQUEST TO VISUAL BIAS AGENT")
        ctx.logger.info("=" * 80)
        ctx.logger.info(f"   🎯 Endpoint URL: {visual_agent_url}")
        if embedding_package.frames_base64:
            ctx.logger.info(f"   🖼️  Frames count: {len(embedding_package.frames_base64)}")
        ctx.logger.info(f"   📝 Request ID: {embedding_package.request_id}")
        ctx.logger.info(f"   ⏱️  Timeout: 180 seconds")
        try:
            ctx.logger.info(f"📡 Making POST request to {visual_agent_url}...")
            async with session.post(visual_agent_url, json=package_dict, timeout=aiohttp.ClientTimeout(total=180)) as response:
                ctx.logger.info(f"📥 Received response from Visual Bias Agent!")
                ctx.logger.info(f"   📊 Status code: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    candidate = result.get('report', {})
                    # Require expected fields to consider it valid
                    if isinstance(candidate, dict) and ("overall_visual_score" in candidate or "bias_instances" in candidate):
                        visual_report = candidate
                        ctx.logger.info(f"✅ SUCCESS - Visual Bias Agent analysis complete")
                        ctx.logger.info(f"   📊 Visual Score: {visual_report.get('overall_visual_score', 'N/A')}")
                        ctx.logger.info(f"   📋 Issues Found: {len(visual_report.get('bias_instances', []))}")
                    else:
                        ctx.logger.error(f"   ❌ Visual report missing expected fields, will not trigger scoring from this report")
                else:
                    error_text = await response.text()
                    ctx.logger.error(f"   ❌ Visual Bias Agent returned status: {response.status}")
                    ctx.logger.error(f"   📄 Error response: {error_text}")
        except Exception as e:
            ctx.logger.error(f"❌ ERROR calling Visual Bias Agent: {e}")
            import traceback
            ctx.logger.error(traceback.format_exc())

    # Execute selected calls in parallel within a single session
    async with aiohttp.ClientSession() as session:
        tasks = []
        if should_call_text:
            tasks.append(call_text_agent(session))
        if has_visual_content:
            tasks.append(call_visual_agent(session))
        if tasks:
            await asyncio.gather(*tasks)

    # Step 3: Call Scoring Agent ONLY when BOTH reports are available
    ctx.logger.info("")
    ctx.logger.info("=" * 80)
    ctx.logger.info("🔍🔍🔍 STEP 3: CHECKING IF BOTH REPORTS ARE READY 🔍🔍🔍")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"   📊 Text Report: {'✅ Available' if text_report else '❌ Missing'}")
    ctx.logger.info(f"   📊 Visual Report: {'✅ Available' if visual_report else '❌ Missing'}")
    
    # Call scoring agent when at least one valid report is available
    final_scored_report = None
    if text_report or visual_report:
        ctx.logger.info(f"✅ Reports available - proceeding to scoring...")
        
        scoring_agent_url = "http://localhost:8103/score"
        ctx.logger.info(f"   🎯 Endpoint URL: {scoring_agent_url}")
        ctx.logger.info(f"   📝 Request ID: {embedding_package.request_id}")
        
        try:
            scoring_payload = {
                "request_id": embedding_package.request_id,
                "text_report": text_report if text_report else None,
                "visual_report": visual_report if visual_report else None,
                "chromadb_collection_id": embedding_package.chromadb_collection_id or embedding_package.request_id
            }
            
            async with aiohttp.ClientSession() as session:
                ctx.logger.info(f"📡 Making POST request to {scoring_agent_url}...")
                async with session.post(scoring_agent_url, json=scoring_payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    ctx.logger.info(f"📥 Received response from Scoring Agent!")
                    ctx.logger.info(f"   📊 Status code: {response.status}")
                    
                    if response.status == 200:
                        final_scored_report = await response.json()
                        ctx.logger.info(f"✅ SUCCESS - Final scoring complete!")
                        ctx.logger.info(f"   📊 Final Score: {final_scored_report.get('overall_bias_score', 'N/A'):.1f}/10")
                        ctx.logger.info(f"   🏷️  Bias Level: {final_scored_report.get('bias_level', 'N/A')}")
                        ctx.logger.info(f"   📋 Total Issues: {final_scored_report.get('total_issues', 0)}")
                    else:
                        ctx.logger.error(f"   ❌ Scoring Agent returned status: {response.status}")
                        error_text = await response.text()
                        ctx.logger.error(f"   📄 Error response: {error_text[:200]}")
        except Exception as e:
            ctx.logger.error(f"❌ ERROR calling Scoring Agent: {e}")
            import traceback
            ctx.logger.error(traceback.format_exc())
    else:
        ctx.logger.error(f"❌ No reports available to score")
        ctx.logger.error(f"   Text Report: {'Present' if text_report else 'MISSING'}")
        ctx.logger.error(f"   Visual Report: {'Present' if visual_report else 'MISSING'}")
        ctx.logger.error(f"   ⚠️  Skipping Scoring Agent call")

    ctx.logger.info("")
    ctx.logger.info("=" * 80)
    ctx.logger.info("📊 ANALYSIS PIPELINE COMPLETE")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"   ✅ Text Analysis: {'Complete' if text_report else 'Skipped/Failed'}")
    ctx.logger.info(f"   ✅ Visual Analysis: {'Complete' if visual_report else 'Skipped/Failed'}")
    ctx.logger.info(f"   ✅ Final Scoring: {'Complete' if final_scored_report else 'Failed'}")
    ctx.logger.info("=" * 80)
    
    # RETURN the final scored report so main.py can pass it to frontend
    return final_scored_report


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
    Returns immediately and processes asynchronously in the background.
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

        # Process the content and WAIT for completion (including scoring)
        ctx.logger.info(f"🔄 Processing request {req.request_id}...")
        ctx.logger.info(f"   ⚙️  This will:")
        ctx.logger.info(f"      1. Extract content from YouTube")
        ctx.logger.info(f"      2. Preprocess with Claude")
        ctx.logger.info(f"      3. Analyze with Text/Visual Bias Agents")
        ctx.logger.info(f"      4. Generate final score")
        ctx.logger.info(f"   ⏳ Please wait - this may take 30-60 seconds...")
        
        # AWAIT the entire pipeline to complete
        acknowledgement = await process_ad_content(ctx, proper_request)
        
        ctx.logger.info(f"✅ Request {req.request_id} fully processed!")
        ctx.logger.info(f"   📊 Final report is now available at GET /report/{req.request_id}")
        
        return acknowledgement

    except Exception as e:
        ctx.logger.error(f"❌ Error accepting REST request {req.request_id}: {e}")
        return IngestionAcknowledgement(
            request_id=req.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )


# Include protocols
ingestion_agent.include(ingestion_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🚀 INGESTION AGENT - Ad Bias Detection              ║
╚══════════════════════════════════════════════════════════════╝

Role: Content Extraction and Agent Routing

Architecture Flow (REST-based):
  1. Receive ad content via REST API
  2. Extract YouTube content (transcript + Claude-selected frames)
  3. Route to Text Bias Agent (for text analysis) via HTTP POST
  4. Route to Visual Bias Agent (for visual analysis) via HTTP POST
  5. Text Bias Agent triggers Scoring Agent via HTTP POST
  6. Scoring Agent fetches reports from Text/Visual agents via HTTP GET
  7. Frontend retrieves final report from Scoring Agent via HTTP GET

Capabilities:
  ✓ Extracts YouTube transcripts and video frames
  ✓ Claude-powered frame selection (most bias-relevant)
  ✓ Async background processing (no timeouts)
  ✓ Fire-and-forget agent communication
  ✓ 1-minute video duration limit enforcement

Endpoints:
  • REST API: http://localhost:8100/submit
  • Agent Protocol: http://localhost:8100/submit

📍 Waiting for content requests...
🛑 Stop with Ctrl+C
    """)
    ingestion_agent.run()

