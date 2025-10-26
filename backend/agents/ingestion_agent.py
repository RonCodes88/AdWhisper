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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import open_clip
import torch
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np

from chroma import ChromaDB


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


# Message Models
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
    """Processed content with embeddings"""
    request_id: str
    text_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None
    chromadb_collection_id: str
    content_type: ContentType
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


# Initialize Ingestion Agent
ingestion_agent = Agent(
    name="ingestion_agent",
    seed="ad_bias_ingestion_agent_unique_seed_2024",
    port=8100,
    endpoint=["http://localhost:8100/submit"],
    mailbox=False  # Local development - direct agent-to-agent communication
)

# Protocol for ingestion
ingestion_protocol = Protocol(name="ingestion_protocol", version="1.0")

# Global models (initialized once on startup)
text_embedding_model = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None
chroma_db = None

# Agent addresses for routing (will be updated with actual addresses)
TEXT_BIAS_AGENT_ADDRESS = os.getenv("TEXT_BIAS_AGENT_ADDRESS", "agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv")
VISUAL_BIAS_AGENT_ADDRESS = os.getenv("VISUAL_BIAS_AGENT_ADDRESS", "agent1qtnatq0rhrj2pauyg2a8dgf56uqkf6tw3757z806w6c57zkw9nry2my2933")


@ingestion_agent.on_event("startup")
async def startup(ctx: Context):
    global text_embedding_model, clip_model, clip_preprocess, clip_tokenizer, chroma_db
    
    ctx.logger.info(f"🚀 Ingestion Agent starting up...")
    ctx.logger.info(f"📍 Agent address: {ingestion_agent.address}")
    
    # Initialize ChromaDB
    ctx.logger.info("💾 Initializing ChromaDB...")
    chroma_db = ChromaDB()
    ctx.logger.info(f"✅ ChromaDB initialized with {len(chroma_db._collections)} collections")
    
    # Initialize text embedding model
    ctx.logger.info("🧠 Loading text embedding model (sentence-transformers)...")
    try:
        text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        ctx.logger.info("✅ Text embedding model loaded (384-dim)")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to load text embedding model: {e}")
    
    # Initialize CLIP model for visual embeddings
    ctx.logger.info("👁️ Loading CLIP model for visual embeddings...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(device)
        clip_model.eval()
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        ctx.logger.info(f"✅ CLIP model loaded (512-dim) on {device}")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to load CLIP model: {e}")
    
    ctx.logger.info(f"🔧 Role: Data Reception, Preprocessing, and Embedding Generation")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8100/submit")
    ctx.logger.info(f"⚡ Ready to receive ad content for bias analysis")


@ingestion_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Ingestion Agent shutting down...")
    ctx.logger.info("🧹 Cleaning up resources...")


async def process_ad_content(ctx: Context, msg: AdContentRequest) -> IngestionAcknowledgement:
    """
    Shared processing logic for ad content (used by both protocol and REST)
    """
    try:
        ctx.logger.info(f"📨 Received ad content request: {msg.request_id}")
        ctx.logger.info(f"📊 Content type: {msg.content_type}")
        
        # Store request in context storage
        ctx.storage.set(msg.request_id, {
            "content_type": msg.content_type.value,
            "received_at": datetime.now(UTC).isoformat()
        })
        
        # Step 1: Preprocess content
        ctx.logger.info(f"🔄 Preprocessing content for request {msg.request_id}...")
        preprocessed_data = await preprocess_content(ctx, msg)

        # Step 2: Generate embeddings
        ctx.logger.info(f"🧠 Generating embeddings for request {msg.request_id}...")
        text_embedding = None
        visual_embedding = None

        # Use preprocessed text (which may include YouTube transcript)
        if preprocessed_data.get("text"):
            text_embedding = await generate_text_embedding(ctx, preprocessed_data["text"])
            ctx.logger.info(f"✅ Text embedding generated (dim: {len(text_embedding) if text_embedding else 0})")

        # Use preprocessed image URL (which may include YouTube thumbnail)
        if preprocessed_data.get("image_url"):
            visual_embedding = await generate_visual_embedding(ctx, preprocessed_data["image_url"])
            ctx.logger.info(f"✅ Visual embedding generated (dim: {len(visual_embedding) if visual_embedding else 0})")
        
        # Step 3: Store in ChromaDB
        ctx.logger.info(f"💾 Storing embeddings in ChromaDB...")
        collection_id = await store_in_chromadb(
            ctx,
            msg.request_id,
            text_embedding,
            visual_embedding,
            preprocessed_data.get("metadata")
        )
        ctx.logger.info(f"✅ Stored in ChromaDB collection: {collection_id}")

        # Step 4: Create embedding package
        embedding_package = EmbeddingPackage(
            request_id=msg.request_id,
            text_content=preprocessed_data.get("text"),  # Include original text for analysis
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
            chromadb_collection_id=collection_id,
            content_type=msg.content_type,
            metadata=preprocessed_data.get("metadata")
        )
        
        # Step 5: Route to analysis agents
        ctx.logger.info(f"🔀 Routing content to analysis agents...")
        await route_to_analysis_agents(ctx, embedding_package, msg)
        
        ctx.logger.info(f"✅ Request {msg.request_id} processed successfully")
        
        return IngestionAcknowledgement(
            request_id=msg.request_id,
            status="success",
            message=f"Content ingested and routed for analysis. Collection ID: {collection_id}"
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


@ingestion_agent.on_rest_post("/analyze", AdContentRequest, IngestionAcknowledgement)
async def handle_rest_analyze(ctx: Context, req: AdContentRequest) -> IngestionAcknowledgement:
    """
    REST endpoint for FastAPI to send content directly
    Usage: POST http://localhost:8100/analyze
    """
    ctx.logger.info(f"🌐 REST request received: {req.request_id}")
    return await process_ad_content(ctx, req)


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
                    ctx.logger.info(f"   - Transcript: {len(transcript_text)} characters")
                else:
                    ctx.logger.warning(f"   - No transcript available: {youtube_data['transcript']['error']}")

                # Use thumbnail as image
                preprocessed["image_url"] = youtube_data["thumbnail_url"]
                ctx.logger.info(f"   - Thumbnail URL: {youtube_data['thumbnail_url']}")

                # Add YouTube metadata
                preprocessed["metadata"]["youtube"] = {
                    "video_id": youtube_data["video_id"],
                    "title": youtube_data["metadata"].get("title"),
                    "channel": youtube_data["metadata"].get("channel"),
                    "duration": youtube_data["metadata"].get("duration"),
                    "views": youtube_data["metadata"].get("views"),
                    "has_transcript": youtube_data["transcript"]["success"],
                    "transcript_language": youtube_data["transcript"].get("language"),
                    "is_auto_generated": youtube_data["transcript"].get("is_generated")
                }

            else:
                ctx.logger.error(f"❌ Failed to extract YouTube content: {youtube_data['error']}")
                preprocessed["metadata"]["youtube_error"] = youtube_data["error"]

        except Exception as e:
            ctx.logger.error(f"❌ Error processing YouTube video: {e}")
            preprocessed["metadata"]["youtube_error"] = str(e)

    ctx.logger.info(f"✅ Content preprocessed for request {content.request_id}")
    return preprocessed


async def generate_text_embedding(ctx: Context, text: str) -> List[float]:
    """
    Generate text embeddings using sentence transformers.
    Uses all-MiniLM-L6-v2 model (384-dim embeddings)
    """
    global text_embedding_model
    
    if text_embedding_model is None:
        ctx.logger.error("❌ Text embedding model not initialized")
        return [0.0] * 384  # Return zero vector as fallback
    
    try:
        ctx.logger.info(f"🧠 Generating text embedding (length: {len(text)} chars)")
        
        # Generate embedding
        embedding = text_embedding_model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()
        
        ctx.logger.info(f"✅ Text embedding generated ({len(embedding_list)}-dim)")
        return embedding_list
        
    except Exception as e:
        ctx.logger.error(f"❌ Error generating text embedding: {e}")
        return [0.0] * 384  # Return zero vector as fallback


async def generate_visual_embedding(ctx: Context, media_url: str) -> List[float]:
    """
    Generate visual embeddings using CLIP.
    Uses ViT-B-32 model (512-dim embeddings)
    Handles both images and videos (extracts first frame for videos)
    """
    global clip_model, clip_preprocess
    
    if clip_model is None or clip_preprocess is None:
        ctx.logger.error("❌ CLIP model not initialized")
        return [0.0] * 512  # Return zero vector as fallback
    
    try:
        ctx.logger.info(f"🧠 Generating visual embedding for: {media_url}")
        
        # Determine if it's a video or image
        is_video = media_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        
        if is_video:
            # Extract first frame from video
            ctx.logger.info("🎬 Detected video, extracting first frame...")
            image = await extract_first_frame_from_video(ctx, media_url)
        else:
            # Load image directly
            if media_url.startswith('http://') or media_url.startswith('https://'):
                response = requests.get(media_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(media_url).convert('RGB')
        
        # Preprocess and generate embedding
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
        
        embedding_list = image_features.cpu().numpy()[0].tolist()
        
        ctx.logger.info(f"✅ Visual embedding generated ({len(embedding_list)}-dim)")
        return embedding_list
        
    except Exception as e:
        ctx.logger.error(f"❌ Error generating visual embedding: {e}")
        return [0.0] * 512  # Return zero vector as fallback


async def extract_first_frame_from_video(ctx: Context, video_url: str) -> Image.Image:
    """
    Extract the first frame from a video file.
    """
    try:
        # Download video if URL
        if video_url.startswith('http://') or video_url.startswith('https://'):
            response = requests.get(video_url, timeout=30)
            video_path = f"/tmp/temp_video_{hash(video_url)}.mp4"
            with open(video_path, 'wb') as f:
                f.write(response.content)
        else:
            video_path = video_url
        
        # Extract first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read video frame")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        return image
        
    except Exception as e:
        ctx.logger.error(f"❌ Error extracting video frame: {e}")
        # Return blank image as fallback
        return Image.new('RGB', (224, 224), color='black')


async def store_in_chromadb(
    ctx: Context,
    request_id: str,
    text_embedding: Optional[List[float]],
    visual_embedding: Optional[List[float]],
    metadata: Dict[str, Any]
) -> str:
    """
    Store embeddings in ChromaDB for future RAG retrieval.
    Stores in the ad_content collection.
    """
    global chroma_db
    
    if chroma_db is None:
        ctx.logger.error("❌ ChromaDB not initialized")
        return f"error_{request_id}"
    
    try:
        ctx.logger.info(f"💾 Storing embeddings in ChromaDB for request {request_id}")
        
        # Store in ad_content collection
        collection_id = chroma_db.store_ad_content(
            ad_id=request_id,
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
            metadata=metadata
        )
        
        ctx.logger.info(f"✅ Stored in collection: {collection_id}")
        return collection_id
        
    except Exception as e:
        ctx.logger.error(f"❌ Error storing in ChromaDB: {e}")
        return f"error_{request_id}"


async def route_to_analysis_agents(
    ctx: Context,
    embedding_package: EmbeddingPackage,
    original_content: AdContentRequest
):
    """
    Route content to Text and Visual Bias agents for parallel analysis.
    Creates proper request messages for each agent.
    """
    routed_count = 0

    # Route to Text Bias Agent if text content exists
    if original_content.text_content and TEXT_BIAS_AGENT_ADDRESS:
        try:
            ctx.logger.info(f"📤 Routing to Text Bias Agent: {TEXT_BIAS_AGENT_ADDRESS}")
            
            # Create TextAnalysisRequest message
            from agents.text_bias_agent import TextAnalysisRequest
            text_request = TextAnalysisRequest(
                request_id=original_content.request_id,
                text_content=original_content.text_content,
                text_embedding=embedding_package.text_embedding,
                chromadb_collection_id=embedding_package.chromadb_collection_id,
                metadata=original_content.metadata
            )
            
            await ctx.send(TEXT_BIAS_AGENT_ADDRESS, text_request)
            routed_count += 1
            ctx.logger.info(f"✅ Text analysis request sent")
            
        except Exception as e:
            ctx.logger.error(f"❌ Error routing to Text Bias Agent: {e}")
    
    # Route to Visual Bias Agent if visual content exists
    if (original_content.image_url or original_content.video_url):
        try:
            ctx.logger.info("=" * 80)
            ctx.logger.info("🌐 SENDING HTTP REQUEST TO VISUAL BIAS AGENT")
            ctx.logger.info("=" * 80)
            
            # Use direct HTTP communication instead of uAgent messaging
            visual_bias_url = "http://localhost:8102/api/analyze"
            ctx.logger.info(f"📤 Target URL: {visual_bias_url}")
            
            # Prepare payload
            payload = {
                "request_id": original_content.request_id,
                "image_url": original_content.image_url,
                "video_url": original_content.video_url,
                "visual_embedding": embedding_package.visual_embedding,
                "chromadb_collection_id": embedding_package.chromadb_collection_id,
                "metadata": original_content.metadata
            }
            
            ctx.logger.info(f"📦 Request ID: {payload['request_id']}")
            ctx.logger.info(f"📦 Image URL: {payload['image_url'][:80] if payload['image_url'] else 'None'}...")
            ctx.logger.info(f"📦 Embedding dimension: {len(payload['visual_embedding']) if payload['visual_embedding'] else 'None'}")
            
            # Make HTTP POST request
            ctx.logger.info(f"📤 Making HTTP POST request...")
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(visual_bias_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
            routed_count += 1
            ctx.logger.info(f"✅ Visual Bias Agent responded: {result.get('status')}")
            ctx.logger.info(f"✅ Message: {result.get('message')}")
            ctx.logger.info("=" * 80)
            
        except httpx.TimeoutException:
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"❌ TIMEOUT - Visual Bias Agent did not respond in time")
            ctx.logger.error("=" * 80)
        except httpx.ConnectError as e:
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"❌ CONNECTION ERROR - Could not connect to Visual Bias Agent")
            ctx.logger.error(f"Is the Visual Bias Agent running on port 8102?")
            ctx.logger.error(f"Error: {e}")
            ctx.logger.error("=" * 80)
        except Exception as e:
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"❌ ERROR ROUTING TO VISUAL BIAS AGENT")
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"Error type: {type(e).__name__}")
            ctx.logger.error(f"Error message: {e}")
            import traceback
            ctx.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            ctx.logger.error("=" * 80)
    
    # Route to Text Bias Agent if text content exists
    if original_content.text_content:
        try:
            ctx.logger.info("=" * 80)
            ctx.logger.info("🌐 SENDING HTTP REQUEST TO TEXT BIAS AGENT")
            ctx.logger.info("=" * 80)
            
            text_bias_url = "http://localhost:8101/api/analyze"
            ctx.logger.info(f"📤 Target URL: {text_bias_url}")
            
            payload = {
                "request_id": original_content.request_id,
                "text_content": original_content.text_content,
                "text_embedding": embedding_package.text_embedding,
                "chromadb_collection_id": embedding_package.chromadb_collection_id,
                "metadata": original_content.metadata
            }
            
            ctx.logger.info(f"📦 Request ID: {payload['request_id']}")
            ctx.logger.info(f"📦 Text length: {len(payload['text_content'])} chars")
            ctx.logger.info(f"📤 Making HTTP POST request...")
            
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(text_bias_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
            routed_count += 1
            ctx.logger.info(f"✅ Text Bias Agent responded: {result.get('status')}")
            ctx.logger.info("=" * 80)
            
        except httpx.TimeoutException:
            ctx.logger.error("=" * 80)
            ctx.logger.error("❌ TIMEOUT - Text Bias Agent did not respond")
            ctx.logger.error("=" * 80)
        except httpx.ConnectError as e:
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"❌ CONNECTION ERROR - Could not connect to Text Bias Agent on port 8101")
            ctx.logger.error(f"Error: {e}")
            ctx.logger.error("=" * 80)
        except Exception as e:
            ctx.logger.error("=" * 80)
            ctx.logger.error(f"❌ ERROR ROUTING TO TEXT BIAS AGENT")
            ctx.logger.error(f"Error: {e}")
            import traceback
            ctx.logger.error(traceback.format_exc())
            ctx.logger.error("=" * 80)
    
    if routed_count == 0:
        ctx.logger.warning(f"⚠️ No analysis agents configured or no content to route.")
    else:
        ctx.logger.info(f"✅ Content routed to {routed_count} analysis agent(s)")


# ==================== REST API Endpoint ====================

class RESTAdContentRequest(Model):
    """REST API version of AdContentRequest"""
    request_id: str
    content_type: str
    text_content: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RESTAcknowledgement(Model):
    """REST API acknowledgement response"""
    request_id: str
    status: str
    message: str
    timestamp: str = ""
    
    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


@ingestion_agent.on_rest_post("/api/analyze", RESTAdContentRequest, RESTAcknowledgement)
async def handle_rest_ad_content(ctx: Context, req: RESTAdContentRequest) -> RESTAcknowledgement:
    """
    REST endpoint for submitting ad content for analysis.
    Accepts HTTP POST requests from FastAPI backend.
    """
    try:
        ctx.logger.info(f"📨 REST API: Received ad content request: {req.request_id}")
        ctx.logger.info(f"📊 Content type: {req.content_type}")
        
        # Convert REST request to internal AdContentRequest
        content_type_enum = ContentType(req.content_type.lower())
        
        ad_request = AdContentRequest(
            request_id=req.request_id,
            content_type=content_type_enum,
            text_content=req.text_content,
            image_url=req.image_url,
            video_url=req.video_url,
            metadata=req.metadata or {}
        )
        
        # Store request in context storage
        ctx.storage.set(req.request_id, {
            "content_type": req.content_type,
            "received_at": datetime.now(UTC).isoformat(),
            "source": "REST_API"
        })
        
        # Step 1: Preprocess content
        ctx.logger.info(f"🔄 Preprocessing content for request {req.request_id}...")
        preprocessed_data = await preprocess_content(ctx, ad_request)
        
        # Step 2: Generate embeddings
        ctx.logger.info(f"🧠 Generating embeddings for request {req.request_id}...")
        text_embedding = None
        visual_embedding = None
        
        if ad_request.text_content:
            text_embedding = await generate_text_embedding(ctx, ad_request.text_content)
            ctx.logger.info(f"✅ Text embedding generated (dim: {len(text_embedding) if text_embedding else 0})")
        
        if ad_request.image_url or ad_request.video_url:
            visual_embedding = await generate_visual_embedding(ctx, ad_request.image_url or ad_request.video_url)
            ctx.logger.info(f"✅ Visual embedding generated (dim: {len(visual_embedding) if visual_embedding else 0})")
        
        # Step 3: Store in ChromaDB
        ctx.logger.info(f"💾 Storing embeddings in ChromaDB...")
        collection_id = await store_in_chromadb(ctx, req.request_id, text_embedding, visual_embedding, ad_request.metadata)
        ctx.logger.info(f"✅ Stored in ChromaDB collection: {collection_id}")
        
        # Step 4: Create embedding package
        embedding_package = EmbeddingPackage(
            request_id=req.request_id,
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
            chromadb_collection_id=collection_id,
            content_type=content_type_enum
        )
        
        # Step 5: Route to analysis agents
        ctx.logger.info(f"🔀 Routing content to analysis agents...")
        await route_to_analysis_agents(ctx, embedding_package, ad_request)
        
        ctx.logger.info(f"✅ REST API: Request {req.request_id} processed successfully")
        
        return RESTAcknowledgement(
            request_id=req.request_id,
            status="success",
            message=f"Content ingested and routed for analysis. Collection ID: {collection_id}"
        )
        
    except Exception as e:
        ctx.logger.error(f"❌ REST API: Error processing request {req.request_id}: {e}")
        return RESTAcknowledgement(
            request_id=req.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )


# Include protocol
ingestion_agent.include(ingestion_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🚀 INGESTION AGENT - Ad Bias Detection              ║
╚══════════════════════════════════════════════════════════════╝

Role: Data Reception, Preprocessing, and Embedding Generation

Capabilities:
  ✓ Receives ad content (text, images, videos)
  ✓ Generates embeddings (text + visual)
  ✓ Stores in ChromaDB for RAG retrieval
  ✓ Routes to specialized analysis agents

Endpoints:
  • Agent Protocol: http://localhost:8100/submit
  • REST API: http://localhost:8100/analyze

📍 Waiting for requests...
🛑 Stop with Ctrl+C
    """)
    ingestion_agent.run()

