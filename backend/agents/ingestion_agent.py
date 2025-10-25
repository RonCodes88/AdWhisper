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

# Import embedding models
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import requests
from io import BytesIO

# Import ChromaDB
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chroma import ChromaDB

# Import YouTube processor
from youtube_processor import extract_youtube_content

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


# Initialize embedding models (lazy loading)
_text_model = None
_visual_model = None
_chroma_db = None

def get_text_model():
    """Lazy load text embedding model"""
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast and efficient
    return _text_model

def get_visual_model():
    """Lazy load visual embedding model (CLIP)"""
    global _visual_model
    if _visual_model is None:
        _visual_model = SentenceTransformer('clip-ViT-B-32')  # 512-dim CLIP model
    return _visual_model

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
    
    Uses 'all-MiniLM-L6-v2' model which produces 384-dimensional embeddings.
    This model is fast, efficient, and great for semantic similarity.
    """
    ctx.logger.info(f"🧠 Generating text embedding (length: {len(text)} chars)")
    
    try:
        # Load the model (cached after first load)
        model = get_text_model()
        
        # Generate embedding
        embedding = model.encode(text, convert_to_tensor=False)
        
        # Convert to list of floats
        embedding_list = embedding.tolist()
        
        ctx.logger.info(f"✅ Generated text embedding: {len(embedding_list)}-dimensional vector")
        return embedding_list
        
    except Exception as e:
        ctx.logger.error(f"❌ Error generating text embedding: {e}")
        # Fallback to zero vector
        return [0.0] * 384


async def generate_visual_embedding(ctx: Context, media_url: str) -> List[float]:
    """
    Generate visual embeddings using CLIP model.
    
    Uses 'clip-ViT-B-32' model which produces 512-dimensional embeddings.
    This model understands both images and text in the same embedding space.
    """
    ctx.logger.info(f"🧠 Generating visual embedding for: {media_url}")
    
    try:
        # Load the CLIP model (cached after first load)
        model = get_visual_model()
        
        # Download and load the image
        response = requests.get(media_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate embedding
        embedding = model.encode(image, convert_to_tensor=False)
        
        # Convert to list of floats
        embedding_list = embedding.tolist()
        
        ctx.logger.info(f"✅ Generated visual embedding: {len(embedding_list)}-dimensional vector")
        return embedding_list
        
    except Exception as e:
        ctx.logger.error(f"❌ Error generating visual embedding: {e}")
        # Fallback to zero vector
        return [0.0] * 512


async def store_in_chromadb(
    ctx: Context,
    request_id: str,
    text_embedding: Optional[List[float]],
    visual_embedding: Optional[List[float]],
    metadata: Dict[str, Any]
) -> str:
    """
    Store embeddings in ChromaDB for future RAG retrieval.
    
    Stores both text and visual embeddings along with metadata.
    Uses the primary embedding (text or visual) for vector search.
    """
    ctx.logger.info(f"💾 Storing embeddings in ChromaDB for request {request_id}")
    
    try:
        # Get ChromaDB instance
        chroma_db = get_chroma_db()
        collection = chroma_db.collection
        
        # Prepare metadata
        storage_metadata = {
            "request_id": request_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "has_text_embedding": text_embedding is not None,
            "has_visual_embedding": visual_embedding is not None,
        }
        
        # Add custom metadata if provided
        if metadata:
            storage_metadata.update(metadata)
        
        # Determine which embedding to use as primary (for vector search)
        # Prefer text embedding, fallback to visual
        primary_embedding = text_embedding if text_embedding else visual_embedding
        
        if primary_embedding is None:
            ctx.logger.warning(f"⚠️ No embeddings to store for request {request_id}")
            return f"no_embedding_{request_id}"
        
        # Create document content
        document_content = f"Ad content - Request ID: {request_id}"
        if metadata and "description" in metadata:
            document_content = metadata["description"]
        
        # Store in ChromaDB with explicit embedding
        collection.add(
            embeddings=[primary_embedding],
            documents=[document_content],
            metadatas=[storage_metadata],
            ids=[request_id]
        )
        
        ctx.logger.info(f"✅ Stored embeddings in ChromaDB: {request_id}")
        ctx.logger.info(f"   - Text embedding: {text_embedding is not None}")
        ctx.logger.info(f"   - Visual embedding: {visual_embedding is not None}")
        ctx.logger.info(f"   - Embedding dimension: {len(primary_embedding)}")
        
        return request_id
        
    except Exception as e:
        ctx.logger.error(f"❌ Error storing in ChromaDB: {e}")
        raise


async def route_to_analysis_agents(
    ctx: Context,
    embedding_package: EmbeddingPackage,
    original_content: AdContentRequest
):
    """
    Route content to Text and Visual Bias agents for parallel analysis.
    """
    routed_count = 0

    # Route to Text Bias Agent if text content exists
    if embedding_package.text_content and TEXT_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing to Text Bias Agent: {TEXT_BIAS_AGENT_ADDRESS}")
        await ctx.send(TEXT_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1

    # Route to Visual Bias Agent if visual content exists
    if embedding_package.visual_embedding and VISUAL_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing to Visual Bias Agent: {VISUAL_BIAS_AGENT_ADDRESS}")
        await ctx.send(VISUAL_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1

    if routed_count == 0:
        ctx.logger.warning(f"⚠️ No analysis agents configured or no content to analyze.")
        ctx.logger.warning(f"   - Text content: {embedding_package.text_content is not None}")
        ctx.logger.warning(f"   - Visual embedding: {embedding_package.visual_embedding is not None}")
        ctx.logger.warning(f"   - Text agent address: {TEXT_BIAS_AGENT_ADDRESS}")
        ctx.logger.warning(f"   - Visual agent address: {VISUAL_BIAS_AGENT_ADDRESS}")
    else:
        ctx.logger.info(f"✅ Content routed to {routed_count} analysis agent(s)")


# REST endpoint for FastAPI to submit content
@ingestion_agent.on_rest_post("/submit", AdContentRequest, IngestionAcknowledgement)
async def rest_submit_content(ctx: Context, req: AdContentRequest) -> IngestionAcknowledgement:
    """
    REST endpoint for receiving ad content from FastAPI backend.
    This allows FastAPI to submit content via HTTP POST.
    """
    ctx.logger.info(f"📨 REST API: Received content request: {req.request_id}")

    try:
        # Process the content (same as message handler)
        # Store request in context storage
        ctx.storage.set(req.request_id, {
            "content_type": req.content_type.value,
            "received_at": datetime.now(UTC).isoformat(),
            "source": "rest_api"
        })

        # Step 1: Preprocess content
        ctx.logger.info(f"🔄 Preprocessing content for request {req.request_id}...")
        preprocessed_data = await preprocess_content(ctx, req)

        # Step 2: Generate embeddings
        ctx.logger.info(f"🧠 Generating embeddings for request {req.request_id}...")
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
            req.request_id,
            text_embedding,
            visual_embedding,
            preprocessed_data.get("metadata")
        )
        ctx.logger.info(f"✅ Stored in ChromaDB collection: {collection_id}")

        # Step 4: Create embedding package
        embedding_package = EmbeddingPackage(
            request_id=req.request_id,
            text_content=preprocessed_data.get("text"),
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
            chromadb_collection_id=collection_id,
            content_type=req.content_type,
            metadata=preprocessed_data.get("metadata")
        )

        # Step 5: Route to analysis agents
        ctx.logger.info(f"🔀 Routing content to analysis agents...")
        await route_to_analysis_agents(ctx, embedding_package, req)

        # Return acknowledgement
        return IngestionAcknowledgement(
            request_id=req.request_id,
            status="success",
            message=f"Content ingested and routed for analysis. Collection ID: {collection_id}"
        )

    except Exception as e:
        ctx.logger.error(f"❌ Error processing REST request {req.request_id}: {e}")
        return IngestionAcknowledgement(
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

