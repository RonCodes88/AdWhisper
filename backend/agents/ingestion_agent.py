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
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for ingestion
ingestion_protocol = Protocol(name="ingestion_protocol", version="1.0")


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


@ingestion_protocol.on_message(model=AdContentRequest, replies=IngestionAcknowledgement)
async def handle_ad_content(ctx: Context, sender: str, msg: AdContentRequest):
    """
    Handle incoming ad content, generate embeddings, and route to analysis agents.
    """
    try:
        ctx.logger.info(f"📨 Received ad content request: {msg.request_id}")
        ctx.logger.info(f"📊 Content type: {msg.content_type}")
        
        # Store request in context storage
        ctx.storage.set(msg.request_id, {
            "sender": sender,
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
        
        if msg.text_content:
            text_embedding = await generate_text_embedding(ctx, msg.text_content)
            ctx.logger.info(f"✅ Text embedding generated (dim: {len(text_embedding) if text_embedding else 0})")
        
        if msg.image_url or msg.video_url:
            visual_embedding = await generate_visual_embedding(ctx, msg.image_url or msg.video_url)
            ctx.logger.info(f"✅ Visual embedding generated (dim: {len(visual_embedding) if visual_embedding else 0})")
        
        # Step 3: Store in ChromaDB
        ctx.logger.info(f"💾 Storing embeddings in ChromaDB...")
        collection_id = await store_in_chromadb(ctx, msg.request_id, text_embedding, visual_embedding, msg.metadata)
        ctx.logger.info(f"✅ Stored in ChromaDB collection: {collection_id}")
        
        # Step 4: Create embedding package
        embedding_package = EmbeddingPackage(
            request_id=msg.request_id,
            text_embedding=text_embedding,
            visual_embedding=visual_embedding,
            chromadb_collection_id=collection_id,
            content_type=msg.content_type
        )
        
        # Step 5: Route to analysis agents
        ctx.logger.info(f"🔀 Routing content to analysis agents...")
        await route_to_analysis_agents(ctx, embedding_package, msg)
        
        # Send acknowledgement back to sender
        acknowledgement = IngestionAcknowledgement(
            request_id=msg.request_id,
            status="success",
            message=f"Content ingested and routed for analysis. Collection ID: {collection_id}"
        )
        
        await ctx.send(sender, acknowledgement)
        ctx.logger.info(f"✅ Request {msg.request_id} processed successfully")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error processing request {msg.request_id}: {e}")
        error_ack = IngestionAcknowledgement(
            request_id=msg.request_id,
            status="error",
            message=f"Error during ingestion: {str(e)}"
        )
        await ctx.send(sender, error_ack)


async def preprocess_content(ctx: Context, content: AdContentRequest) -> Dict[str, Any]:
    """
    Preprocess and clean incoming content.
    """
    preprocessed = {
        "text": content.text_content.strip() if content.text_content else None,
        "image_url": content.image_url,
        "video_url": content.video_url,
        "metadata": content.metadata
    }
    
    ctx.logger.info(f"✅ Content preprocessed for request {content.request_id}")
    return preprocessed


async def generate_text_embedding(ctx: Context, text: str) -> List[float]:
    """
    Generate text embeddings using sentence transformers or OpenAI.
    
    TODO: Implement actual embedding generation
    - Use sentence-transformers (all-mpnet-base-v2) or
    - OpenAI text-embedding-3-large
    """
    # Placeholder: Return dummy embedding
    # In production, use: model.encode(text)
    ctx.logger.info(f"🧠 Generating text embedding (length: {len(text)} chars)")
    
    # Simulated embedding (replace with actual model)
    dummy_embedding = [0.1] * 384  # Example: 384-dim vector
    
    return dummy_embedding


async def generate_visual_embedding(ctx: Context, media_url: str) -> List[float]:
    """
    Generate visual embeddings using CLIP or ViT.
    
    TODO: Implement actual visual embedding generation
    - Use OpenAI CLIP or Google ViT-L/14
    """
    ctx.logger.info(f"🧠 Generating visual embedding for: {media_url}")
    
    # Simulated embedding (replace with actual model)
    dummy_embedding = [0.1] * 512  # Example: 512-dim vector (CLIP)
    
    return dummy_embedding


async def store_in_chromadb(
    ctx: Context,
    request_id: str,
    text_embedding: Optional[List[float]],
    visual_embedding: Optional[List[float]],
    metadata: Dict[str, Any]
) -> str:
    """
    Store embeddings in ChromaDB for future RAG retrieval.
    
    TODO: Implement actual ChromaDB storage
    - Connect to ChromaDB instance
    - Store in appropriate collections
    """
    ctx.logger.info(f"💾 Storing embeddings in ChromaDB for request {request_id}")
    
    # Placeholder: Return dummy collection ID
    collection_id = f"collection_{request_id}"
    
    # In production:
    # - chroma_client.get_or_create_collection("ad_content")
    # - collection.add(embeddings=..., metadatas=..., ids=...)
    
    ctx.logger.info(f"✅ Stored in collection: {collection_id}")
    return collection_id


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
    if original_content.text_content and TEXT_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing to Text Bias Agent: {TEXT_BIAS_AGENT_ADDRESS}")
        # await ctx.send(TEXT_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1
    
    # Route to Visual Bias Agent if visual content exists
    if (original_content.image_url or original_content.video_url) and VISUAL_BIAS_AGENT_ADDRESS:
        ctx.logger.info(f"📤 Routing to Visual Bias Agent: {VISUAL_BIAS_AGENT_ADDRESS}")
        # await ctx.send(VISUAL_BIAS_AGENT_ADDRESS, embedding_package)
        routed_count += 1
    
    if routed_count == 0:
        ctx.logger.warning(f"⚠️ No analysis agents configured. Set agent addresses.")
    else:
        ctx.logger.info(f"✅ Content routed to {routed_count} analysis agent(s)")


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

📍 Waiting for requests...
🛑 Stop with Ctrl+C
    """)
    ingestion_agent.run()

