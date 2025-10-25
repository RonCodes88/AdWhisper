from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from chroma import ChromaDB
import uuid
from typing import Dict, Any
import asyncio

# Import uAgent for agent communication
from uagents import Agent, Context
from agents.shared_models import AdContentRequest, IngestionAcknowledgement

app = FastAPI()

# Lazy load ChromaDB
_db = None
def get_db():
    global _db
    if _db is None:
        print("🔄 Initializing ChromaDB...")
        _db = ChromaDB()
        print("✅ ChromaDB initialized")
    return _db

# Ingestion Agent address (update this after starting the Ingestion Agent)
INGESTION_AGENT_ADDRESS = "agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv"

# Create a simple uAgent for FastAPI to send messages
fastapi_agent = Agent(
    name="fastapi_bridge",
    seed="fastapi_bridge_agent_unique_seed_2024",
    port=8200,
    endpoint=["http://localhost:8200/submit"]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting AdWhisper Backend...")
    print("✅ FastAPI server ready")
    print("📍 Listening on http://localhost:8000")
    print("🔗 CORS enabled for http://localhost:3000")
    print(f"🤖 FastAPI Bridge Agent: {fastapi_agent.address}")
    print(f"📤 Will send messages to Ingestion Agent: {INGESTION_AGENT_ADDRESS}")
    print("")

    # Start the uAgent in the background
    asyncio.create_task(run_agent_background())


async def run_agent_background():
    """Run the uAgent in the background"""
    try:
        # The agent needs to run to be able to send messages
        # We run it in a way that doesn't block FastAPI
        await fastapi_agent._startup()
    except Exception as e:
        print(f"⚠️ Error starting FastAPI agent: {e}")


# Request/Response Models
class YouTubeAnalysisRequest(BaseModel):
    youtube_url: str


def call_ingestion_agent_background(request_id: str, ingestion_payload: Dict[str, Any]):
    """
    Background task to call the Ingestion Agent without blocking the response.
    This runs AFTER the response is sent to the frontend.
    """
    print(f"\n🔄 [BACKGROUND] Calling Ingestion Agent for request {request_id}")
    
    try:
        response = requests.post(
            INGESTION_AGENT_ENDPOINT,
            json=ingestion_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ [BACKGROUND] Ingestion Agent processed request {request_id}")
            try:
                result = response.json()
                print(f"📨 [BACKGROUND] Agent response: {result}")
            except:
                print(f"⚠️ [BACKGROUND] Could not parse agent response")
        else:
            print(f"⚠️ [BACKGROUND] Agent returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"⚠️ [BACKGROUND] Ingestion Agent not running at {INGESTION_AGENT_ENDPOINT}")
    except requests.exceptions.Timeout:
        print(f"⏰ [BACKGROUND] Agent timeout after 30s")
    except Exception as e:
        print(f"❌ [BACKGROUND] Error calling agent: {type(e).__name__}: {str(e)[:100]}")


@app.get("/")
async def root():
    return {"message": "Welcome to AdWhisper API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/documents")
async def get_documents():
    db = get_db()
    documents = db.collection.get()
    return {"documents": documents}


@app.post("/api/analyze-youtube")
async def analyze_youtube_video(request: YouTubeAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Frontend calls this endpoint with YouTube URL.
    Returns immediately with placeholder results.
    Calls Ingestion Agent in background (non-blocking).
    
    Flow:
    Frontend → FastAPI (instant response) → [Background: Ingestion Agent → Text/Visual Agents → Scoring Agent]
    """
    print("\n" + "="*70)
    print("🎬 NEW REQUEST RECEIVED")
    print("="*70)
    
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        print(f"📝 Request ID: {request_id}")
        print(f"🔗 YouTube URL: {request.youtube_url}")
        
        # Create request payload for Ingestion Agent
        ingestion_payload = {
            "request_id": request_id,
            "content_type": "video",
            "text_content": None,  # Will be extracted by ingestion agent
            "image_url": None,
            "video_url": request.youtube_url,
            "metadata": {
                "source": "youtube",
                "youtube_url": request.youtube_url
            },
            "timestamp": ""
        }
        
        # Add background task to call agent AFTER response is sent (if enabled)
        if ENABLE_AGENT_CALLS:
            print(f"\n📋 Adding Ingestion Agent call to background tasks")
            background_tasks.add_task(
                call_ingestion_agent_background,
                request_id,
                ingestion_payload
            )
            agent_contacted = True
            agent_error = None
        else:
            print(f"\n⏭️  Skipping agent call (ENABLE_AGENT_CALLS = False)")
            agent_contacted = False
            agent_error = "Agent calls disabled"
        
        # Build response
        print(f"\n📦 Building response to frontend...")
        response_data = {
            "request_id": request_id,
            "youtube_url": request.youtube_url,
            "status": "processing",
            "message": "Analysis started - Ingestion Agent processing in background",
            "agent_contacted": agent_contacted,
            "bias_score": 7.5,
            "text_bias": {
                "score": 7.0,
                "issues": ["Text bias analysis in progress"],
                "examples": ["Sample example text"]
            },
            "visual_bias": {
                "score": 8.0,
                "issues": ["Visual bias analysis in progress"],
                "examples": ["Sample example visual"]
            },
            "recommendations": [
                "Consider using more inclusive language",
                "Increase diversity in visual representation"
            ]
        }
        
        print(f"✅ Response ready - Status: {response_data['status']}")
        print(f"✅ Sending response to frontend...")
        print("="*70 + "\n")
        
        return response_data
        
    except Exception as e:
        print(f"\n❌ ERROR IN ENDPOINT")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("="*70 + "\n")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🚀 AdWhisper FastAPI Server                     ║
╚══════════════════════════════════════════════════════════════╝

Running on: http://localhost:8000

Features:
  ✅ Instant responses (no blocking!)
  🔄 Background agent processing
  💾 ChromaDB integration
  
Endpoints:
  GET  /              - Welcome message
  GET  /health        - Health check
  GET  /documents     - Get ChromaDB documents
  POST /api/analyze-youtube - YouTube bias analysis (with background agent)

Optional: Start Ingestion Agent for full pipeline:
    ./adwhisper/bin/python agents/ingestion_agent.py

🛑 Stop with Ctrl+C
    """)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    

