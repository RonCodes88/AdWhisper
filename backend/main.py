from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from typing import Dict, Any
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Note: No uAgents imports needed here - we just make HTTP requests!
# ChromaDB is managed by agents, not by FastAPI

app = FastAPI()

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
    print(f"📤 Will send HTTP requests to Ingestion Agent: {INGESTION_AGENT_REST_ENDPOINT}")
    print("")


# Configuration
ENABLE_AGENT_CALLS = True  # Set to False to disable agent communication
INGESTION_AGENT_REST_ENDPOINT = "http://localhost:8100/analyze"  # Ingestion agent REST endpoint


def call_ingestion_agent_background(request_id: str, ingestion_payload: Dict[str, Any]):
    """
    Background task to call Ingestion Agent via REST
    This runs AFTER the response is sent to the frontend (non-blocking)
    """
    print(f"\n{'='*70}")
    print(f"📤 CALLING INGESTION AGENT (Background Task)")
    print(f"{'='*70}")
    print(f"📝 Request ID: {request_id}")
    print(f"🎯 Endpoint: {INGESTION_AGENT_REST_ENDPOINT}")
    
    # Convert the payload to match the expected schema exactly
    try:
        # Format payload for YouTube ingestion agent
        formatted_payload = {
            "request_id": ingestion_payload["request_id"],
            "content_type": ingestion_payload.get("content_type", "video"),
            "video_url": ingestion_payload.get("video_url"),
            "metadata": ingestion_payload.get("metadata", {})
        }
        
        print(f"📦 Payload: {formatted_payload}")
        
        response = requests.post(
            INGESTION_AGENT_REST_ENDPOINT,
            json=formatted_payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Increased timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS - Ingestion Agent responded!")
            print(f"📨 Status: {result.get('status', 'unknown')}")
            print(f"💬 Message: {result.get('message', 'No message')}")
            print(f"🆔 Request ID: {result.get('request_id', request_id)}")
            print(f"📊 Agent is now processing YouTube video and routing to bias analysis agents...")
        else:
            print(f"⚠️ WARNING - YouTube Agent returned HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR - Could not connect to Ingestion Agent")
        print(f"   Make sure it's running: python agents/ingestion_agent.py")
    except requests.exceptions.Timeout:
        print(f"⏱️ ERROR - Ingestion Agent timed out (took > 30s)")
    except Exception as e:
        print(f"❌ ERROR - Unexpected error: {type(e).__name__}: {str(e)}")
    finally:
        print(f"{'='*70}\n")


# Request/Response Models
class YouTubeAnalysisRequest(BaseModel):
    youtube_url: str


@app.get("/")
async def root():
    return {"message": "Welcome to AdWhisper API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/documents")
async def get_documents():
    """ChromaDB is managed by the Ingestion Agent"""
    return {"message": "ChromaDB operations are handled by the Ingestion Agent", "status": "not_available_here"}


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
            "content_type": "video",  # This will be converted to ContentType enum by the agent
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
        else:
            print(f"\n⏭️  Skipping agent call (ENABLE_AGENT_CALLS = False)")
            agent_contacted = False
        
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
    

