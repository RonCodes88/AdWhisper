from fastapi import FastAPI, HTTPException
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
INGESTION_AGENT_REST_ENDPOINT = "http://localhost:8100/analyze"  # Ingestion agent REST endpoint (correct endpoint)


# Removed: call_ingestion_agent_background - no longer needed
# We now call ingestion agent synchronously to wait for complete analysis


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
async def analyze_youtube_video(request: YouTubeAnalysisRequest):
    """
    Frontend calls this endpoint with YouTube URL.
    WAITS for complete analysis before returning.
    Calls Ingestion Agent synchronously (blocking until scoring completes).
    
    Flow:
    Frontend → FastAPI → Ingestion Agent → Text/Visual Agents → Scoring Agent → Response
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
            "video_url": request.youtube_url,
            "metadata": {
                "source": "youtube",
                "youtube_url": request.youtube_url
            }
        }
        
        # Call Ingestion Agent SYNCHRONOUSLY and WAIT for complete pipeline
        if ENABLE_AGENT_CALLS:
            print(f"\n📤 Calling Ingestion Agent (SYNCHRONOUS - will wait for completion)")
            print(f"🎯 Endpoint: {INGESTION_AGENT_REST_ENDPOINT}")
            print(f"⏳ This may take 30-60 seconds for full analysis...")
            
            try:
                response = requests.post(
                    INGESTION_AGENT_REST_ENDPOINT,
                    json=ingestion_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300  # 5 minute timeout for full pipeline
                )
                
                if response.status_code == 200:
                    final_report = response.json()
                    print(f"✅ SUCCESS - Full analysis pipeline completed!")
                    print(f"📊 Final Score: {final_report.get('overall_bias_score', 'N/A')}")
                    print(f"🏷️  Bias Level: {final_report.get('bias_level', 'N/A')}")
                    print(f"📋 Total Issues: {final_report.get('total_issues', 0)}")
                    print(f"💡 Recommendations: {len(final_report.get('recommendations', []))}")
                    agent_contacted = True
                    
                    # Return the complete final report to frontend
                    print(f"✅ Returning complete bias report to frontend...")
                    print("="*70 + "\n")
                    return final_report
                else:
                    print(f"⚠️ WARNING - Ingestion Agent returned HTTP {response.status_code}")
                    print(f"Response: {response.text[:500]}")
                    agent_contacted = False
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ ERROR - Could not connect to Ingestion Agent")
                print(f"   Make sure it's running: python agents/ingestion_agent.py")
                raise HTTPException(status_code=503, detail="Ingestion Agent not available")
            except requests.exceptions.Timeout:
                print(f"⏱️ ERROR - Ingestion Agent timed out (took > 5 minutes)")
                raise HTTPException(status_code=504, detail="Analysis timeout")
        else:
            print(f"\n⏭️  Skipping agent call (ENABLE_AGENT_CALLS = False)")
            agent_contacted = False
        
        # Build response - report is now ready!
        print(f"\n📦 Building response to frontend...")
        response_data = {
            "request_id": request_id,
            "youtube_url": request.youtube_url,
            "status": "completed",
            "message": "Analysis completed - Final report is ready",
            "agent_contacted": agent_contacted
        }
        
        print(f"✅ Response ready - Status: {response_data['status']}")
        print(f"✅ Final report available at: GET /report/{request_id}")
        print(f"✅ Sending response to frontend...")
        print("="*70 + "\n")
        
        return response_data
        
    except HTTPException:
        raise
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
    

