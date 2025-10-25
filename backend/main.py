from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from chroma import ChromaDB
import requests
import uuid
from typing import Dict, Any

app = FastAPI()
db = ChromaDB()

# Ingestion Agent endpoint
INGESTION_AGENT_ENDPOINT = "http://localhost:8100/submit"

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    documents = db.collection.get_all()
    return {"documents": documents}


@app.post("/api/analyze-youtube")
async def analyze_youtube_video(request: YouTubeAnalysisRequest):
    """
    Frontend calls this endpoint with YouTube URL.
    This endpoint then calls the Ingestion Agent.
    
    Flow:
    Frontend → FastAPI (main.py) → Ingestion Agent → Text/Visual Agents → Scoring Agent
    """
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        print(f"📝 Received YouTube analysis request: {request_id}")
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
        
        print(f"📤 Sending request to Ingestion Agent at {INGESTION_AGENT_ENDPOINT}")
        
        # Send POST request to Ingestion Agent (optional - works without agent too)
        agent_contacted = False
        try:
            agent_response = requests.post(
                INGESTION_AGENT_ENDPOINT,
                json=ingestion_payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if agent_response.status_code == 200:
                print(f"✅ Ingestion Agent received request")
                agent_result = agent_response.json()
                print(f"📨 Agent response: {agent_result}")
                agent_contacted = True
            else:
                print(f"⚠️ Ingestion Agent returned status: {agent_response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Ingestion Agent not running (optional for now)")
        except Exception as e:
            print(f"⚠️ Could not reach Ingestion Agent: {e}")
        
        # Return response to frontend
        # (Agent will process in background, for now we return immediate placeholder)
        return {
            "request_id": request_id,
            "youtube_url": request.youtube_url,
            "status": "processing" if agent_contacted else "received",
            "message": "Video sent to Ingestion Agent for processing" if agent_contacted else "Processing locally",
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
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🚀 AdWhisper FastAPI Server                     ║
╚══════════════════════════════════════════════════════════════╝

Running on: http://localhost:8000

Endpoints:
  GET  /              - Welcome message
  GET  /health        - Health check
  POST /api/analyze-youtube - YouTube bias analysis

⚠️  Make sure Ingestion Agent is running:
    cd agents && python ingestion_agent.py

🛑 Stop with Ctrl+C
    """)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    

