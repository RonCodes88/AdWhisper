"""
AdWhisper Backend API - FastAPI Server

Provides REST API endpoints for ad bias analysis.
Coordinates with uAgents for distributed bias detection.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import uuid
import os
import json
import logging
import httpx
from datetime import datetime, UTC

from chroma import ChromaDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AdWhisper API",
    description="AI-powered advertising bias detection system",
    version="1.0.0"
)

# Initialize ChromaDB
db = ChromaDB()

# In-memory storage for request tracking
# In production, use Redis or a proper database
analysis_requests = {}
analysis_results = {}

# Agent status tracking
agent_status = {}  # {request_id: {agent_name: status}}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Models ====================

class AdSubmissionResponse(BaseModel):
    request_id: str
    message: str
    status: str
    timestamp: str


class BiasReportResponse(BaseModel):
    request_id: str
    status: str
    overall_score: Optional[float] = None
    assessment: Optional[str] = None
    bias_issues: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[int] = None
    timestamp: str


class AgentStatus(BaseModel):
    name: str
    status: str  # "pending", "processing", "complete", "error"
    message: Optional[str] = None
    timestamp: Optional[str] = None

class AnalysisStatus(BaseModel):
    request_id: str
    status: str
    current_stage: str
    message: str
    timestamp: str
    agents: Optional[Dict[str, AgentStatus]] = None


# ==================== Root Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "Welcome to AdWhisper API",
        "version": "1.0.0",
        "description": "AI-powered advertising bias detection",
        "endpoints": {
            "health": "/health",
            "submit_ad": "/api/analyze-ad",
            "get_results": "/api/results/{request_id}",
            "check_status": "/api/status/{request_id}",
            "collections": "/api/collections"
        }
    }



@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check ChromaDB connection
        collection_count = len(db._collections)
        
        # Check collection counts
        text_patterns = db.get_collection_count(ChromaDB.COLLECTION_TEXT_PATTERNS)
        visual_patterns = db.get_collection_count(ChromaDB.COLLECTION_VISUAL_PATTERNS)
        case_studies = db.get_collection_count(ChromaDB.COLLECTION_CASE_STUDIES)
        
        return {
            "status": "healthy",
            "database": "connected",
            "collections": collection_count,
            "data": {
                "text_patterns": text_patterns,
                "visual_patterns": visual_patterns,
                "case_studies": case_studies
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }


# ==================== Ad Analysis Endpoints ====================

@app.post("/api/analyze-ad", response_model=AdSubmissionResponse)
async def analyze_ad(
    text_content: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    video_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form("{}")
):
    """
    Submit an ad for bias analysis.
    
    Accepts:
    - text_content: Ad text/copy
    - image_url: URL to ad image
    - video_url: URL to ad video
    - image_file: Upload ad image file
    - video_file: Upload ad video file
    - metadata: JSON string with additional context
    
    Returns:
    - request_id: Unique ID to track analysis
    - status: Request status
    """
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Validate that at least one content type is provided
        if not text_content and not image_url and not video_url and not image_file and not video_file:
            raise HTTPException(
                status_code=400,
                detail="At least one content type (text, image, or video) must be provided"
            )
        
        # Handle file uploads
        if image_file:
            # Save uploaded image
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, f"{request_id}_{image_file.filename}")
            with open(image_path, "wb") as f:
                f.write(await image_file.read())
            image_url = image_path
        
        if video_file:
            # Save uploaded video
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            video_path = os.path.join(upload_dir, f"{request_id}_{video_file.filename}")
            with open(video_path, "wb") as f:
                f.write(await video_file.read())
            video_url = video_path
        
        # Determine content type
        content_type = "mixed"
        if text_content and (image_url or video_url):
            content_type = "mixed"
        elif text_content:
            content_type = "text"
        elif image_url:
            content_type = "image"
        elif video_url:
            content_type = "video"
        
        # Store request for tracking
        analysis_requests[request_id] = {
            "request_id": request_id,
            "text_content": text_content,
            "image_url": image_url,
            "video_url": video_url,
            "metadata": metadata,
            "content_type": content_type,
            "status": "processing",
            "current_stage": "submitted",
            "submitted_at": datetime.now(UTC).isoformat()
        }
        
        # Initialize agent status tracking
        agent_status[request_id] = {
            "ingestion_agent": {
                "name": "Ingestion Agent",
                "status": "pending",
                "message": "Waiting to process...",
                "timestamp": datetime.now(UTC).isoformat()
            },
            "text_bias_agent": {
                "name": "Text Bias Agent",
                "status": "pending",
                "message": "Waiting for ingestion...",
                "timestamp": datetime.now(UTC).isoformat()
            },
            "visual_bias_agent": {
                "name": "Visual Bias Agent",
                "status": "pending",
                "message": "Waiting for ingestion...",
                "timestamp": datetime.now(UTC).isoformat()
            },
            "scoring_agent": {
                "name": "Scoring Agent",
                "status": "pending",
                "message": "Waiting for analysis...",
                "timestamp": datetime.now(UTC).isoformat()
            }
        }
        
        # 🚀 Send to Ingestion Agent
        try:
            agent_payload = {
                "request_id": request_id,
                "content_type": content_type,
                "text_content": text_content,
                "image_url": image_url,
                "video_url": video_url,
                "metadata": json.loads(metadata) if metadata else {},
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            logger.info(f"📤 Sending request {request_id} to Ingestion Agent...")
            logger.info(f"   Content type: {content_type}")
            logger.info(f"   Text: {'Yes' if text_content else 'No'}")
            logger.info(f"   Image: {'Yes' if image_url else 'No'}")
            logger.info(f"   Video: {'Yes' if video_url else 'No'}")
            
            # Update agent status
            analysis_requests[request_id]["current_stage"] = "ingestion"
            agent_status[request_id]["ingestion_agent"]["status"] = "processing"
            agent_status[request_id]["ingestion_agent"]["message"] = "Processing and generating embeddings..."
            agent_status[request_id]["ingestion_agent"]["timestamp"] = datetime.now(UTC).isoformat()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:8100/api/analyze",
                    json=agent_payload
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Ingestion Agent accepted request {request_id}")
                    agent_status[request_id]["ingestion_agent"]["status"] = "complete"
                    agent_status[request_id]["ingestion_agent"]["message"] = "Embeddings generated, routing to analysis agents"
                    agent_status[request_id]["ingestion_agent"]["timestamp"] = datetime.now(UTC).isoformat()
                else:
                    logger.warning(f"⚠️ Ingestion Agent returned status {response.status_code}: {response.text}")
                    agent_status[request_id]["ingestion_agent"]["status"] = "error"
                    agent_status[request_id]["ingestion_agent"]["message"] = f"HTTP {response.status_code}"
                    
        except httpx.TimeoutException:
            logger.error(f"❌ Timeout connecting to Ingestion Agent for request {request_id}")
            analysis_requests[request_id]["status"] = "error"
            analysis_requests[request_id]["message"] = "Timeout connecting to analysis agents"
        except httpx.ConnectError:
            logger.error(f"❌ Failed to connect to Ingestion Agent - is it running on port 8100?")
            analysis_requests[request_id]["status"] = "error"
            analysis_requests[request_id]["message"] = "Analysis agents not available"
        except Exception as e:
            logger.error(f"❌ Error sending to Ingestion Agent: {e}")
            analysis_requests[request_id]["status"] = "error"
            analysis_requests[request_id]["message"] = f"Error: {str(e)}"
        
        return AdSubmissionResponse(
            request_id=request_id,
            message="Ad submitted successfully for bias analysis",
            status="processing",
            timestamp=datetime.now(UTC).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{request_id}", response_model=BiasReportResponse)
async def get_results(request_id: str):
    """
    Get analysis results for a submitted ad.
    
    Returns:
    - Complete bias report if analysis is complete
    - Status information if still processing
    - Error if request not found
    """
    # Check if request exists
    if request_id not in analysis_requests:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    request_info = analysis_requests[request_id]
    
    # Check if results are available
    if request_id in analysis_results:
        result = analysis_results[request_id]
        return BiasReportResponse(
            request_id=request_id,
            status="complete",
            overall_score=result.get("overall_score"),
            assessment=result.get("assessment"),
            bias_issues=result.get("bias_issues", []),
            recommendations=result.get("recommendations", []),
            processing_time_ms=result.get("processing_time_ms"),
            timestamp=result.get("timestamp")
        )
    
    # Still processing
    return BiasReportResponse(
        request_id=request_id,
        status=request_info["status"],
        timestamp=datetime.now(UTC).isoformat()
    )


@app.get("/api/status/{request_id}", response_model=AnalysisStatus)
async def check_status(request_id: str):
    """
    Check the processing status of an ad analysis request.
    
    Returns current stage: submitted, ingestion, analyzing_text, analyzing_visual, scoring, complete
    """
    if request_id not in analysis_requests:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    request_info = analysis_requests[request_id]
    
    # Get agent statuses
    agents_dict = None
    if request_id in agent_status:
        agents_dict = agent_status[request_id]
    
    return AnalysisStatus(
        request_id=request_id,
        status=request_info["status"],
        current_stage=request_info.get("current_stage", "unknown"),
        message=f"Analysis is in '{request_info.get('current_stage')}' stage",
        timestamp=datetime.now(UTC).isoformat(),
        agents=agents_dict
    )


@app.post("/api/agent/status")
async def update_agent_status(
    request_id: str,
    agent_name: str,
    status: str,
    message: Optional[str] = None
):
    """
    Endpoint for agents to update their processing status.
    
    Agents call this to report their current state:
    - "processing": Agent is actively working
    - "complete": Agent finished successfully
    - "error": Agent encountered an error
    """
    if request_id not in analysis_requests:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    if request_id not in agent_status:
        raise HTTPException(status_code=404, detail="Agent status not initialized")
    
    # Map agent names to keys
    agent_key_map = {
        "ingestion": "ingestion_agent",
        "text_bias": "text_bias_agent",
        "visual_bias": "visual_bias_agent",
        "scoring": "scoring_agent"
    }
    
    agent_key = agent_key_map.get(agent_name)
    if not agent_key:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name}")
    
    # Update agent status
    agent_status[request_id][agent_key]["status"] = status
    if message:
        agent_status[request_id][agent_key]["message"] = message
    agent_status[request_id][agent_key]["timestamp"] = datetime.now(UTC).isoformat()
    
    # Update overall stage
    if agent_name == "text_bias" and status == "processing":
        analysis_requests[request_id]["current_stage"] = "analyzing_text"
    elif agent_name == "visual_bias" and status == "processing":
        analysis_requests[request_id]["current_stage"] = "analyzing_visual"
    elif agent_name == "scoring" and status == "processing":
        analysis_requests[request_id]["current_stage"] = "scoring"
    
    logger.info(f"🔄 Agent status updated: {agent_name} -> {status} for request {request_id}")
    
    return {"success": True, "message": "Agent status updated"}


@app.post("/api/results/callback")
async def receive_results_callback(final_report: Dict[str, Any]):
    """
    Callback endpoint for Scoring Agent to send final results.
    
    This endpoint receives the completed bias analysis report from the Scoring Agent
    and stores it for retrieval by the frontend.
    """
    try:
        request_id = final_report.get("request_id")
        
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")
        
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Request ID not found")
        
        logger.info(f"📨 Received final results for request {request_id}")
        logger.info(f"   Overall Score: {final_report.get('overall_score')}")
        logger.info(f"   Total Issues: {final_report.get('total_issues')}")
        
        # Store results
        analysis_results[request_id] = final_report
        
        # Update request status
        analysis_requests[request_id]["status"] = "complete"
        analysis_requests[request_id]["current_stage"] = "complete"
        analysis_requests[request_id]["completed_at"] = datetime.now(UTC).isoformat()
        
        # Mark scoring agent as complete
        if request_id in agent_status:
            agent_status[request_id]["scoring_agent"]["status"] = "complete"
            agent_status[request_id]["scoring_agent"]["message"] = "Final report generated"
            agent_status[request_id]["scoring_agent"]["timestamp"] = datetime.now(UTC).isoformat()
        
        logger.info(f"✅ Results stored successfully for request {request_id}")
        
        return {
            "status": "success",
            "message": "Results received and stored",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error storing results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ChromaDB Info Endpoints ====================

@app.get("/api/collections")
async def get_collections():
    """Get information about ChromaDB collections"""
    try:
        return {
            "collections": {
                "ad_content": {
                    "count": db.get_collection_count(ChromaDB.COLLECTION_AD_CONTENT),
                    "description": "Complete ad submissions"
                },
                "text_patterns": {
                    "count": db.get_collection_count(ChromaDB.COLLECTION_TEXT_PATTERNS),
                    "description": "Historical text bias patterns"
                },
                "visual_patterns": {
                    "count": db.get_collection_count(ChromaDB.COLLECTION_VISUAL_PATTERNS),
                    "description": "Historical visual bias patterns"
                },
                "case_studies": {
                    "count": db.get_collection_count(ChromaDB.COLLECTION_CASE_STUDIES),
                    "description": "Complete case studies"
                }
            },
            "timestamp": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Legacy Compatibility ====================

@app.get("/documents")
async def get_documents():
    """Legacy endpoint for document retrieval"""
    try:
        # This was the original endpoint, maintaining for compatibility
        collection = db.get_collection(ChromaDB.COLLECTION_AD_CONTENT)
        count = collection.count()
        return {
            "documents": f"{count} documents in ad_content collection",
            "use_api": "/api/collections for detailed information"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting AdWhisper Backend API")
    print("=" * 60)
    print(f"📍 API: http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"💾 ChromaDB: {len(db._collections)} collections initialized")
    print("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    

