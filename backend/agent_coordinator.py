"""
Agent Coordinator - Bridge between FastAPI and uAgents

This module handles communication between the FastAPI server and the
distributed uAgent system. It provides functions to send messages to agents
and retrieve results.

Usage:
    from agent_coordinator import send_to_ingestion_agent, get_agent_status
"""

import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, UTC


class AgentCoordinator:
    """
    Coordinates communication between FastAPI and uAgents.
    Manages agent addresses and message routing.
    """
    
    def __init__(self):
        # Load agent addresses from environment
        self.ingestion_agent_addr = os.getenv("INGESTION_AGENT_ADDRESS", "")
        self.text_bias_agent_addr = os.getenv("TEXT_BIAS_AGENT_ADDRESS", "")
        self.visual_bias_agent_addr = os.getenv("VISUAL_BIAS_AGENT_ADDRESS", "")
        self.scoring_agent_addr = os.getenv("SCORING_AGENT_ADDRESS", "")
        
        # Track pending requests
        self.pending_requests = {}
        self.completed_results = {}
    
    def get_agent_addresses(self) -> Dict[str, str]:
        """Get all configured agent addresses"""
        return {
            "ingestion": self.ingestion_agent_addr,
            "text_bias": self.text_bias_agent_addr,
            "visual_bias": self.visual_bias_agent_addr,
            "scoring": self.scoring_agent_addr
        }
    
    def are_agents_configured(self) -> bool:
        """Check if all agent addresses are configured"""
        addresses = self.get_agent_addresses()
        return all(addr for addr in addresses.values())
    
    async def send_to_ingestion_agent(
        self,
        request_id: str,
        content_type: str,
        text_content: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send ad content to Ingestion Agent for processing.
        
        Args:
            request_id: Unique request identifier
            content_type: Type of content (text, image, video, mixed)
            text_content: Text content of the ad
            image_url: URL or path to image
            video_url: URL or path to video
            metadata: Additional metadata
        
        Returns:
            Response from ingestion agent
        """
        if not self.ingestion_agent_addr:
            raise ValueError("Ingestion agent address not configured")
        
        # Track this request
        self.pending_requests[request_id] = {
            "request_id": request_id,
            "status": "submitted_to_ingestion",
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # In production, use actual uAgent messaging:
        # from agents.ingestion_agent import AdContentRequest
        # request = AdContentRequest(
        #     request_id=request_id,
        #     content_type=content_type,
        #     text_content=text_content,
        #     image_url=image_url,
        #     video_url=video_url,
        #     metadata=metadata
        # )
        # 
        # # Send via uAgent context
        # response = await ctx.send(self.ingestion_agent_addr, request)
        
        # For now, return success response
        return {
            "success": True,
            "request_id": request_id,
            "message": "Request sent to ingestion agent",
            "agent_address": self.ingestion_agent_addr
        }
    
    async def get_analysis_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get the current status of an analysis request.
        
        Args:
            request_id: Request identifier
        
        Returns:
            Status information
        """
        if request_id in self.completed_results:
            return {
                "request_id": request_id,
                "status": "completed",
                "result": self.completed_results[request_id]
            }
        
        if request_id in self.pending_requests:
            return {
                "request_id": request_id,
                "status": "processing",
                "details": self.pending_requests[request_id]
            }
        
        return {
            "request_id": request_id,
            "status": "not_found",
            "message": "Request ID not found"
        }
    
    async def receive_result(self, request_id: str, result: Dict[str, Any]):
        """
        Receive a completed result from the agent system.
        
        This would be called by a message handler that receives
        the final report from the scoring agent.
        
        Args:
            request_id: Request identifier
            result: Complete analysis result
        """
        # Move from pending to completed
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]
        
        self.completed_results[request_id] = {
            **result,
            "received_at": datetime.now(UTC).isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "pending_requests": len(self.pending_requests),
            "completed_results": len(self.completed_results),
            "agents_configured": self.are_agents_configured(),
            "agent_addresses": self.get_agent_addresses()
        }


# Global coordinator instance
coordinator = AgentCoordinator()


# Convenience functions for FastAPI integration

async def send_ad_for_analysis(
    request_id: str,
    content_type: str,
    text_content: Optional[str] = None,
    image_url: Optional[str] = None,
    video_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to send ad for analysis.
    """
    return await coordinator.send_to_ingestion_agent(
        request_id=request_id,
        content_type=content_type,
        text_content=text_content,
        image_url=image_url,
        video_url=video_url,
        metadata=metadata
    )


async def get_agent_status(request_id: str) -> Dict[str, Any]:
    """
    Convenience function to get analysis status.
    """
    return await coordinator.get_analysis_status(request_id)


def get_coordinator_stats() -> Dict[str, Any]:
    """
    Get coordinator statistics.
    """
    return coordinator.get_statistics()


def are_agents_ready() -> bool:
    """
    Check if all agents are configured and ready.
    """
    return coordinator.are_agents_configured()


# Example usage in FastAPI:
# 
# from agent_coordinator import send_ad_for_analysis, get_agent_status
# 
# @app.post("/api/analyze-ad")
# async def analyze_ad(...):
#     result = await send_ad_for_analysis(
#         request_id=request_id,
#         content_type="mixed",
#         text_content=text_content,
#         image_url=image_url
#     )
#     return {"request_id": request_id, "status": "submitted"}
# 
# @app.get("/api/status/{request_id}")
# async def check_status(request_id: str):
#     status = await get_agent_status(request_id)
#     return status

