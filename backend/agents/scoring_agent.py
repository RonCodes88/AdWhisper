"""
Scoring Agent - Ad Bias Detection System

Role: Result Aggregation and Final Bias Assessment
Responsibilities:
- Receive results from Text and Visual Bias Agents
- Use Claude LLM to generate comprehensive summary of findings
- Aggregate bias issues and calculate average scores
- Generate executive summary, top concerns, and recommendations
- Provide clean, actionable insights for frontend display
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
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import shared models
from agents.shared_models import (
    BiasAnalysisComplete,
    AgentError
)


# Message Models (importing from other agents)
class BiasCategory(str, Enum):
    GENDER = "gender_bias"
    RACIAL = "racial_bias"
    AGE = "age_bias"
    SOCIOECONOMIC = "socioeconomic_bias"
    DISABILITY = "disability_bias"
    LGBTQ = "lgbtq_bias"
    REPRESENTATION = "representation_bias"
    CONTEXTUAL = "contextual_bias"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentReport(Model):
    """Generic agent report structure"""
    request_id: str
    agent: str
    bias_detected: bool
    score: float
    bias_count: int
    confidence: float
    raw_report: Optional[Dict[str, Any]] = None


class ScoreBreakdown(Model):
    """Detailed score breakdown"""
    text_score: float
    visual_score: float
    intersectional_penalty: float = 0.0
    weighted_score: float


class BiasIssue(Model):
    """Individual bias issue summary"""
    category: str
    severity: SeverityLevel
    source: str  # "text_bias_agent" or "visual_bias_agent"
    description: str  # The "context" field from agents
    impact: str  # The real-world harm this bias perpetuates
    examples: Optional[List[str]] = None
    confidence: float


class Recommendation(Model):
    """Actionable recommendation"""
    priority: SeverityLevel
    category: str
    action: str
    impact: str


class FinalBiasReport(Model):
    """Comprehensive final bias assessment report for frontend consumption"""
    request_id: str
    
    # Source content information (for frontend display)
    content_url: Optional[str] = None
    content_type: Optional[str] = None  # "video", "image", "text"
    
    # Overall assessment
    overall_bias_score: float  # 0-10 scale
    assessment: str  # Human-readable assessment text
    bias_level: str  # "Minimal Bias", "Minor Bias", "Moderate Bias", "Significant Bias"
    
    # Score breakdown for frontend cards (as dict for proper serialization)
    score_breakdown: Dict[str, Any]
    
    # Analysis status (for progress indicators)
    text_analysis_status: str = "completed"
    visual_analysis_status: str = "completed"
    
    # Aggregated findings
    total_issues: int
    critical_severity_count: int = 0
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    
    # Detailed issues with full context (as dicts for proper serialization)
    bias_issues: Optional[List[Dict[str, Any]]] = None
    top_concerns: Optional[List[str]] = None
    
    # Recommendations (as dicts for proper serialization)
    recommendations: Optional[List[Dict[str, Any]]] = None
    
    # RAG context
    similar_cases: Optional[List[str]] = None
    benchmark_comparison: Optional[Dict[str, Any]] = None
    
    # Original agent reports (for transparency and debugging)
    text_report: Optional[Dict[str, Any]] = None
    visual_report: Optional[Dict[str, Any]] = None
    
    # Metadata
    confidence: float
    processing_time_ms: int = 0
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class ScoringRequest(Model):
    """Request to aggregate and score results"""
    request_id: str
    text_report: Optional[Dict[str, Any]] = None
    visual_report: Optional[Dict[str, Any]] = None
    chromadb_collection_id: str
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


# Initialize Scoring Agent
scoring_agent = Agent(
    name="scoring_agent",
    seed="ad_bias_scoring_agent_unique_seed_2024",
    port=8103,
    endpoint=["http://localhost:8103/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for scoring and aggregation
scoring_protocol = Protocol(name="scoring_protocol", version="1.0")

# ASI:ONE LLM agent addresses
OPENAI_AGENT = 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y'
CLAUDE_AGENT = 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx'

# Scoring weights (configurable)
SCORING_WEIGHTS = {
    "text_weight": 0.40,
    "visual_weight": 0.40,
    "intersectional_weight": 0.20
}

# Module-level storage for final reports (persistent across REST calls)
FINAL_REPORTS_CACHE: Dict[str, Dict[str, Any]] = {}


@scoring_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Scoring Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {scoring_agent.address}")
    ctx.logger.info(f"🔧 Role: Result Aggregation and Final Bias Assessment")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8103/submit")
    ctx.logger.info(f"📊 REST API Endpoints:")
    ctx.logger.info(f"   POST /score - Trigger scoring by fetching reports via HTTP")
    ctx.logger.info(f"   GET /report/{{request_id}} - Retrieve final bias report")
    ctx.logger.info(f"⚖️ Scoring Weights: Text={SCORING_WEIGHTS['text_weight']}, Visual={SCORING_WEIGHTS['visual_weight']}, Intersectional={SCORING_WEIGHTS['intersectional_weight']}")
    ctx.logger.info(f"⚡ Ready to aggregate and score bias reports via REST")


@scoring_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Scoring Agent shutting down...")


# REST API endpoint for frontend to retrieve final report
class ReportRequest(Model):
    """Request model for retrieving a report"""
    request_id: str


@scoring_agent.on_rest_get("/report/{request_id}", FinalBiasReport)
async def get_final_report(ctx: Context, request_id: str) -> FinalBiasReport:
    """
    REST endpoint for frontend to retrieve the final bias report.
    
    Usage: GET http://localhost:8103/report/{request_id}
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 GET /report/{request_id} - REST request received")
    ctx.logger.info("=" * 80)
    
    # Check in-memory cache
    ctx.logger.info(f"🔍 Checking cache for request_id: {request_id}")
    ctx.logger.info(f"📦 Cache keys available: {list(FINAL_REPORTS_CACHE.keys())}")
    
    if request_id in FINAL_REPORTS_CACHE:
        report_data = FINAL_REPORTS_CACHE[request_id]
        ctx.logger.info(f"✅ Report found in cache!")
        ctx.logger.info(f"   📊 Score: {report_data.get('overall_bias_score', 'N/A')}")
        ctx.logger.info(f"   📋 Issues: {report_data.get('total_issues', 'N/A')}")
        return FinalBiasReport(**report_data)
    
    # Report not found
    ctx.logger.warning(f"❌ Report NOT found for request_id: {request_id}")
    ctx.logger.warning(f"   Available keys: {list(FINAL_REPORTS_CACHE.keys())}")
    
    # Return a "not found" report
    return FinalBiasReport(
        request_id=request_id,
        overall_bias_score=0.0,
        assessment="Report not found. Analysis may still be in progress or request_id is invalid.",
        bias_level="Unknown",
        score_breakdown={
            "text_score": 0.0,
            "visual_score": 0.0,
            "intersectional_penalty": 0.0,
            "weighted_score": 0.0
        },
        text_analysis_status="not_found",
        visual_analysis_status="not_found",
        total_issues=0,
        confidence=0.0,
        timestamp=datetime.now(UTC).isoformat()
    )


class ScoreRequest(Model):
    """Request model for scoring endpoint - now accepts reports directly"""
    request_id: str
    text_report: Optional[Dict[str, Any]] = None
    visual_report: Optional[Dict[str, Any]] = None
    chromadb_collection_id: Optional[str] = None


@scoring_agent.on_rest_post("/score", ScoreRequest, FinalBiasReport)
async def score_via_rest(ctx: Context, req: ScoreRequest) -> FinalBiasReport:
    """
    REST POST endpoint for scoring with reports passed directly as context.
    Ingestion agent calls this with both text and visual reports.
    
    Usage: POST http://localhost:8103/score
    Body: {
        "request_id": "your_request_id",
        "text_report": {...},  # Full text bias report
        "visual_report": {...},  # Full visual bias report
        "chromadb_collection_id": "optional_collection_id"
    }
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🎯 SCORING AGENT - REST /score ENDPOINT")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 Received scoring request for: {req.request_id}")
    ctx.logger.info(f"   📊 Text Report: {'Provided' if req.text_report else 'Missing'}")
    ctx.logger.info(f"   📊 Visual Report: {'Provided' if req.visual_report else 'Missing'}")
    
    try:
        # Get reports from request (passed directly as context)
        text_report = req.text_report
        visual_report = req.visual_report
        
        # Validate reports
        if not text_report and not visual_report:
            ctx.logger.error(f"❌ No reports provided in request")
            return FinalBiasReport(
                request_id=req.request_id,
                overall_bias_score=0.0,
                assessment="Unable to generate final report. No bias reports were provided. Please ensure both Text and Visual Bias Agents have completed their analysis.",
                bias_level="Error",
                score_breakdown={
                    "text_score": 0.0,
                    "visual_score": 0.0,
                    "intersectional_penalty": 0.0,
                    "weighted_score": 0.0
                },
                text_analysis_status="missing",
                visual_analysis_status="missing",
                total_issues=0,
                bias_issues=[],
                top_concerns=[],
                recommendations=[],
                similar_cases=[],
                benchmark_comparison={},
                confidence=0.0,
                timestamp=datetime.now(UTC).isoformat()
            )
        
        # Log report details
        if text_report:
            ctx.logger.info(f"   ✅ Text report received:")
            ctx.logger.info(f"      Score: {text_report.get('overall_text_score', 'N/A')}")
            ctx.logger.info(f"      Issues: {len(text_report.get('bias_instances', []))}")
        else:
            ctx.logger.warning(f"   ⚠️  Text report not provided")
        
        if visual_report:
            ctx.logger.info(f"   ✅ Visual report received:")
            ctx.logger.info(f"      Score: {visual_report.get('overall_visual_score', 'N/A')}")
            ctx.logger.info(f"      Issues: {len(visual_report.get('bias_instances', []))}")
        else:
            ctx.logger.warning(f"   ⚠️  Visual report not provided")
        
        # Process final scoring
        ctx.logger.info(f"🎯 Processing final scoring with provided reports...")
        collection_id = req.chromadb_collection_id or req.request_id
        final_report = await process_final_scoring(
            ctx,
            req.request_id,
            text_report or {},
            visual_report or {},
            collection_id
        )
        
        # Store final report in memory cache for frontend retrieval
        report_dict = final_report.dict()
        FINAL_REPORTS_CACHE[req.request_id] = report_dict
        ctx.logger.info(f"💾 Final report stored in cache for request_id: {req.request_id}")
        ctx.logger.info(f"📦 Cache now contains {len(FINAL_REPORTS_CACHE)} report(s)")
        ctx.logger.info(f"🔑 Cache keys: {list(FINAL_REPORTS_CACHE.keys())}")
        
        ctx.logger.info(f"✅ Scoring complete!")
        ctx.logger.info(f"   📊 Final Score: {final_report.overall_bias_score:.1f}/10")
        ctx.logger.info(f"   📋 Total Issues: {final_report.total_issues}")
        ctx.logger.info(f"   🏷️ Bias Level: {final_report.bias_level}")
        ctx.logger.info("=" * 80)
        
        return final_report
        
    except Exception as e:
        ctx.logger.error(f"❌ Error during scoring: {e}")
        import traceback
        ctx.logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # Return error report
        return FinalBiasReport(
            request_id=req.request_id,
            overall_bias_score=0.0,
            assessment=f"Error occurred during scoring: {str(e)}",
            bias_level="Error",
            score_breakdown={
                "text_score": 0.0,
                "visual_score": 0.0,
                "intersectional_penalty": 0.0,
                "weighted_score": 0.0
            },
            text_analysis_status="error",
            visual_analysis_status="error",
            total_issues=0,
            bias_issues=[],
            top_concerns=[],
            recommendations=[],
            similar_cases=[],
            benchmark_comparison={},
            confidence=0.0,
            timestamp=datetime.now(UTC).isoformat()
        )


@scoring_protocol.on_message(model=BiasAnalysisComplete, replies=FinalBiasReport)
async def handle_bias_analysis_complete(ctx: Context, sender: str, msg: BiasAnalysisComplete):
    """
    Receive results from Text or Visual Bias agents.
    Wait for both agents before generating final report.
    """
    try:
        ctx.logger.info("=" * 80)
        ctx.logger.info(f"📨 RECEIVED MESSAGE FROM: {msg.sender_agent}")
        ctx.logger.info(f"   Request ID: {msg.request_id}")
        ctx.logger.info(f"   Sender: {sender}")
        ctx.logger.info(f"   Report keys: {list(msg.report.keys())}")
        ctx.logger.info("=" * 80)

        # Store the report in temporary in-memory storage
        temp_storage_key = f"temp_reports_{msg.request_id}"
        if temp_storage_key not in FINAL_REPORTS_CACHE:
            FINAL_REPORTS_CACHE[temp_storage_key] = {}
        
        reports = FINAL_REPORTS_CACHE[temp_storage_key]

        # Add this agent's report
        if msg.sender_agent == "text_bias_agent":
            reports["text"] = msg.report
            ctx.logger.info(f"✅ Text bias report stored in cache")
        elif msg.sender_agent == "visual_bias_agent":
            reports["visual"] = msg.report
            ctx.logger.info(f"✅ Visual bias report stored in cache")

        # Check if we have both reports
        has_text = "text" in reports
        has_visual = "visual" in reports

        ctx.logger.info(f"📊 Report status for {msg.request_id}: Text={has_text}, Visual={has_visual}")

        # If we have both, generate final report
        if has_text and has_visual:
            ctx.logger.info(f"🎯 Both reports received! Generating final assessment...")

            # Create a ScoringRequest-like structure
            text_report = reports["text"]
            visual_report = reports["visual"]

            # Extract ChromaDB collection ID from either report
            collection_id = text_report.get("chromadb_collection_id", msg.request_id)

            # Process the final report
            final_report = await process_final_scoring(
                ctx,
                msg.request_id,
                text_report,
                visual_report,
                collection_id
            )

            # Store final report in cache for frontend retrieval
            FINAL_REPORTS_CACHE[msg.request_id] = final_report.dict()
            ctx.logger.info(f"💾 Final report stored in cache for request_id: {msg.request_id}")
            
            # Clean up temporary reports storage
            del FINAL_REPORTS_CACHE[temp_storage_key]

            ctx.logger.info(f"✅ Final report generated successfully!")
            ctx.logger.info(f"📊 Final Score: {final_report.overall_bias_score:.1f}/10")
            ctx.logger.info(f"📋 Total Issues: {final_report.total_issues}")
            ctx.logger.info(f"   🔴 Critical: {final_report.critical_severity_count}")
            ctx.logger.info(f"   🟠 High: {final_report.high_severity_count}")
            ctx.logger.info(f"   🟡 Medium: {final_report.medium_severity_count}")
            ctx.logger.info(f"   🟢 Low: {final_report.low_severity_count}")
            ctx.logger.info(f"💡 Recommendations: {len(final_report.recommendations or [])}")
            
            # Report is now available for frontend at: GET /report/{request_id}

        else:
            ctx.logger.info(f"⏳ Waiting for remaining reports...")
            if not has_text:
                ctx.logger.info(f"   - Still waiting for: Text Bias Agent")
            if not has_visual:
                ctx.logger.info(f"   - Still waiting for: Visual Bias Agent")

    except Exception as e:
        ctx.logger.error(f"❌ Error handling bias analysis: {e}")


@scoring_protocol.on_message(model=AgentError)
async def handle_agent_error(ctx: Context, sender: str, msg: AgentError):
    """
    Handle errors from bias agents.
    """
    ctx.logger.warning(f"⚠️ Received error from {msg.agent_name} for request {msg.request_id}")
    ctx.logger.warning(f"   Error type: {msg.error_type}")
    ctx.logger.warning(f"   Error message: {msg.error_message}")

    # Still try to generate a report with partial data
    # This allows the system to continue even if one agent fails


async def generate_llm_summary(
    ctx: Context,
    text_report: Dict[str, Any],
    visual_report: Dict[str, Any],
    text_score: float,
    visual_score: float,
    final_score: float,
    bias_issues: List[BiasIssue]
) -> Dict[str, Any]:
    """
    HARDCODED: Use Claude LLM to generate a comprehensive summary of the bias analysis results.
    Takes the outputs from text and visual bias agents and creates a unified, actionable summary.
    """
    from anthropic import Anthropic
    
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    ctx.logger.info(f"🧠 [HARDCODED] Generating Claude LLM summary for {len(bias_issues)} bias instances...")
    
    # Prepare the input data for the LLM
    text_biases = text_report.get("bias_instances", [])
    visual_biases = visual_report.get("bias_instances", [])
    
    # Create a structured prompt for clean, actionable output
    prompt = f"""You are an expert advertising bias analyst. Analyze these results and provide clear, actionable insights.

**TEXT ANALYSIS RESULTS**
Score: {text_score:.1f}/10 (10 = no bias, 0 = severe bias)
Issues Found: {len(text_biases)}

Text Bias Details:
{json.dumps(text_biases, indent=2) if text_biases else "No text biases detected"}

**VISUAL ANALYSIS RESULTS**
Score: {visual_score:.1f}/10 (10 = no bias, 0 = severe bias)
Issues Found: {len(visual_biases)}

Visual Bias Details:
{json.dumps(visual_biases, indent=2) if visual_biases else "No visual biases detected"}

**COMBINED SCORE**: {final_score:.1f}/10

Your task: Generate a comprehensive, actionable summary in JSON format.

Requirements:
1. **assessment**: Write 2-3 clear sentences explaining the overall findings. Be specific about what biases were found and their severity.
2. **top_concerns**: List 3-5 specific, concrete concerns (not vague statements). Include what the bias is and why it matters.
3. **recommendations**: Provide 3-5 actionable recommendations with these fields:
   - priority: "critical", "high", "medium", or "low"
   - category: Short category name (e.g., "Gender Representation", "Language Choice")
   - action: Specific, actionable step (e.g., "Replace 'cancel her drama' with neutral language")
   - impact: Expected positive outcome

Output format (JSON only, no markdown):
{{
    "assessment": "Clear summary here",
    "top_concerns": [
        "Specific concern 1 with details",
        "Specific concern 2 with details",
        "Specific concern 3 with details"
    ],
    "recommendations": [
        {{
            "priority": "high",
            "category": "Category Name",
            "action": "Specific actionable step",
            "impact": "Expected positive outcome"
        }}
    ]
}}

Be direct and specific. Focus on the most impactful issues. Respond with ONLY valid JSON."""
    
    try:
        ctx.logger.info(f"📤 Calling Claude API (haiku-4-5)...")
        
        # Call Claude API
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast model
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        ctx.logger.info(f"📥 Claude response received ({len(response_text)} chars)")
        
        # Try to extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        summary = json.loads(response_text)
        
        # Log the generated summary
        ctx.logger.info(f"✅ Claude LLM summary generated successfully!")
        ctx.logger.info(f"📋 Assessment: {summary.get('assessment', '')[:100]}...")
        ctx.logger.info(f"🎯 Top concerns: {len(summary.get('top_concerns', []))}")
        ctx.logger.info(f"💡 Recommendations: {len(summary.get('recommendations', []))}")
        
        return summary
        
    except Exception as e:
        ctx.logger.error(f"❌ Error generating Claude LLM summary: {e}")
        import traceback
        ctx.logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # Return fallback summary
        ctx.logger.warning(f"⚠️  Using fallback summary due to LLM error")
        return {
            "assessment": f"Analysis complete. Found {len(bias_issues)} bias issues across text and visual content. Text score: {text_score:.1f}/10, Visual score: {visual_score:.1f}/10. Review detailed findings for specific concerns.",
            "top_concerns": [
                f"Text analysis identified {len(text_biases)} potential bias issues requiring attention",
                f"Visual analysis identified {len(visual_biases)} potential representation concerns",
                "Detailed bias instances available for review in the full report"
            ] if bias_issues else ["No significant bias concerns detected in this advertisement"],
            "recommendations": [
                {
                    "priority": "medium",
                    "category": "Content Review",
                    "action": "Review detailed bias instances from both text and visual analysis for specific improvement areas",
                    "impact": "Ensures all identified bias concerns are properly addressed and content is more inclusive"
                }
            ] if bias_issues else [
                {
                    "priority": "low",
                    "category": "Continuous Improvement",
                    "action": "Continue monitoring content for potential bias in future campaigns",
                    "impact": "Maintains high standards of inclusivity and representation"
                }
            ]
        }


async def process_final_scoring(
    ctx: Context,
    request_id: str,
    text_report: Dict[str, Any],
    visual_report: Dict[str, Any],
    collection_id: str
) -> FinalBiasReport:
    """
    Process final scoring with both reports using LLM summarization.
    """
    start_time = datetime.now(UTC)

    ctx.logger.info(f"🧠 Using LLM to generate comprehensive bias summary...")
    
    # Get scores from reports
    text_score = text_report.get("overall_text_score", 5.0)
    visual_score = visual_report.get("overall_visual_score", 5.0)
    
    # Average the scores
    final_score = (text_score + visual_score) / 2.0
    
    ctx.logger.info(f"📊 Text Score: {text_score:.1f}, Visual Score: {visual_score:.1f}, Final Score: {final_score:.1f}")
    
    # Aggregate bias issues from both reports
    bias_issues = await aggregate_bias_issues(ctx, text_report, visual_report)
    severity_counts = categorize_by_severity(bias_issues)
    
    # Use LLM to generate comprehensive summary
    llm_summary = await generate_llm_summary(ctx, text_report, visual_report, text_score, visual_score, final_score, bias_issues)
    
    # Log Claude's summary to console
    ctx.logger.info("=" * 80)
    ctx.logger.info("🤖 CLAUDE LLM GENERATED SUMMARY")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📋 Assessment:")
    ctx.logger.info(f"   {llm_summary.get('assessment', 'N/A')}")
    ctx.logger.info(f"")
    ctx.logger.info(f"🎯 Top Concerns ({len(llm_summary.get('top_concerns', []))} identified):")
    for i, concern in enumerate(llm_summary.get('top_concerns', []), 1):
        ctx.logger.info(f"   {i}. {concern}")
    ctx.logger.info(f"")
    ctx.logger.info(f"💡 Recommendations ({len(llm_summary.get('recommendations', []))} generated):")
    for i, rec in enumerate(llm_summary.get('recommendations', []), 1):
        ctx.logger.info(f"   {i}. [{rec.get('priority', 'medium').upper()}] {rec.get('category', 'N/A')}")
        ctx.logger.info(f"      Action: {rec.get('action', 'N/A')}")
        ctx.logger.info(f"      Impact: {rec.get('impact', 'N/A')}")
    ctx.logger.info("=" * 80)
    
    # Extract content metadata
    content_url = text_report.get("content_url") or visual_report.get("content_url")
    content_type = text_report.get("content_type") or visual_report.get("content_type", "video")
    
    # Calculate processing time
    end_time = datetime.now(UTC)
    processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
    
    # Create final report with LLM summary
    bias_issues_dicts = convert_bias_issues_to_dicts(bias_issues)
    
    final_report = FinalBiasReport(
        request_id=request_id,
        content_url=content_url,
        content_type=content_type,
        overall_bias_score=final_score,
        assessment=llm_summary.get("assessment", "Analysis complete"),
        bias_level=get_bias_level(final_score),
        score_breakdown={
            "text_score": text_score,
            "visual_score": visual_score,
            "intersectional_penalty": 0.0,
            "weighted_score": final_score
        },
        text_analysis_status="completed",
        visual_analysis_status="completed",
        total_issues=len(bias_issues),
        critical_severity_count=severity_counts.get("critical", 0),
        high_severity_count=severity_counts.get("high", 0),
        medium_severity_count=severity_counts.get("medium", 0),
        low_severity_count=severity_counts.get("low", 0),
        bias_issues=bias_issues_dicts,
        top_concerns=llm_summary.get("top_concerns", []),
        recommendations=llm_summary.get("recommendations", []),
        similar_cases=[],
        benchmark_comparison={},
        text_report=text_report,
        visual_report=visual_report,
        confidence=calculate_overall_confidence(text_report, visual_report),
        processing_time_ms=processing_time_ms
    )
    
    ctx.logger.info(f"✅ Final report generated with LLM summary: Score={final_score:.1f}, Issues={len(bias_issues)}, Time={processing_time_ms}ms")
    
    return final_report


@scoring_protocol.on_message(model=ScoringRequest, replies=FinalBiasReport)
async def handle_scoring_request(ctx: Context, sender: str, msg: ScoringRequest):
    """
    Aggregate results from Text and Visual agents and produce final assessment using LLM.
    """
    try:
        ctx.logger.info(f"📨 Received scoring request: {msg.request_id}")
        
        # Use the simplified processing function
        final_report = await process_final_scoring(
            ctx,
            msg.request_id,
            msg.text_report or {},
            msg.visual_report or {},
            msg.chromadb_collection_id
        )
        
        # Send report back to requester
        await ctx.send(sender, final_report)
        
        ctx.logger.info(f"✅ Scoring complete for request {msg.request_id}")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error during scoring for {msg.request_id}: {e}")
        import traceback
        ctx.logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # Send error report
        error_report = FinalBiasReport(
            request_id=msg.request_id,
            overall_bias_score=5.0,
            assessment="Error occurred during scoring analysis. Please try again or contact support.",
            bias_level="Unknown",
            score_breakdown={
                "text_score": 5.0,
                "visual_score": 5.0,
                "intersectional_penalty": 0.0,
                "weighted_score": 5.0
            },
            text_analysis_status="error",
            visual_analysis_status="error",
            total_issues=0,
            bias_issues=[],
            top_concerns=["Error during processing"],
            recommendations=[],
            similar_cases=[],
            confidence=0.0
        )
        await ctx.send(sender, error_report)


# Removed complex scoring functions - now using LLM summarization


async def aggregate_bias_issues(
    ctx: Context,
    text_report: Dict[str, Any],
    visual_report: Dict[str, Any]
) -> List[BiasIssue]:
    """
    Aggregate all bias issues from both agents with full context.
    Extracts detailed information including examples, context, impact, and confidence.
    """
    issues = []
    
    ctx.logger.info(f"🔍 DEBUG - Text report keys: {list(text_report.keys())}")
    ctx.logger.info(f"🔍 DEBUG - Visual report keys: {list(visual_report.keys())}")
    
    # Aggregate text biases from bias_instances
    if "bias_instances" in text_report:
        ctx.logger.info(f"   📝 Processing {len(text_report['bias_instances'])} text bias instances...")
        ctx.logger.info(f"   🔍 DEBUG - First instance structure: {text_report['bias_instances'][0] if text_report['bias_instances'] else 'No instances'}")
        for bias in text_report["bias_instances"]:
            try:
                issue = BiasIssue(
                    category=bias.get("bias_type", "unknown"),
                    severity=SeverityLevel(bias.get("severity", "low")),
                    source="text_bias_agent",
                    description=bias.get("context", "No description provided"),
                    impact=bias.get("impact", "Impact not specified"),
                    examples=bias.get("examples", []),
                    confidence=bias.get("confidence", 0.5)
                )
                issues.append(issue)
                ctx.logger.info(f"      ✓ {issue.category} ({issue.severity.value})")
            except Exception as e:
                ctx.logger.warning(f"      ⚠️ Failed to parse text bias: {e}")
    
    # Aggregate visual biases from bias_instances
    if "bias_instances" in visual_report:
        ctx.logger.info(f"   👁️ Processing {len(visual_report['bias_instances'])} visual bias instances...")
        for bias in visual_report["bias_instances"]:
            try:
                issue = BiasIssue(
                    category=bias.get("bias_type", "unknown"),
                    severity=SeverityLevel(bias.get("severity", "low")),
                    source="visual_bias_agent",
                    description=bias.get("context", "No description provided"),
                    impact=bias.get("impact", "Impact not specified"),
                    examples=bias.get("examples", []),
                    confidence=bias.get("confidence", 0.5)
                )
                issues.append(issue)
                ctx.logger.info(f"      ✓ {issue.category} ({issue.severity.value})")
            except Exception as e:
                ctx.logger.warning(f"      ⚠️ Failed to parse visual bias: {e}")
    
    ctx.logger.info(f"📋 Successfully aggregated {len(issues)} bias issues total")
    return issues


def categorize_by_severity(issues: List[BiasIssue]) -> Dict[str, int]:
    """
    Categorize issues by severity level.
    """
    counts = {"high": 0, "medium": 0, "low": 0, "critical": 0}
    
    for issue in issues:
        counts[issue.severity.value] += 1
    
    return counts


# Removed - now using LLM to generate recommendations


def get_bias_level(score: float) -> str:
    """
    Get bias level label for frontend display.
    Matches the format shown in the UI: "Minimal Bias", "Minor Bias", etc.
    """
    if score >= 9.0:
        return "Minimal Bias"
    elif score >= 7.0:
        return "Minor Bias"
    elif score >= 4.0:
        return "Moderate Bias"
    else:
        return "Significant Bias"


def calculate_overall_confidence(text_report: Dict[str, Any], visual_report: Dict[str, Any]) -> float:
    """
    Calculate overall confidence from both reports.
    """
    text_conf = text_report.get("confidence", 0.5)
    visual_conf = visual_report.get("confidence", 0.5)
    
    # Average confidence
    overall = (text_conf + visual_conf) / 2.0
    return overall


def convert_bias_issues_to_dicts(issues: List[BiasIssue]) -> List[Dict[str, Any]]:
    """
    Convert BiasIssue Model objects to dicts for proper serialization.
    """
    return [
        {
            "category": issue.category,
            "severity": issue.severity.value,  # Convert enum to string
            "source": issue.source,
            "description": issue.description,
            "impact": issue.impact,
            "examples": issue.examples,
            "confidence": issue.confidence
        }
        for issue in issues
    ]


def convert_recommendations_to_dicts(recommendations: List[Recommendation]) -> List[Dict[str, Any]]:
    """
    Convert Recommendation Model objects to dicts for proper serialization.
    """
    return [
        {
            "priority": rec.priority.value,  # Convert enum to string
            "category": rec.category,
            "action": rec.action,
            "impact": rec.impact
        }
        for rec in recommendations
    ]


# Removed - ChromaDB storage not needed for simple LLM summarization


# Include protocol
scoring_agent.include(scoring_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        ⚖️ SCORING AGENT - Ad Bias Detection                  ║
╚══════════════════════════════════════════════════════════════╝

Role: Result Aggregation and LLM-Powered Summary Generation

Capabilities:
  ✓ Receives analysis results from Text and Visual Bias Agents
  ✓ Uses Claude LLM to generate comprehensive summaries
  ✓ Aggregates bias issues from both text and visual analysis
  ✓ Generates executive assessment and top concerns
  ✓ Provides actionable recommendations
  ✓ REST API for frontend report retrieval

REST API Endpoints:
  • POST /score - Generate final report with LLM summary
    URL: http://localhost:8103/score
    Body: {
      "request_id": "your_request_id",
      "text_report": {...},  // Full text bias report
      "visual_report": {...},  // Full visual bias report
    }
  • GET /report/{request_id} - Retrieve final bias report (JSON)
    URL: http://localhost:8103/report/{request_id}

Architecture:
  • Simple LLM-based summarization approach
  • Takes text and visual bias reports as direct input
  • Claude Haiku generates executive summary and recommendations
  • No polling - processes reports passed in the request

Report Output Structure:
  • Overall Score (average of text + visual scores)
  • LLM-Generated Assessment (executive summary)
  • Top Concerns (prioritized by LLM)
  • Recommendations (actionable, generated by LLM)
  • Bias Issues (aggregated from both agents)
  • Severity Counts (critical, high, medium, low)
  • Original Reports (for transparency)

Scoring Scale (0-10):
  • 0-3: Significant bias (high concern)
  • 4-6: Moderate bias (needs revision)
  • 7-8: Minor bias (minor improvements)
  • 9-10: Minimal bias (approved)

Configuration:
  • Port: 8103
  • Endpoint: http://localhost:8103/submit
  • LLM: Claude Haiku 4.5 (fast summarization)

📍 Waiting for scoring requests...
🛑 Stop with Ctrl+C
    """)
    scoring_agent.run()

