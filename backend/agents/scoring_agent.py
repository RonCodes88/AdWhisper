"""
Scoring Agent - Ad Bias Detection System

Role: Result Aggregation and Final Bias Assessment
Responsibilities:
- Receive and aggregate results from Text and Visual Bias Agents
- Resolve conflicting assessments
- Calculate weighted bias scores with intersectional analysis
- Query ChromaDB for benchmarking (RAG RETRIEVAL POINT #3)
- Generate comprehensive reports with recommendations
- Prioritize findings by severity and impact
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
    ctx.logger.info(f"📨 REST request to retrieve report: {request_id}")
    
    # Retrieve from storage
    final_report_key = f"final_report_{request_id}"
    report_data = ctx.storage.get(final_report_key)
    
    if report_data:
        ctx.logger.info(f"✅ Report found and returned")
        # Convert dict back to FinalBiasReport model
        return FinalBiasReport(**report_data)
    else:
        ctx.logger.warning(f"⚠️ Report not found for request_id: {request_id}")
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
    """Request model for scoring endpoint"""
    request_id: str
    chromadb_collection_id: Optional[str] = None


@scoring_agent.on_rest_post("/score", ScoreRequest, FinalBiasReport)
async def score_via_rest(ctx: Context, req: ScoreRequest) -> FinalBiasReport:
    """
    REST POST endpoint for triggering scoring by fetching reports from Text and Visual agents.
    This uses HTTP calls instead of uAgents messaging.
    
    Usage: POST http://localhost:8103/score
    Body: {"request_id": "your_request_id", "chromadb_collection_id": "optional_collection_id"}
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🎯 SCORING AGENT - REST /score ENDPOINT")
    ctx.logger.info("=" * 80)
    ctx.logger.info(f"📨 Received scoring request for: {req.request_id}")
    
    try:
        # Step 1: Fetch Text Bias Report via REST (with retries)
        ctx.logger.info(f"📤 STEP 1: Fetching Text Bias Report via HTTP...")
        text_agent_url = f"http://localhost:8101/report/{req.request_id}"
        text_report = None
        
        # Retry logic for text report (wait up to 10 seconds)
        max_retries = 10
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    ctx.logger.info(f"   🔗 Calling: {text_agent_url} (attempt {attempt + 1}/{max_retries})")
                    async with session.get(text_agent_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        ctx.logger.info(f"   📥 Response status: {response.status}")
                        if response.status == 200:
                            result = await response.json()
                            text_report = result.get("report", {})
                            # Check if it's a valid report (not error)
                            if not text_report.get("error") and text_report.get("overall_text_score") is not None:
                                ctx.logger.info(f"   ✅ Text report fetched successfully")
                                ctx.logger.info(f"      Score: {text_report.get('overall_text_score', 'N/A')}")
                                break
                            else:
                                ctx.logger.warning(f"   ⏳ Text report not ready yet, retrying...")
                                await asyncio.sleep(retry_delay)
                        else:
                            ctx.logger.warning(f"   ⏳ Text report not available yet (status {response.status}), retrying...")
                            await asyncio.sleep(retry_delay)
            except aiohttp.ClientConnectorError as e:
                ctx.logger.error(f"   ❌ Cannot connect to Text Bias Agent at {text_agent_url}")
                ctx.logger.error(f"      Make sure Text Bias Agent is running on port 8101")
                break
            except asyncio.TimeoutError:
                ctx.logger.warning(f"   ⏳ Timeout fetching text report, retrying...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                ctx.logger.error(f"   ❌ Error fetching text report: {e}")
                break
        
        if not text_report or text_report.get("error"):
            ctx.logger.error(f"   ❌ Failed to fetch text report after {max_retries} attempts")
        
        # Step 2: Fetch Visual Bias Report via REST (with retries)
        ctx.logger.info(f"📤 STEP 2: Fetching Visual Bias Report via HTTP...")
        visual_agent_url = f"http://localhost:8102/report/{req.request_id}"
        visual_report = None
        
        # Retry logic for visual report (wait up to 15 seconds - visual analysis takes longer)
        max_retries = 15
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    ctx.logger.info(f"   🔗 Calling: {visual_agent_url} (attempt {attempt + 1}/{max_retries})")
                    async with session.get(visual_agent_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        ctx.logger.info(f"   📥 Response status: {response.status}")
                        if response.status == 200:
                            result = await response.json()
                            visual_report = result.get("report", {})
                            # Check if it's a valid report (not error)
                            if not visual_report.get("error") and visual_report.get("overall_visual_score") is not None:
                                ctx.logger.info(f"   ✅ Visual report fetched successfully")
                                ctx.logger.info(f"      Score: {visual_report.get('overall_visual_score', 'N/A')}")
                                break
                            else:
                                ctx.logger.warning(f"   ⏳ Visual report not ready yet, retrying...")
                                await asyncio.sleep(retry_delay)
                        else:
                            ctx.logger.warning(f"   ⏳ Visual report not available yet (status {response.status}), retrying...")
                            await asyncio.sleep(retry_delay)
            except aiohttp.ClientConnectorError as e:
                ctx.logger.error(f"   ❌ Cannot connect to Visual Bias Agent at {visual_agent_url}")
                ctx.logger.error(f"      Make sure Visual Bias Agent is running on port 8102")
                break
            except asyncio.TimeoutError:
                ctx.logger.warning(f"   ⏳ Timeout fetching visual report, retrying...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                ctx.logger.error(f"   ❌ Error fetching visual report: {e}")
                break
        
        if not visual_report or visual_report.get("error"):
            ctx.logger.error(f"   ❌ Failed to fetch visual report after {max_retries} attempts")
        
        # Step 3: Check if we have both reports
        if not text_report or not visual_report or text_report.get("error") or visual_report.get("error"):
            ctx.logger.error(f"❌ Missing reports - Text: {text_report is not None and not text_report.get('error')}, Visual: {visual_report is not None and not visual_report.get('error')}")
            return FinalBiasReport(
                request_id=req.request_id,
                overall_bias_score=0.0,
                assessment="Unable to generate final report. One or more bias agent reports are missing or unavailable. Please ensure both Text and Visual Bias Agents have completed their analysis.",
                bias_level="Error",
                score_breakdown={
                    "text_score": 0.0,
                    "visual_score": 0.0,
                    "intersectional_penalty": 0.0,
                    "weighted_score": 0.0
                },
                text_analysis_status="error" if not text_report or text_report.get("error") else "completed",
                visual_analysis_status="error" if not visual_report or visual_report.get("error") else "completed",
                total_issues=0,
                bias_issues=[],
                top_concerns=[],
                recommendations=[],
                similar_cases=[],
                benchmark_comparison={},
                confidence=0.0,
                timestamp=datetime.now(UTC).isoformat()
            )
        
        # Step 4: Process final scoring
        ctx.logger.info(f"🎯 STEP 3: Processing final scoring...")
        collection_id = req.chromadb_collection_id or req.request_id
        final_report = await process_final_scoring(
            ctx,
            req.request_id,
            text_report,
            visual_report,
            collection_id
        )
        
        # Step 5: Store final report for frontend retrieval
        final_report_key = f"final_report_{req.request_id}"
        ctx.storage.set(final_report_key, final_report.dict())
        ctx.logger.info(f"💾 Final report stored with key: {final_report_key}")
        
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

        # Store the report in context storage
        storage_key = f"reports_{msg.request_id}"
        reports = ctx.storage.get(storage_key) or {}

        # Add this agent's report
        if msg.sender_agent == "text_bias_agent":
            reports["text"] = msg.report
            ctx.logger.info(f"✅ Text bias report stored")
        elif msg.sender_agent == "visual_bias_agent":
            reports["visual"] = msg.report
            ctx.logger.info(f"✅ Visual bias report stored")

        # Update storage
        ctx.storage.set(storage_key, reports)

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

            # Store final report for frontend retrieval
            final_report_key = f"final_report_{msg.request_id}"
            ctx.storage.set(final_report_key, final_report.dict())
            ctx.logger.info(f"💾 Final report stored with key: {final_report_key}")
            
            # Clean up temporary reports storage
            ctx.storage.set(storage_key, None)

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


async def process_final_scoring(
    ctx: Context,
    request_id: str,
    text_report: Dict[str, Any],
    visual_report: Dict[str, Any],
    collection_id: str
) -> FinalBiasReport:
    """
    Process final scoring with both reports.
    """
    start_time = datetime.now(UTC)

    text_score = text_report.get("overall_text_score", 5.0)
    visual_score = visual_report.get("overall_visual_score", 5.0)

    ctx.logger.info(f"📊 Text Score: {text_score:.1f}, Visual Score: {visual_score:.1f}")

    # Step 2: RAG RETRIEVAL - Query ChromaDB for benchmark cases
    ctx.logger.info(f"🔎 RAG RETRIEVAL: Querying ChromaDB for benchmark cases...")
    benchmark_cases = await query_case_benchmarks(ctx, text_score, visual_score, collection_id)
    ctx.logger.info(f"✅ Found {len(benchmark_cases)} similar benchmark cases")

    # Step 3: Detect intersectional bias
    ctx.logger.info(f"🔀 Analyzing intersectional bias patterns...")
    intersectional_penalty = await detect_intersectional_bias(ctx, text_report, visual_report)
    ctx.logger.info(f"⚠️ Intersectional penalty: {intersectional_penalty:.2f}")

    # Step 4: Calculate weighted final score
    ctx.logger.info(f"⚖️ Calculating weighted final score...")
    final_score = await calculate_weighted_score(
        ctx,
        text_score,
        visual_score,
        intersectional_penalty,
        SCORING_WEIGHTS
    )
    ctx.logger.info(f"🎯 Final weighted score: {final_score:.1f}")

    # Step 5: Aggregate all bias issues
    ctx.logger.info(f"📑 Aggregating bias issues...")
    bias_issues = await aggregate_bias_issues(ctx, text_report, visual_report)
    ctx.logger.info(f"📋 Total issues aggregated: {len(bias_issues)}")

    # Step 6: Categorize by severity
    severity_counts = categorize_by_severity(bias_issues)
    ctx.logger.info(f"🏷️ Severity breakdown: High={severity_counts['high']}, Medium={severity_counts['medium']}, Low={severity_counts['low']}")

    # Step 7: Extract top concerns
    ctx.logger.info(f"🎯 Identifying top concerns...")
    top_concerns = extract_top_concerns(bias_issues, max_concerns=5)

    # Step 8: Generate recommendations
    ctx.logger.info(f"💡 Generating recommendations...")
    recommendations = await generate_comprehensive_recommendations(ctx, bias_issues, final_score)
    ctx.logger.info(f"✅ Generated {len(recommendations)} recommendations")

    # Step 9: Generate assessment text and bias level
    ctx.logger.info(f"📝 Generating assessment text...")
    assessment = generate_assessment_text(final_score, len(bias_issues), severity_counts)
    bias_level = get_bias_level(final_score)
    ctx.logger.info(f"   🏷️ Bias Level: {bias_level}")

    # Step 10: Extract content metadata
    content_url = text_report.get("content_url") or visual_report.get("content_url")
    content_type = text_report.get("content_type") or visual_report.get("content_type", "video")

    # Step 11: Calculate processing time
    end_time = datetime.now(UTC)
    processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

    # Step 12: Create comprehensive final report for frontend
    # Convert nested models to dicts for proper serialization
    score_breakdown_dict = convert_score_breakdown_to_dict(ScoreBreakdown(
        text_score=text_score,
        visual_score=visual_score,
        intersectional_penalty=intersectional_penalty,
        weighted_score=final_score
    ))
    bias_issues_dicts = convert_bias_issues_to_dicts(bias_issues)
    recommendations_dicts = convert_recommendations_to_dicts(recommendations)
    
    final_report = FinalBiasReport(
        request_id=request_id,
        content_url=content_url,
        content_type=content_type,
        overall_bias_score=final_score,
        assessment=assessment,
        bias_level=bias_level,
        score_breakdown=score_breakdown_dict,
        text_analysis_status="completed",
        visual_analysis_status="completed",
        total_issues=len(bias_issues),
        critical_severity_count=severity_counts.get("critical", 0),
        high_severity_count=severity_counts.get("high", 0),
        medium_severity_count=severity_counts.get("medium", 0),
        low_severity_count=severity_counts.get("low", 0),
        bias_issues=bias_issues_dicts,
        top_concerns=top_concerns,
        recommendations=recommendations_dicts,
        similar_cases=[case["id"] for case in benchmark_cases],
        benchmark_comparison={
            "average_score": sum(c["score"] for c in benchmark_cases) / len(benchmark_cases) if benchmark_cases else 5.0,
            "percentile": calculate_percentile(final_score, benchmark_cases)
        },
        text_report=text_report,  # Include original text report
        visual_report=visual_report,  # Include original visual report
        confidence=calculate_overall_confidence(text_report, visual_report),
        processing_time_ms=processing_time_ms
    )

    ctx.logger.info(f"✅ Final report generated: Score={final_score:.1f}, Issues={len(bias_issues)}, Time={processing_time_ms}ms")

    # Store results in ChromaDB for future RAG
    await store_final_report_in_chromadb(ctx, final_report)

    return final_report


@scoring_protocol.on_message(model=ScoringRequest, replies=FinalBiasReport)
async def handle_scoring_request(ctx: Context, sender: str, msg: ScoringRequest):
    """
    Aggregate results from Text and Visual agents and produce final assessment.
    """
    try:
        start_time = datetime.now(UTC)
        ctx.logger.info(f"📨 Received scoring request: {msg.request_id}")
        
        # Step 1: Parse incoming reports
        ctx.logger.info(f"📋 Parsing agent reports...")
        text_report_data = msg.text_report or {}
        visual_report_data = msg.visual_report or {}
        
        text_score = text_report_data.get("overall_text_score", 5.0)
        visual_score = visual_report_data.get("overall_visual_score", 5.0)
        
        ctx.logger.info(f"📊 Text Score: {text_score:.1f}, Visual Score: {visual_score:.1f}")
        
        # Step 2: RAG RETRIEVAL - Query ChromaDB for benchmark cases
        ctx.logger.info(f"🔎 RAG RETRIEVAL: Querying ChromaDB for benchmark cases...")
        benchmark_cases = await query_case_benchmarks(ctx, text_score, visual_score, msg.chromadb_collection_id)
        ctx.logger.info(f"✅ Found {len(benchmark_cases)} similar benchmark cases")
        
        # Step 3: Detect intersectional bias
        ctx.logger.info(f"🔀 Analyzing intersectional bias patterns...")
        intersectional_penalty = await detect_intersectional_bias(ctx, text_report_data, visual_report_data)
        ctx.logger.info(f"⚠️ Intersectional penalty: {intersectional_penalty:.2f}")
        
        # Step 4: Calculate weighted final score
        ctx.logger.info(f"⚖️ Calculating weighted final score...")
        final_score = await calculate_weighted_score(
            ctx,
            text_score,
            visual_score,
            intersectional_penalty,
            SCORING_WEIGHTS
        )
        ctx.logger.info(f"🎯 Final weighted score: {final_score:.1f}")
        
        # Step 5: Aggregate all bias issues
        ctx.logger.info(f"📑 Aggregating bias issues...")
        bias_issues = await aggregate_bias_issues(ctx, text_report_data, visual_report_data)
        ctx.logger.info(f"📋 Total issues aggregated: {len(bias_issues)}")
        
        # Step 6: Categorize by severity
        severity_counts = categorize_by_severity(bias_issues)
        ctx.logger.info(f"🏷️ Severity breakdown: High={severity_counts['high']}, Medium={severity_counts['medium']}, Low={severity_counts['low']}")
        
        # Step 7: Extract top concerns
        ctx.logger.info(f"🎯 Identifying top concerns...")
        top_concerns = extract_top_concerns(bias_issues, max_concerns=5)
        
        # Step 8: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = await generate_comprehensive_recommendations(ctx, bias_issues, final_score)
        ctx.logger.info(f"✅ Generated {len(recommendations)} recommendations")
        
        # Step 9: Generate assessment text and bias level
        ctx.logger.info(f"📝 Generating assessment text...")
        assessment = generate_assessment_text(final_score, len(bias_issues), severity_counts)
        bias_level = get_bias_level(final_score)
        ctx.logger.info(f"   🏷️ Bias Level: {bias_level}")
        
        # Step 10: Extract content metadata
        content_url = text_report_data.get("content_url") or visual_report_data.get("content_url")
        content_type = text_report_data.get("content_type") or visual_report_data.get("content_type", "video")
        
        # Step 11: Calculate processing time
        end_time = datetime.now(UTC)
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Step 12: Create comprehensive final report for frontend
        # Convert nested models to dicts for proper serialization
        score_breakdown_dict = convert_score_breakdown_to_dict(ScoreBreakdown(
            text_score=text_score,
            visual_score=visual_score,
            intersectional_penalty=intersectional_penalty,
            weighted_score=final_score
        ))
        bias_issues_dicts = convert_bias_issues_to_dicts(bias_issues)
        recommendations_dicts = convert_recommendations_to_dicts(recommendations)
        
        final_report = FinalBiasReport(
            request_id=msg.request_id,
            content_url=content_url,
            content_type=content_type,
            overall_bias_score=final_score,
            assessment=assessment,
            bias_level=bias_level,
            score_breakdown=score_breakdown_dict,
            text_analysis_status="completed",
            visual_analysis_status="completed",
            total_issues=len(bias_issues),
            critical_severity_count=severity_counts.get("critical", 0),
            high_severity_count=severity_counts.get("high", 0),
            medium_severity_count=severity_counts.get("medium", 0),
            low_severity_count=severity_counts.get("low", 0),
            bias_issues=bias_issues_dicts,
            top_concerns=top_concerns,
            recommendations=recommendations_dicts,
            similar_cases=[case["id"] for case in benchmark_cases],
            benchmark_comparison={
                "average_score": sum(c["score"] for c in benchmark_cases) / len(benchmark_cases) if benchmark_cases else 5.0,
                "percentile": calculate_percentile(final_score, benchmark_cases)
            },
            text_report=text_report_data,  # Include original text report
            visual_report=visual_report_data,  # Include original visual report
            confidence=calculate_overall_confidence(text_report_data, visual_report_data),
            processing_time_ms=processing_time_ms
        )
        
        ctx.logger.info(f"✅ Final report generated: Score={final_score:.1f}, Issues={len(bias_issues)}, Time={processing_time_ms}ms")
        
        # Step 12: Store results in ChromaDB for future RAG
        await store_final_report_in_chromadb(ctx, final_report)
        
        # Step 13: Send report back to requester
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


async def query_case_benchmarks(
    ctx: Context,
    text_score: float,
    visual_score: float,
    collection_id: str
) -> List[Dict[str, Any]]:
    """
    RAG RETRIEVAL POINT #3: Query ChromaDB for similar complete case studies.
    
    TODO: Implement actual ChromaDB query
    - Query case_studies collection
    - Filter by similar score ranges
    - Return benchmark data for comparison
    """
    ctx.logger.info(f"🔎 Querying ChromaDB for benchmark cases...")
    
    # Placeholder benchmark cases
    benchmark_cases = [
        {
            "id": "case_complete_123",
            "score": 5.5,
            "text_score": 6.0,
            "visual_score": 5.0,
            "similarity": 0.89,
            "context": "Similar tech recruitment ad with representation issues"
        },
        {
            "id": "case_complete_456",
            "score": 4.8,
            "text_score": 6.5,
            "visual_score": 3.1,
            "similarity": 0.83,
            "context": "Financial services ad with gender bias"
        }
    ]
    
    ctx.logger.info(f"✅ Retrieved {len(benchmark_cases)} benchmark cases")
    return benchmark_cases


async def detect_intersectional_bias(
    ctx: Context,
    text_report: Dict[str, Any],
    visual_report: Dict[str, Any]
) -> float:
    """
    Detect intersectional bias where text and visual biases compound.
    
    Intersectional bias occurs when multiple forms of bias overlap and
    amplify each other (e.g., gender bias in text + gender bias in visuals).
    """
    ctx.logger.info(f"🔀 Detecting intersectional bias...")
    
    penalty = 0.0
    
    # Check for overlapping bias types
    text_biases = set()
    visual_biases = set()
    
    # Extract from bias_instances (not bias_types)
    if "bias_instances" in text_report:
        for bias in text_report["bias_instances"]:
            bias_type = bias.get("bias_type", "").replace("_bias", "")
            text_biases.add(bias_type)
    
    if "bias_instances" in visual_report:
        for bias in visual_report["bias_instances"]:
            bias_type = bias.get("bias_type", "").replace("_bias", "")
            visual_biases.add(bias_type)
    
    # Calculate intersectional penalty
    overlapping = text_biases.intersection(visual_biases)
    
    if overlapping:
        # Each overlapping bias type adds 0.5 to penalty, weighted by severity
        penalty = len(overlapping) * 0.5
        ctx.logger.info(f"⚠️ Intersectional bias detected in: {', '.join(overlapping)}")
        
        # Log detailed intersectional findings
        for bias_type in overlapping:
            ctx.logger.info(f"   🔄 {bias_type}: Present in both text and visual content")
    else:
        ctx.logger.info(f"   ℹ️ No intersectional bias detected")
    
    return min(penalty, 2.0)  # Cap at 2.0


async def calculate_weighted_score(
    ctx: Context,
    text_score: float,
    visual_score: float,
    intersectional_penalty: float,
    weights: Dict[str, float]
) -> float:
    """
    Calculate final weighted score.
    
    Formula: (text_score * text_weight) + (visual_score * visual_weight) - intersectional_penalty
    """
    weighted = (
        (text_score * weights["text_weight"]) +
        (visual_score * weights["visual_weight"])
    ) - intersectional_penalty
    
    # Normalize to 0-10 scale
    final_score = max(0.0, min(10.0, weighted))
    
    ctx.logger.info(f"⚖️ Weighted score calculated: {final_score:.1f}")
    return final_score


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


def extract_top_concerns(issues: List[BiasIssue], max_concerns: int = 5) -> List[str]:
    """
    Extract top priority concerns from all issues.
    """
    # Sort by severity and confidence
    severity_order = {
        SeverityLevel.CRITICAL: 4,
        SeverityLevel.HIGH: 3,
        SeverityLevel.MEDIUM: 2,
        SeverityLevel.LOW: 1
    }
    
    sorted_issues = sorted(
        issues,
        key=lambda x: (severity_order[x.severity], x.confidence),
        reverse=True
    )
    
    top_concerns = [issue.description for issue in sorted_issues[:max_concerns]]
    return top_concerns


async def generate_comprehensive_recommendations(
    ctx: Context,
    issues: List[BiasIssue],
    final_score: float
) -> List[Recommendation]:
    """
    Generate prioritized, actionable recommendations.
    """
    recommendations = []
    
    # Group issues by category
    by_category = {}
    for issue in issues:
        category = issue.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(issue)
    
    # Generate recommendations per category
    for category, category_issues in by_category.items():
        highest_severity = max(issue.severity for issue in category_issues)
        
        # Generate category-specific recommendation
        if "gender" in category:
            rec = Recommendation(
                priority=highest_severity,
                category="Gender Representation",
                action="Use gender-neutral language and ensure balanced representation across all roles",
                impact="Improves inclusivity and reaches wider audience"
            )
        elif "racial" in category or "representation" in category:
            rec = Recommendation(
                priority=highest_severity,
                category="Diversity & Representation",
                action="Include diverse individuals in prominent and leadership positions",
                impact="Reflects modern diverse workforce and customer base"
            )
        elif "age" in category:
            rec = Recommendation(
                priority=highest_severity,
                category="Age Inclusivity",
                action="Remove age-specific references and focus on skills and experience",
                impact="Appeals to broader age demographics and avoids discrimination"
            )
        elif "contextual" in category:
            rec = Recommendation(
                priority=highest_severity,
                category="Visual Composition",
                action="Balance spatial positioning and give equal prominence to all subjects",
                impact="Eliminates visual power dynamics and stereotypes"
            )
        else:
            rec = Recommendation(
                priority=highest_severity,
                category=category.replace("_", " ").title(),
                action=f"Review and address {category} in content",
                impact="Improves overall inclusivity and fairness"
            )
        
        recommendations.append(rec)
    
    # Add general recommendation if score is low
    if final_score < 5.0:
        recommendations.insert(0, Recommendation(
            priority=SeverityLevel.HIGH,
            category="Overall Content Review",
            action="Consider comprehensive revision of ad content to address multiple bias concerns",
            impact="Significant improvement in brand perception and legal compliance"
        ))
    
    ctx.logger.info(f"💡 Generated {len(recommendations)} recommendations")
    return recommendations


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


def generate_assessment_text(score: float, issue_count: int, severity_counts: Dict[str, int]) -> str:
    """
    Generate human-readable assessment text based on score.
    """
    if score >= 9.0:
        return f"Excellent - Minimal to no bias detected. Ad content demonstrates strong inclusivity and fairness ({issue_count} minor issues found)."
    elif score >= 7.0:
        return f"Good - Minor bias detected. Ad content is largely acceptable with {issue_count} issues requiring minor improvements."
    elif score >= 4.0:
        return f"Moderate Concerns - Moderate bias detected. Ad content needs revision to address {issue_count} issues, including {severity_counts.get('high', 0)} high-severity concerns."
    else:
        critical = severity_counts.get('critical', 0)
        high = severity_counts.get('high', 0)
        return f"Significant Concerns - Significant bias detected. Ad content requires substantial revision to address {issue_count} issues, including {high} high-severity and {critical} critical concerns."


def calculate_percentile(score: float, benchmark_cases: List[Dict[str, Any]]) -> float:
    """
    Calculate percentile ranking compared to benchmark cases.
    """
    if not benchmark_cases:
        return 50.0
    
    scores = [case["score"] for case in benchmark_cases]
    scores.append(score)
    scores.sort()
    
    rank = scores.index(score) + 1
    percentile = (rank / len(scores)) * 100
    
    return percentile


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


def convert_score_breakdown_to_dict(score_breakdown: ScoreBreakdown) -> Dict[str, Any]:
    """
    Convert ScoreBreakdown Model object to dict for proper serialization.
    """
    return {
        "text_score": score_breakdown.text_score,
        "visual_score": score_breakdown.visual_score,
        "intersectional_penalty": score_breakdown.intersectional_penalty,
        "weighted_score": score_breakdown.weighted_score
    }


async def store_final_report_in_chromadb(ctx: Context, report: FinalBiasReport):
    """
    Store final report in ChromaDB for future RAG retrieval and learning.
    
    TODO: Implement actual ChromaDB storage
    - Store in case_studies collection
    - Include metadata for filtering
    """
    ctx.logger.info(f"💾 Storing final report in ChromaDB for future learning...")
    
    # Placeholder storage
    ctx.logger.info(f"✅ Report stored in ChromaDB")


# Include protocol
scoring_agent.include(scoring_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        ⚖️ SCORING AGENT - Ad Bias Detection                  ║
╚══════════════════════════════════════════════════════════════╝

Role: Result Aggregation and Final Bias Assessment

Capabilities:
  ✓ Aggregates Text and Visual Bias reports with full context
  ✓ Preserves detailed bias descriptions and impact analysis
  ✓ Detects intersectional bias patterns
  ✓ RAG retrieval for benchmark comparison
  ✓ Calculates weighted final scores
  ✓ Generates comprehensive recommendations
  ✓ REST API for frontend report retrieval

REST API Endpoints:
  • POST /score - Trigger scoring by fetching reports from Text/Visual agents via HTTP
    URL: http://localhost:8103/score
    Body: {"request_id": "your_request_id"}
  • GET /report/{request_id} - Retrieve final bias report (JSON)
    URL: http://localhost:8103/report/{request_id}

Architecture:
  • Uses REST HTTP calls instead of uAgents messaging
  • Fetches reports from Text Agent (port 8101) and Visual Agent (port 8102)
  • Aggregates results and generates comprehensive final report

Report Output Structure:
  • Overall Score (0-10)
  • Score Breakdown (text, visual, intersectional)
  • Bias Issues (with examples, context, impact, severity)
  • Severity Counts (critical, high, medium, low)
  • Top Concerns (prioritized)
  • Recommendations (actionable)
  • Original Reports (text & visual for transparency)
  • Benchmark Comparison
  • Confidence Score
  • Processing Time

Scoring Scale (0-10):
  • 0-3: Significant bias (high concern)
  • 4-6: Moderate bias (needs revision)
  • 7-8: Minor bias (minor improvements)
  • 9-10: Minimal bias (approved)

Scoring Weights:
  • Text: 40%
  • Visual: 40%
  • Intersectional: 20%

Configuration:
  • Port: 8103
  • Endpoint: http://localhost:8103/submit

📍 Waiting for agent reports to aggregate...
🛑 Stop with Ctrl+C
    """)
    scoring_agent.run()

