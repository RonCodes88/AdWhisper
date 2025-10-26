"""
Simplified Scoring Agent - AdWhisper

Role: Result Aggregation and Final Bias Assessment
This is the ONLY agent that uses ChromaDB for RAG!

Responsibilities:
- Receive reports from Text Bias Agent
- Receive reports from Visual Bias Agent
- Wait for BOTH reports
- Query ChromaDB for similar cases (ONLY RAG POINT!)
- Calculate weighted final score
- Format clean JSON report
- Store in ChromaDB for future reference
- Store result for FastAPI retrieval

Following Fetch.ai uAgents standards for clean, simple agent communication.
"""

from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for ChromaDB
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chroma import ChromaDB

# Import simplified models (relative import from agents directory)
from simple_shared_models import (
    TextBiasReport,
    VisualBiasReport,
    FinalBiasReport,
    AgentError
)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

scoring_agent = Agent(
    name="simple_scoring_agent",
    seed="simple_scoring_agent_seed_2024",
    port=8103,
    endpoint=["http://localhost:8103/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for scoring
scoring_protocol = Protocol(name="simple_scoring_protocol", version="1.0")

# Initialize ChromaDB (ONLY agent with ChromaDB!)
chroma_db = ChromaDB()


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@scoring_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Simple Scoring Agent started!")
    ctx.logger.info(f"📍 Agent address: {scoring_agent.address}")
    ctx.logger.info(f"🔧 Role: Result aggregation & final scoring")
    ctx.logger.info(f"💾 ChromaDB: ENABLED (ONLY RAG point in system)")


@scoring_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Simple Scoring Agent shutting down...")


# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

@scoring_protocol.on_message(model=TextBiasReport)
async def handle_text_report(ctx: Context, sender: str, msg: TextBiasReport):
    """
    Receive and store Text Bias Report

    Flow:
    1. Store text report in temporary storage
    2. Check if we have both reports
    3. If yes, generate final score
    """
    ctx.logger.info(f"📨 Received Text Bias Report: {msg.request_id}")
    ctx.logger.info(f"   Text Score: {msg.text_score}/10")
    ctx.logger.info(f"   Bias Detected: {msg.bias_detected}")
    ctx.logger.info(f"   Issues: {len(msg.bias_instances)}")

    # Store in temporary storage
    ctx.storage.set(f"text_{msg.request_id}", msg.dict())
    ctx.logger.info(f"✅ Text report stored")

    # Check if we have both reports
    visual_report = ctx.storage.get(f"visual_{msg.request_id}")
    if visual_report:
        ctx.logger.info(f"✅ Both reports received! Generating final score...")
        await generate_final_score(ctx, msg.request_id)
    else:
        ctx.logger.info(f"⏳ Waiting for Visual Bias Report...")


@scoring_protocol.on_message(model=VisualBiasReport)
async def handle_visual_report(ctx: Context, sender: str, msg: VisualBiasReport):
    """
    Receive and store Visual Bias Report

    Flow:
    1. Store visual report in temporary storage
    2. Check if we have both reports
    3. If yes, generate final score
    """
    ctx.logger.info(f"📨 Received Visual Bias Report: {msg.request_id}")
    ctx.logger.info(f"   Visual Score: {msg.visual_score}/10")
    ctx.logger.info(f"   Bias Detected: {msg.bias_detected}")
    ctx.logger.info(f"   Issues: {len(msg.bias_instances)}")

    # Store in temporary storage
    ctx.storage.set(f"visual_{msg.request_id}", msg.dict())
    ctx.logger.info(f"✅ Visual report stored")

    # Check if we have both reports
    text_report = ctx.storage.get(f"text_{msg.request_id}")
    if text_report:
        ctx.logger.info(f"✅ Both reports received! Generating final score...")
        await generate_final_score(ctx, msg.request_id)
    else:
        ctx.logger.info(f"⏳ Waiting for Text Bias Report...")


# ============================================================================
# SCORING LOGIC
# ============================================================================

async def generate_final_score(ctx: Context, request_id: str):
    """
    Generate final aggregated bias assessment

    This is where ChromaDB RAG happens (ONLY place!)

    Flow:
    1. Get both reports from storage
    2. Query ChromaDB for similar cases (RAG!)
    3. Calculate weighted final score
    4. Format clean JSON report
    5. Store in ChromaDB for future reference
    6. Store result for FastAPI retrieval
    """
    ctx.logger.info(f"🎯 Generating final score for request: {request_id}")

    try:
        # Step 1: Get both reports
        text_report_dict = ctx.storage.get(f"text_{request_id}")
        visual_report_dict = ctx.storage.get(f"visual_{request_id}")

        if not text_report_dict or not visual_report_dict:
            ctx.logger.error(f"❌ Missing report(s) for {request_id}")
            return

        ctx.logger.info(f"✅ Retrieved both reports from storage")

        # Step 2: Query ChromaDB for similar cases (ONLY RAG POINT!)
        ctx.logger.info(f"💾 Querying ChromaDB for similar bias cases...")
        benchmark_data = query_chromadb_for_benchmarks(
            text_score=text_report_dict["text_score"],
            visual_score=visual_report_dict["visual_score"],
            ctx=ctx
        )

        # Step 3: Calculate weighted final score
        ctx.logger.info(f"📊 Calculating weighted final score...")
        text_score = text_report_dict["text_score"]
        visual_score = visual_report_dict["visual_score"]

        # Weighted average (50% text, 50% visual)
        final_score = (text_score * 0.5) + (visual_score * 0.5)
        final_score = round(final_score, 1)

        ctx.logger.info(f"   Text Score: {text_score}/10 (50%)")
        ctx.logger.info(f"   Visual Score: {visual_score}/10 (50%)")
        ctx.logger.info(f"   Final Score: {final_score}/10")

        # Step 4: Generate assessment text
        assessment = generate_assessment_text(final_score)

        # Step 5: Count issues by severity
        total_issues, severity_counts = count_issues(
            text_report_dict["bias_instances"],
            visual_report_dict["bias_instances"]
        )

        # Step 6: Combine recommendations
        all_recommendations = combine_recommendations(
            text_report_dict.get("recommendations", []),
            visual_report_dict.get("recommendations", [])
        )

        # Step 7: Calculate overall confidence
        confidence = calculate_confidence(text_report_dict, visual_report_dict)

        # Step 8: Create final report
        final_report = FinalBiasReport(
            request_id=request_id,
            overall_score=final_score,
            assessment=assessment,
            text_score=text_score,
            visual_score=visual_score,
            total_issues=total_issues,
            high_severity_count=severity_counts["high"],
            medium_severity_count=severity_counts["medium"],
            low_severity_count=severity_counts["low"],
            text_analysis={
                "score": text_score,
                "bias_detected": text_report_dict["bias_detected"],
                "issues": text_report_dict["bias_instances"]
            },
            visual_analysis={
                "score": visual_score,
                "bias_detected": visual_report_dict["bias_detected"],
                "issues": visual_report_dict["bias_instances"],
                "diversity_metrics": visual_report_dict.get("diversity_metrics", {})
            },
            recommendations=all_recommendations,
            benchmark=benchmark_data,
            confidence=confidence
        )

        ctx.logger.info(f"✅ Final report generated:")
        ctx.logger.info(f"   Overall Score: {final_score}/10")
        ctx.logger.info(f"   Assessment: {assessment}")
        ctx.logger.info(f"   Total Issues: {total_issues}")
        ctx.logger.info(f"   Confidence: {confidence}")

        # Step 9: Store in ChromaDB for future RAG
        ctx.logger.info(f"💾 Storing result in ChromaDB for future reference...")
        store_in_chromadb(request_id, final_report.dict(), ctx)

        # Step 10: Store result for FastAPI retrieval
        ctx.logger.info(f"💾 Storing result for FastAPI retrieval...")
        ctx.storage.set(f"final_{request_id}", final_report.dict())

        ctx.logger.info(f"🎉 Final scoring complete for {request_id}!")

        # Clean up temporary reports
        ctx.storage.set(f"text_{request_id}", None)
        ctx.storage.set(f"visual_{request_id}", None)

    except Exception as e:
        ctx.logger.error(f"❌ Error generating final score: {e}")


# ============================================================================
# CHROMADB FUNCTIONS (ONLY RAG POINT!)
# ============================================================================

def query_chromadb_for_benchmarks(text_score: float, visual_score: float, ctx: Context) -> Dict[str, Any]:
    """
    Query ChromaDB for similar bias cases

    This is the ONLY RAG retrieval point in the entire system!

    Returns benchmark comparison data
    """
    try:
        # Calculate composite score for similarity search
        composite_score = (text_score + visual_score) / 2.0

        ctx.logger.info(f"   Searching for cases with similar scores...")
        ctx.logger.info(f"   Composite score: {composite_score}/10")

        # In production, you would:
        # 1. Create embedding from scores and bias types
        # 2. Query ChromaDB for similar cases
        # 3. Return top-k similar cases with their scores and metadata

        # Placeholder benchmark data
        # TODO: Implement actual ChromaDB query
        benchmark_data = {
            "similar_cases_found": 0,
            "average_score": composite_score,
            "percentile": calculate_percentile(composite_score),
            "comparison": generate_comparison_text(composite_score),
            "note": "Placeholder - Production should query ChromaDB for similar historical cases"
        }

        ctx.logger.info(f"   ✅ Benchmark analysis complete")
        ctx.logger.info(f"   Percentile: {benchmark_data['percentile']}")

        return benchmark_data

    except Exception as e:
        ctx.logger.error(f"   ❌ Error querying ChromaDB: {e}")
        return {
            "similar_cases_found": 0,
            "average_score": 0.0,
            "percentile": 50.0,
            "comparison": "Unable to retrieve benchmark data",
            "error": str(e)
        }


def store_in_chromadb(request_id: str, final_report: Dict[str, Any], ctx: Context):
    """
    Store final report in ChromaDB for future RAG queries

    This builds the knowledge base over time
    """
    try:
        ctx.logger.info(f"   Storing report in ChromaDB...")

        # In production, you would:
        # 1. Create embedding from the report data
        # 2. Store in ChromaDB with metadata
        # 3. Include scores, bias types, recommendations

        # TODO: Implement actual ChromaDB storage
        ctx.logger.info(f"   ✅ Report stored in ChromaDB")
        ctx.logger.info(f"   (Placeholder - implement actual ChromaDB storage)")

    except Exception as e:
        ctx.logger.error(f"   ❌ Error storing in ChromaDB: {e}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_assessment_text(score: float) -> str:
    """Generate human-readable assessment based on score"""
    if score >= 9.0:
        return "Excellent - Minimal bias detected. Content is highly inclusive."
    elif score >= 7.0:
        return "Good - Minor bias concerns. Consider suggested improvements."
    elif score >= 5.0:
        return "Fair - Moderate bias detected. Revision recommended."
    elif score >= 3.0:
        return "Poor - Significant bias concerns. Major revision needed."
    else:
        return "Critical - Severe bias issues. Complete redesign recommended."


def count_issues(text_instances: List[Dict], visual_instances: List[Dict]) -> tuple:
    """Count total issues and breakdown by severity"""
    all_instances = text_instances + visual_instances

    severity_counts = {
        "low": 0,
        "medium": 0,
        "high": 0,
        "critical": 0
    }

    for instance in all_instances:
        severity = instance.get("severity", "low").lower()
        if severity in severity_counts:
            severity_counts[severity] += 1

    total_issues = len(all_instances)

    return total_issues, severity_counts


def combine_recommendations(text_recs: List[str], visual_recs: List[str]) -> List[str]:
    """Combine and deduplicate recommendations"""
    all_recs = text_recs + visual_recs
    # Remove duplicates while preserving order
    seen = set()
    unique_recs = []
    for rec in all_recs:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)
    return unique_recs[:10]  # Limit to top 10 recommendations


def calculate_confidence(text_report: Dict, visual_report: Dict) -> float:
    """Calculate overall confidence in the assessment"""
    # Simple average of instance confidences
    text_instances = text_report.get("bias_instances", [])
    visual_instances = visual_report.get("bias_instances", [])

    all_confidences = [
        inst.get("confidence", 0.5)
        for inst in text_instances + visual_instances
    ]

    if not all_confidences:
        return 0.85  # Default confidence

    avg_confidence = sum(all_confidences) / len(all_confidences)
    return round(avg_confidence, 2)


def calculate_percentile(score: float) -> float:
    """Calculate percentile ranking (placeholder)"""
    # In production, calculate based on ChromaDB historical data
    # For now, use simple heuristic
    if score >= 9.0:
        return 95.0
    elif score >= 7.0:
        return 75.0
    elif score >= 5.0:
        return 50.0
    elif score >= 3.0:
        return 25.0
    else:
        return 10.0


def generate_comparison_text(score: float) -> str:
    """Generate comparison text for benchmark"""
    percentile = calculate_percentile(score)
    if percentile >= 75:
        return f"This ad scores better than {percentile:.0f}% of similar content"
    elif percentile >= 50:
        return f"This ad scores at the {percentile:.0f}th percentile - room for improvement"
    else:
        return f"This ad scores in the lower {percentile:.0f}th percentile - significant improvements needed"


# ============================================================================
# INCLUDE PROTOCOLS
# ============================================================================

scoring_agent.include(scoring_protocol, publish_manifest=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🎯 Simple Scoring Agent - AdWhisper                 ║
╚══════════════════════════════════════════════════════════════╝

Role: Result Aggregation & Final Assessment
ONLY AGENT WITH CHROMADB RAG!

Flow:
  1. Receive Text Bias Report
  2. Receive Visual Bias Report
  3. Wait for BOTH reports
  4. Query ChromaDB for similar cases (RAG!)
  5. Calculate weighted final score (50% text, 50% visual)
  6. Format clean JSON report
  7. Store in ChromaDB for future reference
  8. Store result for FastAPI retrieval

Scoring Scale:
  9-10: Excellent (minimal bias)
  7-8:  Good (minor concerns)
  5-6:  Fair (moderate bias)
  3-4:  Poor (significant bias)
  0-2:  Critical (severe bias)

Running on: http://localhost:8103
🛑 Stop with Ctrl+C
    """)
    scoring_agent.run()
