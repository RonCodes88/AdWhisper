"""
Simplified Visual Bias Agent - AdWhisper

Role: Visual Content Bias Analysis (NO ChromaDB)
Responsibilities:
- Receive frames from Ingestion Agent
- Analyze for visual bias patterns
- Generate diversity metrics
- Generate JSON report
- Send report to Scoring Agent

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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import simplified models (relative import from agents directory)
from simple_shared_models import (
    VisualAnalysisRequest,
    VisualBiasReport,
    BiasType,
    SeverityLevel,
    create_bias_instance_dict,
    AgentError
)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

visual_bias_agent = Agent(
    name="simple_visual_bias_agent",
    seed="simple_visual_bias_agent_seed_2024",
    port=8102,
    endpoint=["http://localhost:8102/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for visual bias analysis
visual_bias_protocol = Protocol(name="simple_visual_bias_protocol", version="1.0")

# Scoring Agent address
SCORING_AGENT_ADDRESS = os.getenv("SCORING_AGENT_ADDRESS", "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@visual_bias_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Simple Visual Bias Agent started!")
    ctx.logger.info(f"📍 Agent address: {visual_bias_agent.address}")
    ctx.logger.info(f"🔧 Role: Visual content bias analysis")
    ctx.logger.info(f"📤 Scoring Agent: {SCORING_AGENT_ADDRESS}")


@visual_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Simple Visual Bias Agent shutting down...")


# ============================================================================
# MESSAGE HANDLER
# ============================================================================

@visual_bias_protocol.on_message(model=VisualAnalysisRequest, replies=VisualBiasReport)
async def handle_visual_analysis(ctx: Context, sender: str, msg: VisualAnalysisRequest):
    """
    Analyze video frames for visual bias

    Flow:
    1. Receive frames from Ingestion Agent
    2. Analyze for visual bias patterns (NO ChromaDB)
    3. Generate diversity metrics
    4. Create JSON report
    5. Send to Scoring Agent

    Note: This is a simplified implementation. In production, you would use:
    - Computer vision models for face/object detection
    - Demographic classification models
    - Scene context analysis
    - Claude API for visual analysis
    """
    ctx.logger.info(f"📨 Received visual analysis request: {msg.request_id}")
    ctx.logger.info(f"🎬 Frames to analyze: {msg.num_frames}")

    try:
        # Step 1: Analyze frames for visual bias
        ctx.logger.info(f"🔍 Analyzing {msg.num_frames} frames for visual bias...")
        bias_instances = analyze_frames_for_bias(msg.frames_base64, msg.num_frames, ctx)

        # Step 2: Generate diversity metrics
        ctx.logger.info(f"📊 Generating diversity metrics...")
        diversity_metrics = generate_diversity_metrics(msg.num_frames)

        # Step 3: Calculate visual score
        ctx.logger.info(f"📊 Calculating visual bias score...")
        visual_score = calculate_visual_score(bias_instances, diversity_metrics)

        # Step 4: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = generate_visual_recommendations(bias_instances, diversity_metrics)

        # Step 5: Create report
        report = VisualBiasReport(
            request_id=msg.request_id,
            agent_name="simple_visual_bias_agent",
            bias_detected=len(bias_instances) > 0,
            bias_instances=bias_instances,
            visual_score=visual_score,
            diversity_metrics=diversity_metrics,
            recommendations=recommendations
        )

        ctx.logger.info(f"✅ Analysis complete:")
        ctx.logger.info(f"   Score: {visual_score:.1f}/10")
        ctx.logger.info(f"   Issues found: {len(bias_instances)}")
        ctx.logger.info(f"   Bias detected: {report.bias_detected}")

        # Step 6: Send report to Scoring Agent
        ctx.logger.info(f"📤 Sending report to Scoring Agent...")
        await ctx.send(SCORING_AGENT_ADDRESS, report)
        ctx.logger.info(f"✅ Report sent successfully")

    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing visual content: {e}")

        # Send error to Scoring Agent
        error_msg = AgentError(
            request_id=msg.request_id,
            agent_name="simple_visual_bias_agent",
            error_type="analysis_error",
            error_message=str(e)
        )
        await ctx.send(SCORING_AGENT_ADDRESS, error_msg)


# ============================================================================
# VISUAL BIAS ANALYSIS FUNCTIONS
# ============================================================================

def analyze_frames_for_bias(frames_base64: List[str], num_frames: int, ctx: Context) -> List[Dict[str, Any]]:
    """
    Simplified visual bias detection

    In production, this would use:
    - Computer vision models (face detection, object detection)
    - Demographic classification
    - Scene context analysis with Claude Vision API
    - Representation analysis

    For now, we'll use heuristic-based detection as a placeholder
    """
    bias_instances = []

    # Heuristic 1: Check if there are enough frames for analysis
    if num_frames < 3:
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.REPRESENTATION.value,
            severity=SeverityLevel.MEDIUM.value,
            examples=["Limited visual content"],
            context="Insufficient frames for comprehensive visual bias analysis",
            confidence=0.60
        ))
        ctx.logger.info(f"   ⚠️ Limited frames detected: {num_frames} frames")

    # Heuristic 2: Placeholder for actual visual analysis
    # In production, you would:
    # - Use Claude Vision API to analyze each frame
    # - Detect people, their demographics, positioning
    # - Analyze power dynamics, spatial positioning
    # - Check for stereotypical representations

    # Example placeholder findings
    if num_frames >= 5:
        # Simulate representation bias detection
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.REPRESENTATION.value,
            severity=SeverityLevel.LOW.value,
            examples=["Visual analysis placeholder - In production, use ML models"],
            context="This is a placeholder. Production version would use computer vision models to detect representation bias in demographics, positioning, and context",
            confidence=0.50
        ))
        ctx.logger.info(f"   ℹ️ Placeholder analysis - production would use CV models")

    if not bias_instances:
        ctx.logger.info(f"   ✅ No obvious visual bias patterns detected")

    return bias_instances


def generate_diversity_metrics(num_frames: int) -> Dict[str, Any]:
    """
    Generate diversity metrics from visual analysis

    In production, this would include:
    - Gender distribution (male/female/non-binary representation)
    - Age distribution (young/middle-aged/senior)
    - Ethnicity distribution
    - Role distribution (leadership vs support roles)
    - Spatial positioning analysis

    For now, returns placeholder metrics
    """
    # Placeholder metrics
    # In production, use computer vision models to extract these
    metrics = {
        "total_frames_analyzed": num_frames,
        "people_detected": 0,  # Would come from face detection
        "gender_distribution": {
            "analysis_note": "Placeholder - production would use demographic classification models",
            "male": 0.0,
            "female": 0.0,
            "non_binary": 0.0
        },
        "age_distribution": {
            "analysis_note": "Placeholder - production would use age estimation models",
            "young": 0.0,
            "middle_aged": 0.0,
            "senior": 0.0
        },
        "ethnicity_distribution": {
            "analysis_note": "Placeholder - production would use ethnicity classification models",
            "diverse": 0.0,
            "homogeneous": 0.0
        },
        "role_context": {
            "analysis_note": "Placeholder - production would use scene understanding models",
            "leadership_roles": 0,
            "support_roles": 0
        }
    }

    return metrics


def calculate_visual_score(bias_instances: List[Dict[str, Any]], diversity_metrics: Dict[str, Any]) -> float:
    """
    Calculate overall visual bias score (0-10 scale)

    0-3: Significant visual bias
    4-6: Moderate visual bias
    7-8: Minor visual bias
    9-10: Minimal visual bias
    """
    if not bias_instances:
        return 8.5  # Default good score when no bias detected

    # Calculate weighted penalties
    severity_weights = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "critical": 2.0
    }

    total_penalty = sum(
        severity_weights.get(instance.get("severity", "low"), 0.5) *
        instance.get("confidence", 0.5)
        for instance in bias_instances
    )

    # Start from 10 and subtract penalties
    score = max(0.0, 10.0 - total_penalty)

    return round(score, 1)


def generate_visual_recommendations(bias_instances: List[Dict[str, Any]], diversity_metrics: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations for visual content"""
    recommendations = []

    for instance in bias_instances:
        bias_type = instance.get("bias_type", "")

        if bias_type == BiasType.REPRESENTATION.value:
            recommendations.append("Ensure diverse representation across demographics (age, gender, ethnicity)")
            recommendations.append("Show people from various backgrounds in empowered positions")
        elif bias_type == BiasType.CONTEXTUAL.value:
            recommendations.append("Review power dynamics and spatial positioning in visual content")
            recommendations.append("Avoid stereotypical representations or contexts")

    # Add general visual recommendations
    recommendations.append("Production version should use Claude Vision API for comprehensive analysis")
    recommendations.append("Consider implementing computer vision models for demographic analysis")

    return list(set(recommendations))  # Remove duplicates


# ============================================================================
# INCLUDE PROTOCOLS
# ============================================================================

visual_bias_agent.include(visual_bias_protocol, publish_manifest=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          👁️ Simple Visual Bias Agent - AdWhisper            ║
╚══════════════════════════════════════════════════════════════╝

Role: Visual Content Bias Analysis (NO ChromaDB)

Flow:
  1. Receive frames from Ingestion Agent
  2. Analyze for visual bias (placeholder heuristics)
  3. Generate diversity metrics
  4. Create JSON report with score
  5. Send report to Scoring Agent

Visual Bias Types:
  • Representation bias (demographic diversity)
  • Contextual bias (power dynamics, positioning)

Note: This version uses placeholder analysis.
Production should use:
  - Claude Vision API for frame analysis
  - Computer vision models for demographics
  - Scene understanding for context

Running on: http://localhost:8102
🛑 Stop with Ctrl+C
    """)
    visual_bias_agent.run()
