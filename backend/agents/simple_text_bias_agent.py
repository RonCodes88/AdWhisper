"""
Simplified Text Bias Agent - AdWhisper

Role: Text Content Bias Analysis (NO ChromaDB)
Responsibilities:
- Receive text from Ingestion Agent
- Analyze for bias patterns
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
    TextAnalysisRequest,
    TextBiasReport,
    BiasType,
    SeverityLevel,
    create_bias_instance_dict,
    AgentError
)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

text_bias_agent = Agent(
    name="simple_text_bias_agent",
    seed="simple_text_bias_agent_seed_2024",
    port=8101,
    endpoint=["http://localhost:8101/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for text bias analysis
text_bias_protocol = Protocol(name="simple_text_bias_protocol", version="1.0")

# Scoring Agent address
SCORING_AGENT_ADDRESS = os.getenv("SCORING_AGENT_ADDRESS", "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

@text_bias_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Simple Text Bias Agent started!")
    ctx.logger.info(f"📍 Agent address: {text_bias_agent.address}")
    ctx.logger.info(f"🔧 Role: Text content bias analysis")
    ctx.logger.info(f"📤 Scoring Agent: {SCORING_AGENT_ADDRESS}")


@text_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Simple Text Bias Agent shutting down...")


# ============================================================================
# MESSAGE HANDLER
# ============================================================================

@text_bias_protocol.on_message(model=TextAnalysisRequest, replies=TextBiasReport)
async def handle_text_analysis(ctx: Context, sender: str, msg: TextAnalysisRequest):
    """
    Analyze text for bias - Simple pattern matching approach

    Flow:
    1. Receive text from Ingestion Agent
    2. Analyze for bias patterns (NO ChromaDB)
    3. Create JSON report
    4. Send to Scoring Agent
    """
    ctx.logger.info(f"📨 Received text analysis request: {msg.request_id}")
    ctx.logger.info(f"📝 Text length: {len(msg.text_content)} characters")

    try:
        # Step 1: Analyze text for bias patterns
        ctx.logger.info(f"🔍 Analyzing text for bias patterns...")
        bias_instances = analyze_text_for_bias(msg.text_content, ctx)

        # Step 2: Calculate text score
        ctx.logger.info(f"📊 Calculating text bias score...")
        text_score = calculate_text_score(bias_instances)

        # Step 3: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = generate_recommendations(bias_instances)

        # Step 4: Create report
        report = TextBiasReport(
            request_id=msg.request_id,
            agent_name="simple_text_bias_agent",
            bias_detected=len(bias_instances) > 0,
            bias_instances=bias_instances,
            text_score=text_score,
            recommendations=recommendations
        )

        ctx.logger.info(f"✅ Analysis complete:")
        ctx.logger.info(f"   Score: {text_score:.1f}/10")
        ctx.logger.info(f"   Issues found: {len(bias_instances)}")
        ctx.logger.info(f"   Bias detected: {report.bias_detected}")

        # Step 5: Send report to Scoring Agent
        ctx.logger.info(f"📤 Sending report to Scoring Agent...")
        await ctx.send(SCORING_AGENT_ADDRESS, report)
        ctx.logger.info(f"✅ Report sent successfully")

    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing text: {e}")

        # Send error to Scoring Agent
        error_msg = AgentError(
            request_id=msg.request_id,
            agent_name="simple_text_bias_agent",
            error_type="analysis_error",
            error_message=str(e)
        )
        await ctx.send(SCORING_AGENT_ADDRESS, error_msg)


# ============================================================================
# BIAS ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text_for_bias(text: str, ctx: Context) -> List[Dict[str, Any]]:
    """
    Simple pattern-based bias detection (NO ChromaDB, NO LLM)

    Detects common bias patterns through keyword matching
    """
    bias_instances = []
    text_lower = text.lower()

    # Gender bias keywords
    gender_bias_keywords = [
        "guys", "brotherhood", "rockstar", "ninja", "guru", "manpower",
        "chairman", "policeman", "fireman", "mankind"
    ]
    gender_examples = [word for word in gender_bias_keywords if word in text_lower]

    if gender_examples:
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.GENDER.value,
            severity=SeverityLevel.MEDIUM.value,
            examples=gender_examples[:3],  # Limit to 3 examples
            context="Uses male-default or gendered language that may exclude other genders",
            confidence=0.85
        ))
        ctx.logger.info(f"   🚨 Gender bias detected: {len(gender_examples)} instances")

    # Age bias keywords
    age_bias_keywords = [
        "young", "energetic", "digital native", "recent graduate",
        "new blood", "fresh perspective", "entry level"
    ]
    age_examples = [word for word in age_bias_keywords if word in text_lower]

    if age_examples:
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.AGE.value,
            severity=SeverityLevel.MEDIUM.value,
            examples=age_examples[:3],
            context="Language suggests preference for younger candidates or excludes older workers",
            confidence=0.80
        ))
        ctx.logger.info(f"   🚨 Age bias detected: {len(age_examples)} instances")

    # Socioeconomic bias keywords
    socio_bias_keywords = [
        "ivy league", "prestigious", "elite", "top tier",
        "world class", "privileged"
    ]
    socio_examples = [word for word in socio_bias_keywords if word in text_lower]

    if socio_examples:
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.SOCIOECONOMIC.value,
            severity=SeverityLevel.LOW.value,
            examples=socio_examples[:3],
            context="Language may favor candidates from privileged backgrounds",
            confidence=0.70
        ))
        ctx.logger.info(f"   🚨 Socioeconomic bias detected: {len(socio_examples)} instances")

    # Disability bias (ableist language)
    disability_bias_keywords = [
        "stand up", "walk in", "see the vision", "hear our call",
        "blind to", "deaf to", "crippled by"
    ]
    disability_examples = [word for word in disability_bias_keywords if word in text_lower]

    if disability_examples:
        bias_instances.append(create_bias_instance_dict(
            bias_type=BiasType.DISABILITY.value,
            severity=SeverityLevel.MEDIUM.value,
            examples=disability_examples[:3],
            context="Uses ableist language that assumes physical abilities",
            confidence=0.75
        ))
        ctx.logger.info(f"   🚨 Disability bias detected: {len(disability_examples)} instances")

    if not bias_instances:
        ctx.logger.info(f"   ✅ No obvious bias patterns detected")

    return bias_instances


def calculate_text_score(bias_instances: List[Dict[str, Any]]) -> float:
    """
    Calculate overall text bias score (0-10 scale)

    0-3: Significant bias (high concern)
    4-6: Moderate bias (needs revision)
    7-8: Minor bias (minor improvements)
    9-10: Minimal bias (approved)
    """
    if not bias_instances:
        return 9.5  # No bias detected

    # Calculate weighted penalties based on severity
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


def generate_recommendations(bias_instances: List[Dict[str, Any]]) -> List[str]:
    """Generate actionable recommendations based on detected biases"""
    recommendations = []

    for instance in bias_instances:
        bias_type = instance.get("bias_type", "")

        if bias_type == BiasType.GENDER.value:
            recommendations.append("Use gender-neutral language (e.g., 'team members' instead of 'guys', 'workforce' instead of 'manpower')")
        elif bias_type == BiasType.AGE.value:
            recommendations.append("Remove age-specific references and focus on skills and experience rather than age indicators")
        elif bias_type == BiasType.SOCIOECONOMIC.value:
            recommendations.append("Avoid language that assumes specific educational or economic backgrounds")
        elif bias_type == BiasType.DISABILITY.value:
            recommendations.append("Replace ableist language with inclusive alternatives that don't assume physical abilities")

    # Add general recommendation if multiple biases detected
    if len(bias_instances) > 2:
        recommendations.insert(0, "Consider comprehensive review of language to address multiple bias concerns")

    return list(set(recommendations))  # Remove duplicates


# ============================================================================
# INCLUDE PROTOCOLS
# ============================================================================

text_bias_agent.include(text_bias_protocol, publish_manifest=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🧠 Simple Text Bias Agent - AdWhisper               ║
╚══════════════════════════════════════════════════════════════╝

Role: Text Content Bias Analysis (NO ChromaDB)

Flow:
  1. Receive text from Ingestion Agent
  2. Analyze for bias patterns (keyword matching)
  3. Generate JSON report with score
  4. Send report to Scoring Agent

Bias Types Detected:
  • Gender bias (gendered language)
  • Age bias (ageism indicators)
  • Socioeconomic bias (class assumptions)
  • Disability bias (ableist language)

Running on: http://localhost:8101
🛑 Stop with Ctrl+C
    """)
    text_bias_agent.run()
