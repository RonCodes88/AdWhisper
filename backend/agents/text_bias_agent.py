"""
Text Bias Agent - Ad Bias Detection System

Role: Text Content Analysis and Bias Detection
Responsibilities:
- Analyze textual content for bias indicators
- Query ChromaDB for similar historical cases (RAG RETRIEVAL POINT #1)
- Identify specific bias types (gender, racial, age, socioeconomic, disability, LGBTQ+)
- Extract problematic phrases and provide contextual explanations
- Generate structured findings with confidence scores
"""

from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from enum import Enum
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import ChromaDB
from chroma import ChromaDB

# Import shared models
from agents.shared_models import (
    EmbeddingPackage,
    TextBiasReport,
    BiasAnalysisComplete,
    BiasCategory,
    create_bias_instance_dict,
    AgentError
)


# Severity levels
class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Initialize Text Bias Agent
text_bias_agent = Agent(
    name="text_bias_agent",
    seed="ad_bias_text_agent_unique_seed_2024",
    port=8101,
    endpoint=["http://localhost:8101/submit"],
    mailbox=True  # Enable for Agentverse integration
)

# Protocol for text bias analysis
text_bias_protocol = Protocol(name="text_bias_protocol", version="1.0")

# ASI:ONE LLM agent addresses (rate limited to 6 requests/hour)
OPENAI_AGENT = 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y'
CLAUDE_AGENT = 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx'

# Scoring Agent address (to send results to)
SCORING_AGENT_ADDRESS = "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8"


@text_bias_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Text Bias Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {text_bias_agent.address}")
    ctx.logger.info(f"🔧 Role: Text Content Analysis and Bias Detection")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8101/submit")
    ctx.logger.info(f"🧠 ASI:ONE Integration: Ready for LLM-powered analysis")
    ctx.logger.info(f"⚡ Ready to analyze text content for bias")


@text_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Text Bias Agent shutting down...")


@text_bias_protocol.on_message(model=EmbeddingPackage, replies={BiasAnalysisComplete, AgentError})
async def handle_text_analysis(ctx: Context, sender: str, msg: EmbeddingPackage):
    """
    Analyze text content for bias using ASI:ONE LLM and RAG retrieval.
    """
    try:
        ctx.logger.info(f"📨 Received content for text analysis: {msg.request_id}")
        ctx.logger.info(f"   - Has text content: {msg.text_content is not None}")
        ctx.logger.info(f"   - Has text embedding: {msg.text_embedding is not None}")

        # Check if we have text to analyze
        if not msg.text_content:
            ctx.logger.warning(f"⚠️ No text content for request {msg.request_id}")
            error_msg = AgentError(
                request_id=msg.request_id,
                agent_name="text_bias_agent",
                error_type="no_content",
                error_message="No text content provided"
            )
            await ctx.send(SCORING_AGENT_ADDRESS, error_msg)
            return

        ctx.logger.info(f"📝 Text length: {len(msg.text_content)} characters")

        # Step 1: Initial text analysis
        ctx.logger.info(f"🔍 Starting bias detection analysis...")
        initial_analysis = await analyze_text_with_llm(ctx, msg.text_content)

        # Step 2: RAG RETRIEVAL - Query ChromaDB for similar patterns
        ctx.logger.info(f"🔎 RAG RETRIEVAL: Querying ChromaDB for similar text patterns...")
        rag_results = await query_bias_knowledge_base(ctx, msg.text_embedding, msg.chromadb_collection_id)
        ctx.logger.info(f"✅ Found {len(rag_results)} similar cases from knowledge base")

        # Step 3: Classify and extract bias types
        ctx.logger.info(f"🏷️ Classifying detected bias types...")
        bias_instances = await classify_and_extract_biases(ctx, initial_analysis, rag_results)

        # Step 4: Calculate overall text bias score
        ctx.logger.info(f"📊 Calculating overall text bias score...")
        text_score = await calculate_text_score(ctx, bias_instances)

        # Step 5: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = await generate_recommendations(ctx, bias_instances)

        # Step 6: Create report
        report = TextBiasReport(
            request_id=msg.request_id,
            agent_name="text_bias_agent",
            bias_detected=len(bias_instances) > 0,
            bias_instances=bias_instances,
            overall_text_score=text_score,
            recommendations=recommendations,
            rag_similar_cases=[ref["id"] for ref in rag_results]
        )

        ctx.logger.info(f"✅ Analysis complete: Score={text_score:.1f}, Issues={len(bias_instances)}")

        # Step 7: Send report to Scoring Agent
        await send_to_scoring_agent(ctx, msg.request_id, report)

    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing text: {e}")
        error_msg = AgentError(
            request_id=msg.request_id,
            agent_name="text_bias_agent",
            error_type="analysis_error",
            error_message=str(e)
        )
        await ctx.send(SCORING_AGENT_ADDRESS, error_msg)


async def analyze_text_with_llm(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Use ASI:ONE LLM to perform initial bias detection analysis.
    
    TODO: Implement actual ASI:ONE integration
    - Send structured prompt to OPENAI_AGENT or CLAUDE_AGENT
    - Request bias detection with chain-of-thought reasoning
    """
    ctx.logger.info(f"🧠 Analyzing text with ASI:ONE LLM...")
    
    # Placeholder analysis
    analysis = {
        "potential_biases": ["gender", "age"],
        "problematic_phrases": [
            "young energetic team",
            "looking for rockstars"
        ],
        "reasoning": "Text shows preference for youth and uses gendered sports metaphors"
    }
    
    ctx.logger.info(f"✅ LLM analysis complete")
    return analysis


async def query_bias_knowledge_base(
    ctx: Context,
    text_embedding: Optional[List[float]],
    collection_id: str
) -> List[Dict[str, Any]]:
    """
    RAG RETRIEVAL POINT #1: Query ChromaDB for similar historical text bias cases.
    
    TODO: Implement actual ChromaDB query
    - Use text_embedding for semantic similarity search
    - Filter by bias_patterns_text collection
    - Return top-k similar cases with metadata
    """
    ctx.logger.info(f"🔎 Querying ChromaDB for similar text patterns...")
    
    # Placeholder RAG results
    rag_results = [
        {
            "id": "case_text_001",
            "bias_type": "gender_bias",
            "similarity": 0.87,
            "context": "Similar use of gendered language in tech recruitment"
        },
        {
            "id": "case_text_015",
            "bias_type": "age_bias",
            "similarity": 0.82,
            "context": "Youth-centric language pattern detected"
        }
    ]
    
    ctx.logger.info(f"✅ Retrieved {len(rag_results)} similar cases")
    return rag_results


async def classify_and_extract_biases(
    ctx: Context,
    initial_analysis: Dict[str, Any],
    rag_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Classify bias types and extract specific examples.
    Returns list of bias instance dicts.
    """
    ctx.logger.info(f"🏷️ Classifying bias types...")

    # Placeholder bias detections using BiasCategory
    detections = [
        create_bias_instance_dict(
            bias_type=BiasCategory.GENDER,
            severity="medium",
            examples=["rockstars", "guys"],
            context="Uses gendered sports metaphors and male-default language",
            confidence=0.87
        ),
        create_bias_instance_dict(
            bias_type=BiasCategory.AGE,
            severity="medium",
            examples=["young energetic team", "digital natives"],
            context="Implies preference for younger candidates",
            confidence=0.82
        )
    ]

    ctx.logger.info(f"✅ Classified {len(detections)} bias instances")
    return detections


async def calculate_text_score(ctx: Context, detections: List[Dict[str, Any]]) -> float:
    """
    Calculate overall text bias score (0-10 scale).
    
    Scoring:
    - 0-3: Significant bias (high concern)
    - 4-6: Moderate bias (needs revision)
    - 7-8: Minor bias (minor improvements)
    - 9-10: Minimal bias (approved)
    """
    if not detections:
        return 9.5  # No bias detected

    # Calculate weighted score based on severity
    severity_weights = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "critical": 2.0
    }

    total_penalty = sum(severity_weights.get(d.get("severity", "low"), 0.5) * d.get("confidence", 0.5) for d in detections)

    # Start from 10 and subtract penalties
    score = max(0.0, 10.0 - total_penalty)

    ctx.logger.info(f"📊 Calculated text score: {score:.1f}")
    return score


async def generate_recommendations(ctx: Context, detections: List[Dict[str, Any]]) -> List[str]:
    """
    Generate actionable recommendations based on detected biases.
    """
    recommendations = []

    for detection in detections:
        bias_type = detection.get("bias_type", "")

        if bias_type == "gender_bias":
            recommendations.append("Use gender-neutral language and avoid male-default terminology")
        elif bias_type == "age_bias":
            recommendations.append("Remove age-specific references and focus on skills/experience")
        elif bias_type == "racial_bias":
            recommendations.append("Ensure cultural sensitivity and avoid stereotypes")
        elif bias_type == "disability_bias":
            recommendations.append("Use inclusive language that doesn't assume physical abilities")
        elif bias_type == "lgbtq_bias":
            recommendations.append("Avoid heteronormative assumptions and use inclusive terminology")
        elif bias_type == "socioeconomic_bias":
            recommendations.append("Avoid language that assumes specific socioeconomic backgrounds")

    ctx.logger.info(f"💡 Generated {len(recommendations)} recommendations")
    return recommendations


# ChromaDB instance
_chroma_db = None

def get_chroma_db():
    """Get ChromaDB instance"""
    global _chroma_db
    if _chroma_db is None:
        _chroma_db = ChromaDB()
    return _chroma_db


async def send_to_scoring_agent(ctx: Context, request_id: str, report: TextBiasReport):
    """
    Send analysis results to Scoring Agent.
    """
    if not SCORING_AGENT_ADDRESS:
        ctx.logger.warning(f"⚠️ Scoring agent address not set")
        return

    # Convert report to dict
    report_dict = {
        "request_id": report.request_id,
        "agent_name": report.agent_name,
        "bias_detected": report.bias_detected,
        "bias_instances": report.bias_instances,
        "overall_text_score": report.overall_text_score,
        "recommendations": report.recommendations,
        "rag_similar_cases": report.rag_similar_cases,
        "timestamp": report.timestamp
    }

    analysis_complete = BiasAnalysisComplete(
        request_id=request_id,
        sender_agent="text_bias_agent",
        report=report_dict
    )

    ctx.logger.info(f"📤 Sending results to Scoring Agent")
    await ctx.send(SCORING_AGENT_ADDRESS, analysis_complete)
    ctx.logger.info(f"✅ Results sent successfully")


# Include protocol
text_bias_agent.include(text_bias_protocol, publish_manifest=True)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        🧠 TEXT BIAS AGENT - Ad Bias Detection                ║
╚══════════════════════════════════════════════════════════════╝

Role: Text Content Analysis and Bias Detection

Capabilities:
  ✓ Analyzes text for linguistic bias patterns
  ✓ RAG retrieval from ChromaDB for similar cases
  ✓ Detects gender, racial, age, socioeconomic bias
  ✓ ASI:ONE LLM integration for intelligent analysis
  ✓ Provides confidence scores and recommendations

Bias Types Detected:
  • Gender bias (stereotyping, exclusionary language)
  • Racial/ethnic bias (cultural appropriation, stereotypes)
  • Age bias (ageism, generational stereotypes)
  • Socioeconomic bias (class assumptions)
  • Disability bias (ableist language)
  • LGBTQ+ bias (heteronormative assumptions)

📍 Waiting for text analysis requests...
🛑 Stop with Ctrl+C
    """)
    text_bias_agent.run()

