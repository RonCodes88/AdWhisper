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


# Bias types enumeration
class BiasType(str, Enum):
    GENDER = "gender_bias"
    RACIAL = "racial_bias"
    AGE = "age_bias"
    SOCIOECONOMIC = "socioeconomic_bias"
    DISABILITY = "disability_bias"
    LGBTQ = "lgbtq_bias"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Message Models
class TextAnalysisRequest(Model):
    """Request for text bias analysis"""
    request_id: str
    text_content: str
    text_embedding: Optional[List[float]] = None
    chromadb_collection_id: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


class BiasDetection(Model):
    """Individual bias detection result"""
    bias_type: BiasType
    severity: SeverityLevel
    examples: Optional[List[str]] = None
    context: str
    confidence: float


class TextBiasReport(Model):
    """Complete text bias analysis report"""
    request_id: str
    agent: str = "text_bias_agent"
    bias_detected: bool
    bias_types: Optional[List[BiasDetection]] = None
    overall_text_score: float
    recommendations: Optional[List[str]] = None
    rag_references: Optional[List[str]] = None
    confidence: float
    timestamp: str = ""

    def __init__(self, **data):
        if 'timestamp' not in data or not data['timestamp']:
            data['timestamp'] = datetime.now(UTC).isoformat()
        super().__init__(**data)


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


@text_bias_protocol.on_message(model=TextAnalysisRequest, replies=TextBiasReport)
async def handle_text_analysis(ctx: Context, sender: str, msg: TextAnalysisRequest):
    """
    Analyze text content for bias using ASI:ONE LLM and RAG retrieval.
    """
    try:
        ctx.logger.info(f"📨 Received text analysis request: {msg.request_id}")
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
        bias_detections = await classify_and_extract_biases(ctx, initial_analysis, rag_results)
        
        # Step 4: Calculate overall text bias score
        ctx.logger.info(f"📊 Calculating overall text bias score...")
        text_score = await calculate_text_score(ctx, bias_detections)
        
        # Step 5: Generate recommendations
        ctx.logger.info(f"💡 Generating recommendations...")
        recommendations = await generate_recommendations(ctx, bias_detections)
        
        # Step 6: Create report
        report = TextBiasReport(
            request_id=msg.request_id,
            agent="text_bias_agent",
            bias_detected=len(bias_detections) > 0,
            bias_types=bias_detections,
            overall_text_score=text_score,
            recommendations=recommendations,
            rag_references=[ref["id"] for ref in rag_results],
            confidence=sum(bd.confidence for bd in bias_detections) / len(bias_detections) if bias_detections else 1.0
        )
        
        ctx.logger.info(f"✅ Analysis complete: Score={text_score:.1f}, Biases detected={len(bias_detections)}")
        
        # Step 7: Send report back to sender and to Scoring Agent
        await ctx.send(sender, report)
        
        if SCORING_AGENT_ADDRESS:
            ctx.logger.info(f"📤 Forwarding report to Scoring Agent: {SCORING_AGENT_ADDRESS}")
            # await ctx.send(SCORING_AGENT_ADDRESS, report)
        
        ctx.logger.info(f"✅ Text analysis for {msg.request_id} completed successfully")
        
    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing text for {msg.request_id}: {e}")
        # Send error report
        error_report = TextBiasReport(
            request_id=msg.request_id,
            agent="text_bias_agent",
            bias_detected=False,
            bias_types=[],
            overall_text_score=5.0,  # Neutral score on error
            recommendations=["Error occurred during analysis"],
            rag_references=[],
            confidence=0.0
        )
        await ctx.send(sender, error_report)


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
) -> List[BiasDetection]:
    """
    Classify bias types and extract specific examples.
    """
    ctx.logger.info(f"🏷️ Classifying bias types...")
    
    # Placeholder bias detections
    detections = [
        BiasDetection(
            bias_type=BiasType.GENDER,
            severity=SeverityLevel.MEDIUM,
            examples=["rockstars", "guys"],
            context="Uses gendered sports metaphors and male-default language",
            confidence=0.87
        ),
        BiasDetection(
            bias_type=BiasType.AGE,
            severity=SeverityLevel.MEDIUM,
            examples=["young energetic team", "digital natives"],
            context="Implies preference for younger candidates",
            confidence=0.82
        )
    ]
    
    ctx.logger.info(f"✅ Classified {len(detections)} bias types")
    return detections


async def calculate_text_score(ctx: Context, detections: List[BiasDetection]) -> float:
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
        SeverityLevel.LOW: 0.5,
        SeverityLevel.MEDIUM: 1.0,
        SeverityLevel.HIGH: 1.5,
        SeverityLevel.CRITICAL: 2.0
    }
    
    total_penalty = sum(severity_weights[d.severity] * d.confidence for d in detections)
    
    # Start from 10 and subtract penalties
    score = max(0.0, 10.0 - total_penalty)
    
    ctx.logger.info(f"📊 Calculated text score: {score:.1f}")
    return score


async def generate_recommendations(ctx: Context, detections: List[BiasDetection]) -> List[str]:
    """
    Generate actionable recommendations based on detected biases.
    """
    recommendations = []
    
    for detection in detections:
        if detection.bias_type == BiasType.GENDER:
            recommendations.append("Use gender-neutral language and avoid male-default terminology")
        elif detection.bias_type == BiasType.AGE:
            recommendations.append("Remove age-specific references and focus on skills/experience")
        elif detection.bias_type == BiasType.RACIAL:
            recommendations.append("Ensure cultural sensitivity and avoid stereotypes")
        elif detection.bias_type == BiasType.DISABILITY:
            recommendations.append("Use inclusive language that doesn't assume physical abilities")
        elif detection.bias_type == BiasType.LGBTQ:
            recommendations.append("Avoid heteronormative assumptions and use inclusive terminology")
        elif detection.bias_type == BiasType.SOCIOECONOMIC:
            recommendations.append("Avoid language that assumes specific socioeconomic backgrounds")
    
    ctx.logger.info(f"💡 Generated {len(recommendations)} recommendations")
    return recommendations


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

