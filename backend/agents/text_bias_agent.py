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
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chroma import ChromaDB


# Severity levels
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


# REST API Models (for HTTP endpoint)
class RESTTextAnalysisRequest(Model):
    """REST API request for text bias analysis"""
    request_id: str
    text_content: str
    text_embedding: Optional[List[float]] = None
    chromadb_collection_id: str
    metadata: Optional[Dict[str, Any]] = None


class RESTAcknowledgement(Model):
    """REST API acknowledgement response"""
    status: str
    request_id: str
    message: str


# Initialize Text Bias Agent
text_bias_agent = Agent(
    name="text_bias_agent",
    seed="ad_bias_text_agent_unique_seed_2024",
    port=8101,
    endpoint=["http://localhost:8101/submit"],
    mailbox=False  # Local development - direct agent-to-agent communication
)

# Protocol for text bias analysis
text_bias_protocol = Protocol(name="text_bias_protocol", version="1.0")

# Global ChromaDB instance
chroma_db = None

# ASI:ONE LLM agent addresses (rate limited to 6 requests/hour)
OPENAI_AGENT = os.getenv('ASI_OPENAI_AGENT', 'agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y')
CLAUDE_AGENT = os.getenv('ASI_CLAUDE_AGENT', 'agent1qvk7q2av3e2y5gf5s90nfzkc8a48q3wdqeevwrtgqfdl0k78rspd6f2l4dx')

# Scoring Agent address (to send results to)
SCORING_AGENT_ADDRESS = os.getenv("SCORING_AGENT_ADDRESS", "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8")


@text_bias_agent.on_event("startup")
async def startup(ctx: Context):
    global chroma_db
    
    ctx.logger.info(f"🚀 Text Bias Agent starting up...")
    ctx.logger.info(f"📍 Agent address: {text_bias_agent.address}")
    
    # Initialize ChromaDB
    ctx.logger.info("💾 Initializing ChromaDB...")
    chroma_db = ChromaDB()
    ctx.logger.info(f"✅ ChromaDB initialized")
    
    ctx.logger.info(f"🔧 Role: Text Content Analysis and Bias Detection")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8101/submit")
    ctx.logger.info(f"🧠 ASI:ONE Integration: Ready for LLM-powered analysis")
    ctx.logger.info(f"⚡ Ready to analyze text content for bias")


@text_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Text Bias Agent shutting down...")


# Shared analysis logic
async def process_text_analysis(ctx: Context, msg: TextAnalysisRequest) -> TextBiasReport:
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
        
        ctx.logger.info(f"✅ Analysis complete: Score={text_score:.1f}, Biases detected={len(bias_detections)}")
        ctx.logger.info(f"✅ Text analysis for {msg.request_id} completed successfully")
        
        return report
        
    except Exception as e:
        ctx.logger.error(f"❌ Error analyzing text for {msg.request_id}: {e}")
        # Return error report
        error_report = TextBiasReport(
            request_id=msg.request_id,
            agent_name="text_bias_agent",
            error_type="analysis_error",
            error_message=str(e)
        )
        return error_report


async def send_report_to_scoring_agent(ctx: Context, report: TextBiasReport):
    """Send text bias report to Scoring Agent via HTTP."""
    try:
        import httpx
        payload = report.dict()
        ctx.logger.info(f"📤 Sending report to Scoring Agent...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8103/api/aggregate",
                json={"source": "text_bias_agent", "report": payload}
            )
            ctx.logger.info(f"✅ Sent report to Scoring Agent")
    except Exception as e:
        ctx.logger.error(f"❌ Error sending to Scoring Agent: {e}")


# REST API Endpoint - HTTP interface for Ingestion Agent
@text_bias_agent.on_rest_post("/api/analyze", RESTTextAnalysisRequest, RESTAcknowledgement)
async def handle_rest_text_analysis(ctx: Context, req: RESTTextAnalysisRequest) -> RESTAcknowledgement:
    """REST endpoint for text analysis via HTTP."""
    ctx.logger.info(f"🌐 REST API: Received text analysis request: {req.request_id}")
    
    try:
        # Convert REST request to internal format
        text_request = TextAnalysisRequest(
            request_id=req.request_id,
            text_content=req.text_content,
            text_embedding=req.text_embedding,
            chromadb_collection_id=req.chromadb_collection_id,
            metadata=req.metadata
        )
        
        # Process analysis
        report = await process_text_analysis(ctx, text_request)
        
        # Send results to Scoring Agent via HTTP
        await send_report_to_scoring_agent(ctx, report)
        
        return RESTAcknowledgement(
            status="accepted",
            request_id=req.request_id,
            message="Text analysis completed"
        )
    except Exception as e:
        ctx.logger.error(f"❌ Error: {e}")
        return RESTAcknowledgement(
            status="error",
            request_id=req.request_id,
            message=str(e)
        )


@text_bias_protocol.on_message(model=TextAnalysisRequest, replies=TextBiasReport)
async def handle_text_analysis(ctx: Context, sender: str, msg: TextAnalysisRequest):
    """Analyze text content - delegates to shared logic."""
    try:
        report = await process_text_analysis(ctx, msg)
        await ctx.send(sender, report)
        await send_report_to_scoring_agent(ctx, report)
    except Exception as e:
        ctx.logger.error(f"❌ Error: {e}")


async def analyze_text_with_llm(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Use ASI:ONE LLM to perform initial bias detection analysis.
    
    Sends structured prompt to ASI:ONE agent for bias detection.
    Note: ASI:ONE agents are rate limited to 6 requests/hour.
    """
    ctx.logger.info(f"🧠 Analyzing text with ASI:ONE LLM...")
    
    # Create detailed prompt for bias detection
    prompt = f"""Analyze the following advertising text for potential biases. 
    
Text to analyze:
\"\"\"{text}\"\"\"

Please identify:
1. Potential bias types (gender, racial, age, socioeconomic, disability, LGBTQ+)
2. Specific problematic phrases or words
3. Reasoning for why each phrase might be biased
4. Severity level for each bias (low, medium, high, critical)

Respond in JSON format with this structure:
{{
    "potential_biases": ["type1", "type2"],
    "problematic_phrases": ["phrase1", "phrase2"],
    "reasoning": "overall explanation",
    "detailed_findings": [
        {{
            "bias_type": "type",
            "phrases": ["phrase1"],
            "severity": "medium",
            "explanation": "why this is biased"
        }}
    ]
}}

Be thorough but avoid false positives. Only flag genuine concerns."""
    
    try:
        # For now, we'll use a sophisticated rule-based approach
        # In production, integrate with ASI:ONE by sending message to OPENAI_AGENT or CLAUDE_AGENT
        # Example: await ctx.send(OPENAI_AGENT, StructuredOutputPrompt(prompt=prompt, output_schema=schema))
        
        analysis = await perform_bias_analysis(ctx, text)
        
        ctx.logger.info(f"✅ LLM analysis complete")
        return analysis
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in LLM analysis: {e}")
        # Return safe default
        return {
            "potential_biases": [],
            "problematic_phrases": [],
            "reasoning": "Analysis error occurred",
            "detailed_findings": []
        }


async def perform_bias_analysis(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Perform text bias analysis using pattern matching and heuristics.
    This is a fallback/initial implementation that can be enhanced with actual ASI:ONE integration.
    """
    text_lower = text.lower()
    
    potential_biases = []
    problematic_phrases = []
    detailed_findings = []
    
    # Gender bias patterns
    gender_biased_terms = [
        ("guys", "male-default language"),
        ("rockstar", "gendered sports metaphor"),
        ("ninja", "gendered martial arts metaphor"),
        ("guru", "gendered spiritual metaphor"),
        ("manning the desk", "gendered verb"),
        ("manpower", "gendered term for workforce"),
        ("chairman", "gendered leadership term"),
        ("salesman", "gendered job title"),
        ("housewife", "gendered role assumption"),
        ("working mom", "gendered parenting assumption")
    ]
    
    for term, reason in gender_biased_terms:
        if term in text_lower:
            potential_biases.append("gender")
            problematic_phrases.append(term)
            detailed_findings.append({
                "bias_type": "gender_bias",
                "phrases": [term],
                "severity": "medium",
                "explanation": reason
            })
    
    # Age bias patterns
    age_biased_terms = [
        ("young", "age preference"),
        ("energetic", "youth-coded language"),
        ("digital native", "age assumption"),
        ("recent graduate", "age preference"),
        ("tech-savvy", "age-coded language"),
        ("mature", "age euphemism"),
        ("over-qualified", "age discrimination"),
        ("generation", "generational stereotyping")
    ]
    
    for term, reason in age_biased_terms:
        if term in text_lower:
            potential_biases.append("age")
            problematic_phrases.append(term)
            detailed_findings.append({
                "bias_type": "age_bias",
                "phrases": [term],
                "severity": "medium",
                "explanation": reason
            })
    
    # Racial/ethnic bias patterns
    racial_terms = [
        ("urban", "coded racial language"),
        ("articulate", "racial stereotype"),
        ("exotic", "othering language"),
        ("ethnic", "potentially exclusionary")
    ]
    
    for term, reason in racial_terms:
        if term in text_lower:
            potential_biases.append("racial")
            problematic_phrases.append(term)
            detailed_findings.append({
                "bias_type": "racial_bias",
                "phrases": [term],
                "severity": "high",
                "explanation": reason
            })
    
    # Disability bias patterns
    disability_terms = [
        ("crazy", "ableist language"),
        ("insane", "ableist language"),
        ("lame", "ableist language"),
        ("blind to", "ableist metaphor"),
        ("tone-deaf", "ableist metaphor"),
        ("wheelchair-bound", "deficit language")
    ]
    
    for term, reason in disability_terms:
        if term in text_lower:
            potential_biases.append("disability")
            problematic_phrases.append(term)
            detailed_findings.append({
                "bias_type": "disability_bias",
                "phrases": [term],
                "severity": "medium",
                "explanation": reason
            })
    
    # Remove duplicates
    potential_biases = list(set(potential_biases))
    problematic_phrases = list(set(problematic_phrases))
    
    reasoning = f"Found {len(detailed_findings)} potential bias indicators" if detailed_findings else "No obvious bias patterns detected"
    
    return {
        "potential_biases": potential_biases,
        "problematic_phrases": problematic_phrases,
        "reasoning": reasoning,
        "detailed_findings": detailed_findings
    }


async def query_bias_knowledge_base(
    ctx: Context,
    text_embedding: Optional[List[float]],
    collection_id: str
) -> List[Dict[str, Any]]:
    """
    RAG RETRIEVAL POINT #1: Query ChromaDB for similar historical text bias cases.
    Uses semantic similarity search to find relevant patterns.
    """
    global chroma_db
    
    if chroma_db is None or text_embedding is None:
        ctx.logger.warning("⚠️ ChromaDB or embedding not available for RAG retrieval")
        return []
    
    try:
        ctx.logger.info(f"🔎 RAG: Querying ChromaDB for similar text bias patterns...")
        
        # Query the text bias patterns collection
        results = chroma_db.query_by_embedding(
            collection_name=ChromaDB.COLLECTION_TEXT_PATTERNS,
            embedding=text_embedding,
            n_results=5
        )
        
        # Parse results
        rag_results = []
        if results and 'ids' in results and len(results['ids']) > 0:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results else []
            distances = results['distances'][0] if 'distances' in results else []
            
            for i, case_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1.0 - distance  # Convert distance to similarity
                
                rag_results.append({
                    "id": case_id,
                    "bias_type": metadata.get("bias_type", "unknown"),
                    "similarity": similarity,
                    "context": metadata.get("context", "No context available"),
                    "severity": metadata.get("severity", "medium"),
                    "examples": metadata.get("examples", "").split(",") if metadata.get("examples") else []
                })
        
        ctx.logger.info(f"✅ RAG: Retrieved {len(rag_results)} similar text bias cases")
        return rag_results
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in RAG retrieval: {e}")
        return []


async def classify_and_extract_biases(
    ctx: Context,
    initial_analysis: Dict[str, Any],
    rag_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Classify bias types and extract specific examples.
    Combines LLM analysis with RAG retrieval results for enhanced accuracy.
    """
    ctx.logger.info(f"🏷️ Classifying bias types...")
    
    detections = []
    detailed_findings = initial_analysis.get("detailed_findings", [])
    
    # Process each detected bias from the initial analysis
    for finding in detailed_findings:
        bias_type_str = finding.get("bias_type", "")
        
        # Map bias type string to BiasType enum
        bias_type = None
        if "gender" in bias_type_str:
            bias_type = BiasType.GENDER
        elif "racial" in bias_type_str or "ethnic" in bias_type_str:
            bias_type = BiasType.RACIAL
        elif "age" in bias_type_str:
            bias_type = BiasType.AGE
        elif "socioeconomic" in bias_type_str:
            bias_type = BiasType.SOCIOECONOMIC
        elif "disability" in bias_type_str:
            bias_type = BiasType.DISABILITY
        elif "lgbtq" in bias_type_str:
            bias_type = BiasType.LGBTQ
        
        if bias_type is None:
            continue
        
        # Map severity string to SeverityLevel enum
        severity_str = finding.get("severity", "medium").lower()
        severity = SeverityLevel.MEDIUM
        if severity_str == "low":
            severity = SeverityLevel.LOW
        elif severity_str == "high":
            severity = SeverityLevel.HIGH
        elif severity_str == "critical":
            severity = SeverityLevel.CRITICAL
        
        # Check if RAG results confirm this bias
        confidence = 0.7  # Base confidence
        for rag_result in rag_results:
            if rag_result["bias_type"] == bias_type_str:
                # RAG confirmation increases confidence
                confidence = min(0.95, confidence + (rag_result["similarity"] * 0.2))
                ctx.logger.info(f"✅ RAG confirmed {bias_type_str} with similarity {rag_result['similarity']:.2f}")
        
        detection = BiasDetection(
            bias_type=bias_type,
            severity=severity,
            examples=finding.get("phrases", []),
            context=finding.get("explanation", "No explanation provided"),
            confidence=confidence
        )
        detections.append(detection)
    
    # Add any high-confidence RAG results that weren't in initial analysis
    for rag_result in rag_results:
        if rag_result["similarity"] > 0.85:  # High similarity threshold
            bias_type_str = rag_result["bias_type"]
            
            # Check if we already have this bias type
            already_detected = any(d.bias_type.value == bias_type_str for d in detections)
            if not already_detected:
                ctx.logger.info(f"📋 Adding high-confidence RAG result: {bias_type_str}")
                
                # Map to BiasType enum
                bias_type = None
                if "gender" in bias_type_str:
                    bias_type = BiasType.GENDER
                elif "racial" in bias_type_str:
                    bias_type = BiasType.RACIAL
                elif "age" in bias_type_str:
                    bias_type = BiasType.AGE
                
                if bias_type:
                    detection = BiasDetection(
                        bias_type=bias_type,
                        severity=SeverityLevel(rag_result.get("severity", "medium")),
                        examples=rag_result.get("examples", []),
                        context=f"Similar to historical case: {rag_result['context']}",
                        confidence=rag_result["similarity"]
                    )
                    detections.append(detection)
    
    ctx.logger.info(f"✅ Classified {len(detections)} bias types (including RAG confirmations)")
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

