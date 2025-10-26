"""
Text Bias Agent - Ad Bias Detection System

Role: Text Content Analysis and Bias Detection
Responsibilities:
- Analyze textual content for bias indicators using Claude Haiku LLM
- Identify specific bias types (gender, racial, age, socioeconomic, disability, LGBTQ+)
- Extract problematic phrases and provide contextual explanations
- Generate structured findings with confidence scores
- Provide RESTful API for integration with other agents
- Send results to Scoring Agent for final assessment
"""

from uagents import Agent, Context, Model, Protocol
from pydantic import Field
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from enum import Enum
import sys
import os
import json
import aiohttp
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import shared models
from agents.shared_models import (
    EmbeddingPackage,
    TextBiasReport,
    BiasAnalysisComplete,
    BiasCategory,
    create_bias_instance_dict,
    AgentError
)

# Initialize Anthropic client for fast LLM analysis
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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

# Scoring Agent address (to send results to)
SCORING_AGENT_ADDRESS = "agent1qv8q8vexn2l4hx08m30ecu329g0gfw3ede4ngf7j2gg756er4y5wcqlx9s8"


@text_bias_agent.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"🚀 Text Bias Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {text_bias_agent.address}")
    ctx.logger.info(f"🔧 Role: Text Content Analysis and Bias Detection")
    ctx.logger.info(f"🌐 Endpoint: http://localhost:8101/submit")
    ctx.logger.info(f"🧠 Claude Haiku Integration: Ready for fast LLM-powered analysis")
    ctx.logger.info(f"⚡ Ready to analyze text content for bias")
    
    # Check if Anthropic API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        ctx.logger.warning(f"⚠️ ANTHROPIC_API_KEY not set - LLM analysis will fail!")
    else:
        ctx.logger.info(f"✅ Anthropic API key configured")


@text_bias_agent.on_event("shutdown")
async def shutdown(ctx: Context):
    ctx.logger.info("🛑 Text Bias Agent shutting down...")


@text_bias_agent.on_rest_get("/report/{request_id}", BiasAnalysisComplete)
async def get_text_report(ctx: Context, request_id: str) -> BiasAnalysisComplete:
    """
    REST GET endpoint for Scoring Agent to retrieve stored text bias report.
    
    Usage: GET http://localhost:8101/report/{request_id}
    """
    ctx.logger.info(f"📨 REST GET request for report: {request_id}")
    
    # Retrieve from storage
    report_key = f"text_report_{request_id}"
    report_data = ctx.storage.get(report_key)
    
    if report_data:
        ctx.logger.info(f"✅ Text bias report found and returned")
        # Convert dict back to TextBiasReport model, then to BiasAnalysisComplete
        return BiasAnalysisComplete(
            request_id=request_id,
            sender_agent="text_bias_agent",
            report=report_data
        )
    else:
        ctx.logger.warning(f"⚠️ Text bias report not found for request_id: {request_id}")
        # Return a "not found" response
        return BiasAnalysisComplete(
            request_id=request_id,
            sender_agent="text_bias_agent",
            report={
                "request_id": request_id,
                "agent_name": "text_bias_agent",
                "error": "Report not found. Analysis may still be in progress or request_id is invalid.",
                "bias_detected": False,
                "overall_text_score": 0.0,
                "recommendations": []
            }
        )


@text_bias_agent.on_rest_post("/analyze", EmbeddingPackage, BiasAnalysisComplete)
async def handle_text_analysis_rest(ctx: Context, req: EmbeddingPackage) -> BiasAnalysisComplete:
    """
    REST endpoint for text bias analysis.
    Analyzes text content for bias and returns results.
    """
    ctx.logger.info("=" * 80)
    ctx.logger.info("🎯 TEXT BIAS AGENT - REST REQUEST RECEIVED")
    ctx.logger.info("=" * 80)

    try:
        ctx.logger.info(f"📨 Received REST request for text analysis")
        ctx.logger.info(f"   📝 Request ID: {req.request_id}")
        ctx.logger.info(f"   📄 Has text content: {req.text_content is not None}")
        ctx.logger.info(f"   🔢 Has text embedding: {req.text_embedding is not None}")

        if req.text_content:
            ctx.logger.info(f"   📏 Text length: {len(req.text_content)} characters")
            ctx.logger.info(f"   📖 Text preview: {req.text_content[:100]}...")
        else:
            ctx.logger.warning(f"   ⚠️ Text content is None or empty!")

        # Check if we have text to analyze
        if not req.text_content:
            ctx.logger.error(f"❌ No text content for request {req.request_id}")
            # Return error response
            return BiasAnalysisComplete(
                request_id=req.request_id,
                sender_agent="text_bias_agent",
                report={
                    "request_id": req.request_id,
                    "agent_name": "text_bias_agent",
                    "error": "No text content provided"
                }
            )

        ctx.logger.info(f"✅ Text content validated - length: {len(req.text_content)} characters")

        # Step 1: Initial text analysis
        ctx.logger.info(f"🔍 STEP 1: Starting bias detection analysis...")
        initial_analysis = await analyze_text_with_llm(ctx, req.text_content)
        ctx.logger.info(f"   ✅ Initial analysis complete")
        ctx.logger.info(f"   📊 Has bias: {initial_analysis.get('has_bias', False)}")
        ctx.logger.info(f"   📊 Bias types found: {len(initial_analysis.get('bias_types', []))}")

        # Step 2: Classify and extract bias types (ChromaDB removed for speed)
        ctx.logger.info(f"🏷️ STEP 2: Classifying detected bias types...")
        bias_instances = await classify_and_extract_biases(ctx, initial_analysis)
        ctx.logger.info(f"   ✅ Classified {len(bias_instances)} bias instances")

        # Step 3: Calculate overall text bias score
        ctx.logger.info(f"📊 STEP 3: Calculating overall text bias score...")
        text_score = await calculate_text_score(ctx, bias_instances)
        ctx.logger.info(f"   ✅ Text score calculated: {text_score:.2f}/10")

        # Step 4: Generate recommendations
        ctx.logger.info(f"💡 STEP 4: Generating recommendations...")
        recommendations = await generate_recommendations(ctx, bias_instances)
        ctx.logger.info(f"   ✅ Generated {len(recommendations)} recommendations")

        # Step 5: Create report
        ctx.logger.info(f"📋 STEP 5: Creating bias report...")
        report = TextBiasReport(
            request_id=req.request_id,
            agent_name="text_bias_agent",
            bias_detected=len(bias_instances) > 0,
            bias_instances=bias_instances,
            overall_text_score=text_score,
            recommendations=recommendations,
            rag_similar_cases=[]  # ChromaDB removed for speed
        )
        ctx.logger.info(f"   ✅ Report created")

        ctx.logger.info(f"🎉 Analysis complete!")
        ctx.logger.info(f"   📊 Overall Score: {text_score:.1f}/10")
        ctx.logger.info(f"   🚨 Bias detected: {len(bias_instances) > 0}")
        ctx.logger.info(f"   📝 Issues found: {len(bias_instances)}")
        ctx.logger.info(f"   💡 Recommendations: {len(recommendations)}")

        # Step 6: Store report for Scoring Agent to retrieve via REST
        ctx.logger.info(f"📤 STEP 6: Storing report for Scoring Agent retrieval...")
        report_key = f"text_report_{req.request_id}"
        ctx.storage.set(report_key, report.dict())
        ctx.logger.info(f"   💾 Report stored with key: {report_key}")
        ctx.logger.info(f"   🌐 Scoring Agent can retrieve via: GET /report/{req.request_id}")
        
        # Step 7: Trigger Scoring Agent via REST to aggregate reports
        ctx.logger.info(f"📤 STEP 7: Triggering Scoring Agent via HTTP...")
        await trigger_scoring_agent(ctx, req.request_id, req.chromadb_collection_id)

        # Return response to REST caller
        response = BiasAnalysisComplete(
            request_id=req.request_id,
            sender_agent="text_bias_agent",
            report={
                "request_id": report.request_id,
                "agent_name": report.agent_name,
                "bias_detected": report.bias_detected,
                "bias_instances": report.bias_instances,
                "overall_text_score": report.overall_text_score,
                "recommendations": report.recommendations,
                "rag_similar_cases": report.rag_similar_cases,
                "timestamp": report.timestamp
            }
        )
        ctx.logger.info(f"✅ Returning response to REST caller")
        ctx.logger.info("=" * 80)
        return response

    except Exception as e:
        ctx.logger.error("=" * 80)
        ctx.logger.error(f"❌ ERROR IN TEXT BIAS AGENT")
        ctx.logger.error("=" * 80)
        ctx.logger.error(f"   Request ID: {req.request_id}")
        ctx.logger.error(f"   Error: {e}")
        ctx.logger.error(f"   Type: {type(e).__name__}")
        import traceback
        ctx.logger.error(f"   Traceback:\n{traceback.format_exc()}")
        ctx.logger.error("=" * 80)

        # Return error response
        return BiasAnalysisComplete(
            request_id=req.request_id,
            sender_agent="text_bias_agent",
            report={
                "request_id": req.request_id,
                "agent_name": "text_bias_agent",
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


async def analyze_text_with_llm(ctx: Context, text: str) -> Dict[str, Any]:
    """
    Use Claude (Haiku) for fast bias detection analysis.
    Returns structured analysis with bias findings.
    Uses plain text parsing instead of JSON for reliability.
    """
    ctx.logger.info(f"🧠 Analyzing text with Claude LLM...")
    
    try:
        # Create prompt for bias detection - using structured text instead of JSON
        system_prompt = """You are an expert TEXT BIAS detection system for advertising content. Your role is to analyze TEXTUAL content for linguistic bias patterns.

Focus specifically on analyzing THE WORDS, PHRASES, AND LANGUAGE used in the advertisement. Look for:

1. **TEXT-BASED RACIAL/ETHNIC BIAS** (CRITICAL PRIORITY):
   - Does the text contain color associations with value judgments? (e.g., "white = pure/clean", "black = dirty/bad")
   - Does the text contain racial stereotypes or coded language?
   - Does the text reference cultural stereotypes or appropriation?
   - Does the text contain exclusionary language targeting specific ethnic groups?

2. **TEXT-BASED GENDER BIAS** (CRITICAL PRIORITY):
   - Does the text perpetuate gender stereotypes? (e.g., women = emotional/dramatic/nagging, men = stoic/dismissive)
   - Does the text use male-default language or exclusionary gendered terms?
   - Does the text suggest using products to avoid or "cancel" partners?
   - Does the text contain relationship stereotypes (e.g., "naggy wife", "ignoring husband")?
   - Does the text use gendered language that excludes non-binary individuals?
   - **PLAY ON WORDS & OBJECTIFICATION**: Does the text use double entendres, puns, or wordplay that sexualizes or objectifies individuals (especially women)? (e.g., "butt" products with suggestive phrasing, body part references with sexual undertones)
   - Does the text reduce people (especially women) to body parts or physical attributes?
   - Does the text use euphemisms or coded language for sexual objectification?

3. **TEXT-BASED AGE BIAS**:
   - Does the text contain ageist language or generational stereotypes?
   - Does the text assume age-related capabilities or interests?
   - Does the text use condescending language toward any age group?

4. **TEXT-BASED SOCIOECONOMIC BIAS**:
   - Does the text make class assumptions or use classist language?
   - Does the text exclude or stereotype based on economic status?
   - Does the text assume access to resources or privileges?

5. **TEXT-BASED DISABILITY BIAS**:
   - Does the text use ableist language (e.g., "crazy", "lame", "blind to")?
   - Does the text assume physical or mental abilities?
   - Does the text use disability as metaphor for negative traits?

6. **TEXT-BASED LGBTQ+ BIAS**:
   - Does the text make heteronormative assumptions?
   - Does the text exclude or stereotype LGBTQ+ individuals?
   - Does the text use language that marginalizes gender/sexual minorities?

7. **SUBTLE LINGUISTIC BIASES - PLAY ON WORDS & INNUENDO** (CRITICAL):
   - **Double Entendres**: Does the text use words/phrases with dual meanings where one is sexual/objectifying? (e.g., "butt cream" ads featuring women in suggestive poses, "size" references with sexual connotations)
   - **Body Part Objectification**: Does the text focus on body parts (especially women's bodies) as selling points? (e.g., "perfect butt", "curves", "assets")
   - **Sexualized Product Names/Slogans**: Does the text pair product names with celebrity endorsements or imagery that creates sexual associations?
   - **Innuendo & Coded Language**: Does the text use suggestive language that implies sexual content without being explicit?
   - **Context-Dependent Bias**: Consider the FULL context - a word like "butt" might be neutral alone, but becomes problematic when paired with certain imagery, celebrity names, or suggestive phrasing
   - **Celebrity Sexualization**: Does the text reference celebrities in ways that reduce them to physical attributes or sexual appeal?

**CRITICAL**: Be DIRECT and SPECIFIC about bias detection. Don't soften language - if "white = purity" appears, that's CRITICAL RACIAL BIAS. If "women = drama" appears, that's CRITICAL GENDER BIAS. If text uses wordplay to sexualize/objectify, that's GENDER BIAS through objectification.

**IMPORTANT FOR WORDPLAY DETECTION**:
- Look for puns, double meanings, and suggestive language
- Consider how words might be interpreted in sexual/objectifying contexts
- Analyze whether celebrities (especially women) are being used as sexual props
- Check if product names/descriptions have sexual undertones
- Evaluate if the language reduces individuals to physical attributes

For EACH bias you find, report it in this EXACT format:
---BIAS---
TYPE: [gender_bias|racial_bias|age_bias|socioeconomic_bias|disability_bias|lgbtq_bias]
SEVERITY: [low|medium|high|critical]
EXAMPLES: [quote the EXACT problematic words/phrases from the text, separated by ||| ]
CONTEXT: [2-3 sentence detailed explanation: What is the bias? Why is it problematic? For wordplay/innuendo, explain BOTH meanings. What harmful message does it send? Who is affected?]
IMPACT: [1-2 sentences describing the real-world harm this language perpetuates, especially for objectification/sexualization cases]
CONFIDENCE: [0.0-1.0]
---END---

**SEVERITY GUIDELINES**:
- CRITICAL: Explicit racial slurs, severe stereotypes, deeply harmful language, overt sexual objectification
- HIGH: Clear stereotyping, significant exclusionary language, harmful associations, sexualization through wordplay/innuendo, body part objectification
- MEDIUM: Implicit bias, coded language, subtle exclusion, mild double entendres
- LOW: Minor linguistic issues, easily correctable phrasing, unintentional ambiguous language

Start your response with either:
HAS_BIAS: YES
or
HAS_BIAS: NO

Then provide your analysis using the format above for each bias found.
End with:
SUMMARY: [2-3 sentence summary covering: What biases were found (including any wordplay/innuendo)? What are the main concerns? What groups are affected? For objectification cases, specify who is being sexualized/objectified.]"""

        user_prompt = f"""Analyze this advertising text for bias:

TEXT:
{text}

Provide detailed bias analysis using the structured format."""

        # Call Claude API with fast model (Haiku)
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast model
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Parse response
        response_text = message.content[0].text
        ctx.logger.info(f"   📝 LLM response received ({len(response_text)} chars)")
        
        # DEBUG: Log the actual LLM response
        ctx.logger.info(f"   🔍 DEBUG - LLM Response Content:")
        ctx.logger.info(f"   {response_text}")
        ctx.logger.info(f"   ----------------------------------------")
        
        # Parse the structured text response
        analysis = parse_structured_response(ctx, response_text)
        
        ctx.logger.info(f"✅ LLM analysis complete - Bias detected: {analysis.get('has_bias', False)}")
        ctx.logger.info(f"   📊 Found {len(analysis.get('bias_types', []))} bias instances")
        return analysis
        
    except Exception as e:
        ctx.logger.error(f"❌ Error in LLM analysis: {e}")
        import traceback
        ctx.logger.error(f"   Traceback: {traceback.format_exc()}")
        # Return safe fallback
        return {
            "has_bias": False,
            "bias_types": [],
            "overall_reasoning": f"Analysis failed due to error: {str(e)}"
        }


def parse_structured_response(ctx: Context, response_text: str) -> Dict[str, Any]:
    """
    Parse the structured text response from Claude into our analysis format.
    Much more reliable than JSON parsing.
    """
    try:
        # Check if bias was detected
        has_bias = "HAS_BIAS: YES" in response_text.upper()
        
        # Extract bias instances
        bias_types = []
        
        # Split by bias markers
        bias_sections = response_text.split("---BIAS---")
        
        for section in bias_sections[1:]:  # Skip first empty section
            if "---END---" not in section:
                continue
                
            # Extract the bias content
            bias_content = section.split("---END---")[0].strip()
            
            # Parse each field
            bias_instance = {
                "type": "gender_bias",  # default
                "severity": "medium",
                "examples": [],
                "context": "",
                "impact": "",
                "confidence": 0.75
            }
            
            for line in bias_content.split("\n"):
                line = line.strip()
                if line.startswith("TYPE:"):
                    bias_instance["type"] = line.replace("TYPE:", "").strip()
                elif line.startswith("SEVERITY:"):
                    bias_instance["severity"] = line.replace("SEVERITY:", "").strip()
                elif line.startswith("EXAMPLES:"):
                    examples_str = line.replace("EXAMPLES:", "").strip()
                    bias_instance["examples"] = [ex.strip() for ex in examples_str.split("|||") if ex.strip()]
                elif line.startswith("CONTEXT:"):
                    bias_instance["context"] = line.replace("CONTEXT:", "").strip()
                elif line.startswith("IMPACT:"):
                    bias_instance["impact"] = line.replace("IMPACT:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.replace("CONFIDENCE:", "").strip()
                        bias_instance["confidence"] = float(conf_str)
                    except ValueError:
                        bias_instance["confidence"] = 0.75
            
            # Only add if we have meaningful data
            if bias_instance["type"] and (bias_instance["examples"] or bias_instance["context"]):
                bias_types.append(bias_instance)
                ctx.logger.info(f"   📌 Parsed {bias_instance['type']} - {bias_instance['severity']}")
        
        # Extract summary
        overall_reasoning = "Analysis completed"
        if "SUMMARY:" in response_text:
            summary_start = response_text.find("SUMMARY:") + 8
            overall_reasoning = response_text[summary_start:].strip()
            # Take first line if multi-line
            if "\n" in overall_reasoning:
                overall_reasoning = overall_reasoning.split("\n")[0].strip()
        
        return {
            "has_bias": has_bias,
            "bias_types": bias_types,
            "overall_reasoning": overall_reasoning
        }
        
    except Exception as e:
        ctx.logger.error(f"❌ Error parsing structured response: {e}")
        import traceback
        ctx.logger.error(f"   Traceback: {traceback.format_exc()}")
        ctx.logger.debug(f"   Response text: {response_text[:500]}")
        
        # Return safe default
        return {
            "has_bias": False,
            "bias_types": [],
            "overall_reasoning": "Failed to parse analysis response"
        }


async def classify_and_extract_biases(
    ctx: Context,
    initial_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Classify bias types and extract specific examples from LLM analysis.
    Returns list of bias instance dicts formatted for the scoring agent.
    """
    ctx.logger.info(f"🏷️ Classifying bias types from LLM analysis...")

    detections = []
    
    # Check if bias was detected
    if not initial_analysis.get("has_bias", False):
        ctx.logger.info(f"   ℹ️ No bias detected in analysis")
        return detections
    
    # Map LLM bias types to BiasCategory enum
    bias_type_mapping = {
        "gender_bias": BiasCategory.GENDER,
        "racial_bias": BiasCategory.RACIAL,
        "age_bias": BiasCategory.AGE,
        "socioeconomic_bias": BiasCategory.SOCIOECONOMIC,
        "disability_bias": BiasCategory.DISABILITY,
        "lgbtq_bias": BiasCategory.LGBTQ
    }
    
    # Process each bias type from LLM analysis
    for bias_item in initial_analysis.get("bias_types", []):
        bias_type_str = bias_item.get("type", "")
        bias_category = bias_type_mapping.get(bias_type_str, BiasCategory.GENDER)
        
        detection = create_bias_instance_dict(
            bias_type=bias_category,
            severity=bias_item.get("severity", "medium"),
            examples=bias_item.get("examples", []),
            context=bias_item.get("context", "Bias detected by LLM analysis"),
            confidence=bias_item.get("confidence", 0.75)
        )
        
        # Add the impact field for richer context to scoring agent
        detection["impact"] = bias_item.get("impact", "This bias may perpetuate harmful stereotypes and exclude certain groups.")
        
        detections.append(detection)
        ctx.logger.info(f"   📌 Detected {bias_type_str} - Severity: {bias_item.get('severity', 'medium')}")

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


async def trigger_scoring_agent(ctx: Context, request_id: str, chromadb_collection_id: str):
    """
    Trigger Scoring Agent via REST to fetch and aggregate reports from Text and Visual agents.
    Uses HTTP POST instead of uAgents messaging.
    """
    scoring_agent_url = "http://localhost:8103/score"
    ctx.logger.info(f"   🔗 Calling: {scoring_agent_url}")
    ctx.logger.info(f"   📝 Request ID: {request_id}")
    
    try:
        scoring_payload = {
            "request_id": request_id,
            "chromadb_collection_id": chromadb_collection_id
        }
        
        async with aiohttp.ClientSession() as session:
            ctx.logger.info(f"   📡 Making POST request...")
            async with session.post(scoring_agent_url, json=scoring_payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                ctx.logger.info(f"   📥 Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    ctx.logger.info(f"   ✅ Scoring completed successfully!")
                    ctx.logger.info(f"   📊 Final Score: {result.get('overall_bias_score', 'N/A'):.1f}/10")
                    ctx.logger.info(f"   🏷️ Bias Level: {result.get('bias_level', 'N/A')}")
                    ctx.logger.info(f"   📋 Total Issues: {result.get('total_issues', 0)}")
                else:
                    ctx.logger.error(f"   ❌ Scoring request failed with status {response.status}")
                    error_text = await response.text()
                    ctx.logger.error(f"   📄 Error response: {error_text[:200]}")
    
    except aiohttp.ClientConnectorError as e:
        ctx.logger.error(f"   ❌ Cannot connect to Scoring Agent at {scoring_agent_url}")
        ctx.logger.error(f"      Make sure Scoring Agent is running on port 8103")
        ctx.logger.error(f"      Error: {e}")
    except asyncio.TimeoutError:
        ctx.logger.error(f"   ❌ Timeout calling Scoring Agent")
    except Exception as e:
        ctx.logger.error(f"   ❌ Error calling Scoring Agent: {e}")
        import traceback
        ctx.logger.error(f"      Traceback: {traceback.format_exc()}")


async def send_to_scoring_agent(ctx: Context, request_id: str, report: TextBiasReport):
    """
    Send analysis results to Scoring Agent.
    DEPRECATED: Use trigger_scoring_agent() with REST instead.
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
  ✓ Claude Haiku LLM integration for fast, intelligent analysis
  ✓ Detects gender, racial, age, socioeconomic bias
  ✓ Provides confidence scores and recommendations
  ✓ RESTful API for easy integration

Bias Types Detected:
  • Gender bias (stereotyping, exclusionary language)
  • Racial/ethnic bias (cultural appropriation, stereotypes)
  • Age bias (ageism, generational stereotypes)
  • Socioeconomic bias (class assumptions)
  • Disability bias (ableist language)
  • LGBTQ+ bias (heteronormative assumptions)

REST API Endpoints:
  • POST /analyze - Analyze text content for bias
    URL: http://localhost:8101/analyze
  • GET /report/{request_id} - Retrieve stored text bias report
    URL: http://localhost:8101/report/{request_id}

Architecture:
  • Analyzes text content and stores report in local storage
  • Triggers Scoring Agent via HTTP POST to aggregate final report
  • Scoring Agent fetches reports from Text/Visual agents via HTTP GET
  • Uses REST HTTP endpoints instead of uAgents messaging
  
Configuration:
  • Port: 8101
  • Endpoint: http://localhost:8101/submit

📍 Waiting for text analysis requests...
🛑 Stop with Ctrl+C
    """)
    text_bias_agent.run()

