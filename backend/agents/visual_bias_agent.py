"""
Visual Bias Agent - Ad Bias Detection System

Role: Visual Content Analysis and Bias Detection
Responsibilities:
- Analyze visual content for representation bias
- Generate bias analysis and recommendations
- Detect bias in subject representation, contextual placement, color usage
- Frame-by-frame analysis for video content
- Identify subtle visual cues and microaggressions
- Generate diversity metrics and recommendations

Vision Analysis Module using Claude Vision API

Based on Fetch.ai's image analysis example:
https://innovationlab.fetch.ai/resources/docs/next/examples/chat-protocol/image-analysis-agent

This module provides real visual bias detection using Anthropic's Claude Vision API.
"""

import json
import os
import base64
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import requests
from uagents import Agent, Context, Model
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

# Import shared models
try:
    from .shared_models import EmbeddingPackage, VisualBiasReport, BiasAnalysisComplete, BiasCategory
except ImportError:
    # Fallback for direct execution
    from shared_models import EmbeddingPackage, VisualBiasReport, BiasAnalysisComplete, BiasCategory


def repair_json(json_str: str) -> str:
    """
    Repair malformed JSON from Claude Vision API.
    Simple approach: escape all control characters within string contexts.
    """
    # Remove markdown code blocks
    json_str = re.sub(r'^```json\s*', '', json_str)
    json_str = re.sub(r'\s*```$', '', json_str)
    
    # Simple strategy: go through character by character
    # Track if we're in a string by counting unescaped quotes
    # Only escape control chars when inside strings
    
    result = []
    i = 0
    in_string = False
    prev_char = ''
    
    while i < len(json_str):
        char = json_str[i]
        
        # Check if this is an escape sequence
        if prev_char == '\\' and in_string:
            # This character is being escaped, just keep it
            result.append(char)
            prev_char = char if char != '\\' else ''  # Don't chain escapes
            i += 1
            continue
        
        if char == '"':
            # Toggle string state (unless it was escaped, which we handled above)
            in_string = not in_string
            result.append(char)
            prev_char = char
            i += 1
            continue
        
        if in_string:
            # We're inside a string - escape control characters
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif char == '\b':
                result.append('\\b')
            elif char == '\f':
                result.append('\\f')
            elif ord(char) < 32:
                # Other control characters
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            # Outside string, keep as-is
            result.append(char)
        
        prev_char = char
        i += 1
    
    return ''.join(result)


# Claude API Configuration
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY":
    print("⚠️  WARNING: ANTHROPIC_API_KEY not set. Vision analysis will use fallback mode.")
    print("   Set ANTHROPIC_API_KEY environment variable to enable Claude Vision API.")
    ANTHROPIC_API_KEY = None

MODEL_ENGINE = os.getenv("MODEL_ENGINE", "claude-3-5-haiku-latest")

HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
} if ANTHROPIC_API_KEY else {}


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Encode image file to base64 string.
    
    Returns:
        tuple: (base64_string, mime_type)
    """
    # Determine mime type from extension
    ext = image_path.lower().split('.')[-1]
    mime_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    mime_type = mime_type_map.get(ext, 'image/jpeg')
    
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string, mime_type


def analyze_image_for_bias(image_path: str, additional_context: str = "") -> Dict[str, Any]:
    """
    Analyze an image for potential bias using Claude Vision API.
    
    Args:
        image_path: Path to the image file
        additional_context: Additional context or specific aspects to analyze
    
    Returns:
        Dictionary containing bias analysis results
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(image_path)
    
    try:
        # Encode image
        image_base64, mime_type = encode_image_to_base64(image_path)
        
        # Construct the bias detection prompt
        prompt = f"""Analyze this advertising image for potential biases and representation issues.

**CRITICAL**: You MUST analyze any text visible in the image. Text messaging is a primary source of bias.

Please provide a detailed analysis covering:

1. **TEXT ANALYSIS** (HIGHEST PRIORITY):
   - What text is visible in the image? Quote it exactly.
   - Does the text contain any racial, gender, or other group associations?
   - Does the text associate certain colors (white, black, etc.) with value judgments (purity, cleanliness, etc.)?
   - Does the text contain stereotypes, assumptions, or exclusionary language?
   - Does the text contain any problematic messaging about identity, appearance, or groups?

2. **People Representation** (if people are visible):
   - How many people are visible?
   - What is their apparent gender distribution? (Be specific with counts)
   - What is their apparent ethnic/racial representation?
   - What is their apparent age distribution?
   - What body types are represented?
   - What roles or activities are they engaged in?

3. **Spatial Positioning & Power Dynamics**:
   - Who is positioned centrally vs. peripherally?
   - Who is in the foreground vs. background?
   - Who appears larger or more prominent?
   - Who has direct eye contact with the viewer?
   - Who appears active vs. passive?

4. **Context & Stereotyping**:
   - Are there any stereotypical representations?
   - Are certain groups shown only in specific roles?
   - Is there tokenism (minimal diverse representation)?
   - Are there any cultural appropriation concerns?

5. **Gender Dynamics & Relationship Stereotypes** (CRITICAL):
   - Does the text or visual content perpetuate gender stereotypes?
   - Are women portrayed as dramatic, emotional, nagging, or interrupting?
   - Are men portrayed as ignoring, dismissing, or needing to escape from women?
   - Does the ad suggest using the product to avoid, ignore, or "cancel" partners?
   - Are relationship dynamics portrayed in a healthy, respectful way?
   - Look for terms like: "drama", "nagging", "annoying girlfriend", "ignore her", etc.

6. **Overall Assessment**:
   - What potential biases do you detect?
   - What groups are overrepresented or underrepresented?
   - Are there any accessibility concerns (contrast, visibility)?

**IMPORTANT**: 
- Be direct about racial bias. If text associates "white" with positive traits (purity, cleanliness) or "black" with negative traits, this is CRITICAL racial bias.
- Be direct about gender bias. If text associates women with "drama", "nagging", or being a problem to solve/ignore, this is CRITICAL gender bias.
- Consider the CONTEXT: text + visuals together can create problematic messaging even if individual elements seem neutral.

{additional_context}

**CRITICAL - bias_detections array**:
The "bias_detections" field is the PRIMARY OUTPUT. It MUST be an array of OBJECTS, NOT strings.

WRONG: "bias_detections": ["Gender stereotyping", "Sexist narrative"]
RIGHT: "bias_detections": [{{"type": "gender", "severity": "high", "description": "...", "evidence": [...], "affected_groups": [...]}}]

Each object in bias_detections MUST have exactly these 5 fields:
1. "type": string (one of: representation, contextual, tokenism, stereotyping, cultural_appropriation, gender)
2. "severity": string (one of: low, medium, high, critical)
3. "description": string (detailed description of the bias)
4. "evidence": array of strings (specific visual or text evidence)
5. "affected_groups": array of strings (groups harmed)

If you identify ANY biases, they MUST be in this object format.

Respond in JSON format with this EXACT structure:
{{
    "text_detected": {{
        "has_text": <true/false>,
        "text_content": "<exact text visible in the image>",
        "text_analysis": "<analysis of problematic messaging, associations, or bias in the text>",
        "racial_color_associations": "<any problematic associations between colors and racial/value judgments>"
    }},
    "people_detected": {{
        "total_count": <number>,
        "visible_demographics": {{
            "gender": {{"male": <count>, "female": <count>, "non_binary": <count>, "unknown": <count>}},
            "ethnicity": {{"white": <count>, "black": <count>, "asian": <count>, "hispanic": <count>, "middle_eastern": <count>, "mixed": <count>, "unknown": <count>}},
            "age_groups": {{"children": <count>, "young_adult": <count>, "middle_aged": <count>, "senior": <count>, "unknown": <count>}},
            "body_types": {{"slim": <count>, "average": <count>, "athletic": <count>, "plus_size": <count>, "unknown": <count>}}
        }}
    }},
    "spatial_analysis": {{
        "power_positioning": "<description>",
        "central_subjects": ["<demographic description>"],
        "peripheral_subjects": ["<demographic description>"],
        "prominence_analysis": "<who appears most prominent and why>"
    }},
    "gender_dynamics": {{
        "has_gender_bias": <true/false>,
        "relationship_dynamics": "<description of relationship portrayals>",
        "gender_stereotypes": ["<list of stereotypes detected>"],
        "problematic_messaging": "<description of any problematic gender-based messaging>"
    }},
    "bias_detections": [
        {{
            "type": "gender",
            "severity": "high",
            "description": "Women portrayed as overly dramatic or emotional",
            "evidence": ["Text says 'Drama: off'", "Visual shows woman speaking"],
            "affected_groups": ["Women", "Female partners"]
        }},
        {{
            "type": "contextual",
            "severity": "critical",
            "description": "Technology used to silence or ignore a person",
            "evidence": ["Noise cancelling headphones blocking communication"],
            "affected_groups": ["Women", "Partners in relationships"]
        }}
    ],
    "accessibility_issues": [
        {{
            "type": "<contrast|visibility|color_blindness>",
            "severity": "<low|medium|high>",
            "description": "<description>"
        }}
    ],
    "overall_assessment": {{
        "bias_score": <0-10, where 0=severe bias, 10=no bias>,
        "diversity_score": <0-10, where 0=no diversity, 10=excellent diversity>,
        "main_concerns": ["<primary concerns>"],
        "positive_aspects": ["<positive aspects>"]
    }}
}}

Be objective and specific. Provide actual counts and observations. If you identify ANY bias, it MUST appear in bias_detections as an object with all 5 required fields.

**CRITICAL JSON FORMATTING**:
- NEVER use double quotes inside string values - use single quotes or paraphrase instead
- Example: WRONG: "text says \\"Drama\\"" RIGHT: "text says 'Drama'" or "text references drama"
- All string values must be valid JSON with proper escaping
- Respond with ONLY the JSON object, no additional text"""

        # Prepare content for Claude
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_base64,
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Make API request with prefill for guaranteed JSON
        data = {
            "model": MODEL_ENGINE,
            "max_tokens": MAX_TOKENS,
            "system": "You are a bias detection AI. You respond ONLY with valid JSON matching the exact schema provided. No other text.",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        }
        
        response = requests.post(
            CLAUDE_URL, 
            headers=HEADERS, 
            data=json.dumps(data), 
            timeout=120
        )
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        # Handle error responses
        if "error" in response_data:
            return {
                "error": f"API Error: {response_data['error'].get('message', 'Unknown error')}",
                "fallback": True
            }
        
        messages = response_data.get("content", [])
        if messages and len(messages) > 0:
            text_response = messages[0].get("text", "")
            
            # Try to parse JSON from response (prefilled with "{")
            try:
                # We prefilled with "{", so prepend it back
                json_str = "{" + text_response.strip()
                
                # Try to parse - if it fails, use repair function
                try:
                    analysis_result = json.loads(json_str)
                    print(f"✅ JSON parsed successfully on first try!")
                except json.JSONDecodeError as parse_err:
                    print(f"⚠️  Initial parse failed: {parse_err}")
                    print(f"   Error at: {json_str[max(0, parse_err.pos-50):parse_err.pos+50]}")
                    print(f"🔧 Attempting to repair JSON with character-by-character fixing...")
                    
                    # Use aggressive JSON repair function
                    repaired_json = repair_json(json_str)
                    
                    # Try parsing the repaired JSON
                    try:
                        analysis_result = json.loads(repaired_json)
                        print(f"✅ JSON repaired and parsed successfully!")
                    except json.JSONDecodeError as second_err:
                        print(f"❌ Could not repair JSON: {second_err}")
                        print(f"   Original length: {len(json_str)}")
                        print(f"   Repaired length: {len(repaired_json)}")
                        print(f"   Error at position {second_err.pos}")
                        if second_err.pos < len(repaired_json):
                            print(f"   Context: {repaired_json[max(0, second_err.pos-100):second_err.pos+100]}")
                        return {
                            "raw_analysis": text_response,
                            "api_used": "claude_vision",
                            "model": MODEL_ENGINE,
                            "note": "Could not parse structured JSON, returning raw analysis",
                            "parse_error": str(second_err),
                            "original_error": str(parse_err)
                        }
                
                analysis_result["api_used"] = "claude_vision"
                analysis_result["model"] = MODEL_ENGINE
                
                # FIX: Convert string bias_detections to objects if Claude ignored schema
                if "bias_detections" in analysis_result:
                    bias_detections = analysis_result["bias_detections"]
                    if isinstance(bias_detections, list) and len(bias_detections) > 0:
                        # Check if first item is a string (Claude ignored schema)
                        if isinstance(bias_detections[0], str):
                            print(f"⚠️  CONVERTING: Claude returned strings, converting to objects...")
                            fixed_detections = []
                            for bias_str in bias_detections:
                                fixed_detections.append({
                                    "type": "gender" if any(word in bias_str.lower() for word in ["gender", "women", "drama"]) else "contextual",
                                    "severity": "high" if any(word in bias_str.lower() for word in ["sexist", "critical"]) else "medium",
                                    "description": bias_str,
                                    "evidence": analysis_result.get("text_detected", {}).get("text_content", "").split("\n") if analysis_result.get("text_detected") else [],
                                    "affected_groups": ["Women"] if "women" in bias_str.lower() else ["General audience"]
                                })
                            analysis_result["bias_detections"] = fixed_detections
                            print(f"✅ Converted {len(fixed_detections)} string detections to objects")
                
                # Log the full response for debugging
                print("=" * 80)
                print("🔍 CLAUDE VISION RAW RESPONSE:")
                print("=" * 80)
                print(json.dumps(analysis_result, indent=2))
                print("=" * 80)
                print(f"📊 Bias detections count: {len(analysis_result.get('bias_detections', []))}")
                if analysis_result.get('bias_detections'):
                    for i, bias in enumerate(analysis_result['bias_detections'], 1):
                        print(f"   {i}. Type: {bias.get('type')}, Severity: {bias.get('severity')}")
                        print(f"      Description: {bias.get('description', '')[:100]}...")
                print("=" * 80)
                
                return analysis_result
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                return {
                    "raw_analysis": text_response,
                    "api_used": "claude_vision",
                    "model": MODEL_ENGINE,
                    "note": "Could not parse structured JSON, returning raw analysis"
                }
        else:
            return {"error": "No response from API"}
            
    except requests.exceptions.Timeout:
        return {"error": "The request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def _fallback_analysis(image_path: str) -> Dict[str, Any]:
    """
    Fallback analysis when API key is not available.
    Performs basic image analysis without Vision-LLM.
    """
    try:
        return {
            "fallback": True,
            "error": "ANTHROPIC_API_KEY not configured",
            "message": "Set ANTHROPIC_API_KEY environment variable to enable real bias detection",
            "basic_analysis": {
                "image_path": image_path,
                "analysis_type": "fallback_mode"
            },
            "recommendation": "Configure Claude Vision API key for demographic and bias detection"
        }
    except Exception as e:
        return {
            "fallback": True,
            "error": str(e)
        }


# Helper function for batch analysis
def analyze_multiple_frames(frame_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze multiple video frames by sending ALL frames to Claude in ONE request.
    This allows Claude to understand temporal context and patterns across the video.
    
    Args:
        frame_paths: List of image file paths
    
    Returns:
        Analysis across all frames with temporal context
    """
    if not frame_paths:
        return {"error": "No frames provided"}
    
    if not ANTHROPIC_API_KEY:
        return _fallback_analysis(frame_paths[0])
    
    print(f"🎬 Analyzing {len(frame_paths)} video frames in ONE request for temporal context...")
    print(f"📍 ANTHROPIC_API_KEY present: {bool(ANTHROPIC_API_KEY)}")
    print(f"📍 API URL: {CLAUDE_URL}")
    print(f"📍 Model: {MODEL_ENGINE}")
    
    try:
        # Encode ALL frames
        frame_images = []
        print(f"🔄 Encoding {len(frame_paths)} frames...")
        for i, frame_path in enumerate(frame_paths):
            print(f"   Encoding frame {i+1}: {frame_path}")
            image_base64, mime_type = encode_image_to_base64(frame_path)
            frame_images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_base64,
                }
            })
        print(f"✅ All {len(frame_images)} frames encoded")
        
        # Construct prompt for video analysis
        print(f"🔄 Constructing prompt...")
        prompt = f"""Analyze these {len(frame_paths)} frames from a video advertisement for potential biases and representation issues.

**CRITICAL**: These frames are from the SAME video ad. Look for patterns, narrative, and how the message develops across time.

Please provide a detailed analysis covering:

1. **TEXT ANALYSIS** (HIGHEST PRIORITY):
   - What text is visible? Quote it exactly.
   - Does the text contain any racial, gender, or other group associations?
   - Does the text associate certain colors (white, black, etc.) with value judgments?
   - Does the text contain stereotypes or exclusionary language?

2. **People Representation**:
   - How many people are visible?
   - Gender distribution, ethnicity, age, body types?
   - What roles or activities are they engaged in?

3. **Spatial Positioning & Power Dynamics**:
   - Who is positioned centrally vs. peripherally?
   - Who appears more prominent?

4. **Context & Stereotyping**:
   - Any stereotypical representations?
   - Tokenism or cultural appropriation?

5. **Gender Dynamics & Relationship Stereotypes** (CRITICAL):
   - Does the content perpetuate gender stereotypes?
   - Are women portrayed as dramatic, emotional, or nagging?
   - Are men portrayed as ignoring or dismissing women?
   - Does it suggest using the product to avoid or "cancel" partners?

6. **Overall Assessment**:
   - What potential biases do you detect?
   - What groups are affected?

**IMPORTANT**: 
- Be direct about racial bias. "White = purity" is CRITICAL racial bias.
- Be direct about gender bias. "Women = drama" is CRITICAL gender bias.
- Consider CONTEXT: text + visuals together create the message.

Since this is a VIDEO, also consider:
- How does the narrative or message progress across frames?
- Are there temporal patterns in how people are portrayed?
- Does the sequence of frames reinforce or challenge stereotypes?
- Does the temporal sequence show one person "canceling" or ignoring another?

**CRITICAL JSON FORMATTING**: 
1. Respond in the SAME JSON format as above
2. Your analysis should consider ALL frames together, not individually
3. The "bias_detections" array is MANDATORY and MUST contain ALL biases you identify as OBJECTS (not strings)
4. If frames show text like 'Drama: off' with a woman speaking, this MUST be in bias_detections as gender bias
5. NO conversational text - ONLY the JSON object
6. NEVER use double quotes inside string values - use single quotes or paraphrase instead
7. Example: WRONG: ["text says \\"Drama: off\\""] RIGHT: ["text says 'Drama: off'"] or ["text references drama being turned off"]
8. All string values must be valid JSON strings with proper escaping"""

        # Build content array: alternating images and text
        print(f"🔄 Building API request content...")
        content = []
        for i, frame_img in enumerate(frame_images):
            content.append(frame_img)
            if i == 0:
                # Add the prompt after the first image
                content.append({"type": "text", "text": prompt})
        print(f"✅ Content array built with {len(content)} items")
        
        # Make API request with ALL frames + prefill for guaranteed JSON
        print(f"🔄 Preparing API request...")
        
        data = {
            "model": MODEL_ENGINE,
            "max_tokens": MAX_TOKENS,
            "system": "You are a bias detection AI. You respond ONLY with valid JSON matching the exact schema provided. No other text.",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        }
        print(f"✅ Request payload prepared")
        print(f"📤 Making POST request to Claude API...")
        print(f"   URL: {CLAUDE_URL}")
        print(f"   Model: {MODEL_ENGINE}")
        print(f"   Max tokens: {MAX_TOKENS}")
        print(f"   Timeout: 180s")
        
        response = requests.post(
            CLAUDE_URL, 
            headers=HEADERS, 
            data=json.dumps(data), 
            timeout=180  # Longer timeout for multiple frames
        )
        
        print(f"✅ Response received: HTTP {response.status_code}")
        print(f"📊 Response size: {len(response.content)} bytes")
        
        response.raise_for_status()
        
        # Parse response
        print(f"🔄 Parsing JSON response...")
        response_data = response.json()
        print(f"✅ JSON parsed successfully")
        
        if "error" in response_data:
            print(f"❌ API returned error: {response_data['error']}")
            return {"error": f"API Error: {response_data['error'].get('message', 'Unknown error')}"}
        
        messages = response_data.get("content", [])
        print(f"📨 Response contains {len(messages)} message(s)")
        
        if messages and len(messages) > 0:
            text_response = messages[0].get("text", "")
            print(f"📄 Text response length: {len(text_response)} characters")
            print(f"📄 First 500 chars of response: {text_response[:500]}")
            
            try:
                print(f"🔄 Extracting JSON from response...")
                # We prefilled with "{", so prepend it back
                json_str = "{" + text_response.strip()
                
                print(f"📊 JSON string length: {len(json_str)} characters")
                print(f"🔄 Parsing JSON...")
                
                # Try to parse - if it fails, use repair function
                try:
                    analysis_result = json.loads(json_str)
                    print(f"✅ JSON parsed successfully on first try!")
                except json.JSONDecodeError as parse_err:
                    print(f"⚠️  Initial parse failed: {parse_err}")
                    print(f"   Error at position {parse_err.pos}: {parse_err.msg}")
                    if parse_err.pos < len(json_str):
                        print(f"   Context: {json_str[max(0, parse_err.pos-50):min(len(json_str), parse_err.pos+50)]}")
                    print(f"🔧 Attempting to repair JSON with character-by-character fixing...")
                    
                    # Use aggressive JSON repair function
                    repaired_json = repair_json(json_str)
                    print(f"📊 Repaired JSON length: {len(repaired_json)} characters")
                    
                    # Try parsing the repaired JSON
                    try:
                        analysis_result = json.loads(repaired_json)
                        print(f"✅ JSON repaired and parsed successfully!")
                    except json.JSONDecodeError as second_err:
                        print(f"❌ Could not repair JSON: {second_err}")
                        print(f"   Error at position {second_err.pos}: {second_err.msg}")
                        if second_err.pos < len(repaired_json):
                            print(f"   Error context: {repaired_json[max(0, second_err.pos-100):min(len(repaired_json), second_err.pos+100)]}")
                        # Return raw response with error details
                        return {
                            "raw_analysis": text_response,
                            "api_used": "claude_vision",
                            "model": MODEL_ENGINE,
                            "note": "Could not parse structured JSON, returning raw analysis",
                            "parse_error": str(second_err),
                            "original_error": str(parse_err),
                            "repair_attempted": True
                        }
                
                analysis_result["api_used"] = "claude_vision"
                analysis_result["model"] = MODEL_ENGINE
                analysis_result["frames_analyzed"] = len(frame_paths)
                analysis_result["analysis_mode"] = "multi_frame_temporal"
                
                # FIX: Convert string bias_detections to objects if Claude ignored schema
                if "bias_detections" in analysis_result:
                    bias_detections = analysis_result["bias_detections"]
                    if isinstance(bias_detections, list) and len(bias_detections) > 0:
                        # Check if first item is a string (Claude ignored schema)
                        if isinstance(bias_detections[0], str):
                            print(f"⚠️  CONVERTING: Claude returned strings, converting to objects...")
                            fixed_detections = []
                            
                            for bias_str in bias_detections:
                                # Determine type based on content
                                bias_type = "contextual"
                                if any(word in bias_str.lower() for word in ["gender", "women", "female", "male", "sexist", "misogyn", "drama"]):
                                    bias_type = "gender"
                                elif any(word in bias_str.lower() for word in ["beauty", "body", "appearance", "diversity", "representation"]):
                                    bias_type = "representation"
                                elif any(word in bias_str.lower() for word in ["youth", "age", "young", "old"]):
                                    bias_type = "stereotyping"
                                elif any(word in bias_str.lower() for word in ["racial", "ethnic", "race"]):
                                    bias_type = "racial"
                                
                                # Determine severity - be more aggressive about gender/sexist bias
                                severity = "medium"  # Default to medium instead of low
                                if any(word in bias_str.lower() for word in ["implicit", "subtle", "minor"]):
                                    severity = "low"
                                if any(word in bias_str.lower() for word in ["problematic", "stereotyp", "reinfor"]):
                                    severity = "medium"
                                if any(word in bias_str.lower() for word in ["critical", "severe", "extreme", "sexist", "racist", "misogyn", "explicit"]):
                                    severity = "high"
                                
                                # Build evidence from available data
                                evidence = []
                                
                                # Add text evidence if available
                                if analysis_result.get("text_detected") and "text_content" in analysis_result["text_detected"]:
                                    evidence.append(f"Text: {analysis_result['text_detected']['text_content']}")
                                
                                if not evidence:
                                    evidence = [bias_str]
                                
                                # Determine affected groups
                                affected = []
                                if any(word in bias_str.lower() for word in ["women", "female", "drama", "emotion"]):
                                    affected.append("Women")
                                if "body" in bias_str.lower() or "diversity" in bias_str.lower():
                                    affected.append("Body diversity")
                                if "youth" in bias_str.lower() or "age" in bias_str.lower():
                                    affected.append("Age groups")
                                if not affected:
                                    affected = ["General audience"]
                                
                                fixed_detections.append({
                                    "type": bias_type,
                                    "severity": severity,
                                    "description": bias_str,
                                    "evidence": evidence,
                                    "affected_groups": affected
                                })
                            
                            analysis_result["bias_detections"] = fixed_detections
                            print(f"✅ Converted {len(fixed_detections)} string detections to proper format")
                
                # Log the response
                print("=" * 80)
                print(f"🔍 CLAUDE VISION VIDEO ANALYSIS ({len(frame_paths)} frames):")
                print("=" * 80)
                print(json.dumps(analysis_result, indent=2))
                print("=" * 80)
                print(f"📊 Bias detections count: {len(analysis_result.get('bias_detections', []))}")
                if analysis_result.get('bias_detections'):
                    for i, bias in enumerate(analysis_result['bias_detections'], 1):
                        # Handle both object and string formats (safety)
                        if isinstance(bias, dict):
                            print(f"   {i}. Type: {bias.get('type')}, Severity: {bias.get('severity')}")
                            print(f"      Description: {bias.get('description', '')[:100]}...")
                        else:
                            print(f"   {i}. (String format - not expected): {str(bias)[:100]}...")
                else:
                    print(f"   ⚠️  NO BIAS DETECTIONS IN RESPONSE!")
                print("=" * 80)
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"   Attempted to parse: {json_str[:200]}...")
                return {
                    "raw_analysis": text_response,
                    "api_used": "claude_vision",
                    "model": MODEL_ENGINE,
                    "note": "Could not parse structured JSON, returning raw analysis",
                    "parse_error": str(e)
                }
        else:
            print(f"❌ No messages in API response!")
            return {"error": "No response from API"}
            
    except requests.exceptions.Timeout as e:
        print(f"❌ TIMEOUT: Request timed out after 180s")
        print(f"   Error: {e}")
        return {"error": "The request timed out. Please try again.", "fallback": True}
    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return {"error": f"An error occurred: {e}", "fallback": True}
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return {"error": f"Unexpected error: {e}", "fallback": True}


# Initialize Visual Bias Agent
visual_bias_agent = Agent(
    name="visual_bias_agent",
    seed="ad_bias_visual_agent_unique_seed_2024",
    port=8102,
    endpoint=["http://localhost:8102/analyze"]
)


@visual_bias_agent.on_message(model=EmbeddingPackage, replies=VisualBiasReport)
async def handle_visual_analysis(ctx: Context, sender: str, msg: EmbeddingPackage):
    """
    Handle visual bias analysis requests.
    """
    ctx.logger.info(f"🎬 Visual bias analysis request received from {sender}")
    ctx.logger.info(f"   📋 Request ID: {msg.request_id}")
    ctx.logger.info(f"   📁 Content type: {msg.content_type}")
    
    try:
        # Check if we have frames to analyze
        has_frames = msg.frames_base64 and len(msg.frames_base64) > 0
        
        if not has_frames:
            ctx.logger.warning("⚠️ No frames provided for analysis")
            return VisualBiasReport(
                request_id=msg.request_id,
                bias_detected=False,
                bias_instances=[],
                overall_visual_score=5.0,
                diversity_metrics={
                    "gender_distribution": {"male": 0.5, "female": 0.5, "non-binary": 0.0, "unknown": 0.0},
                    "apparent_ethnicity": {"white": 0.8, "black": 0.1, "asian": 0.05, "hispanic": 0.05, "unknown": 0.0},
                    "age_distribution": {"young_adult": 0.4, "middle_aged": 0.5, "senior": 0.1, "unknown": 0.0},
                    "body_type_diversity": 0.5,
                    "power_dynamics_score": 0.5
                },
                recommendations=["Provide video frames for analysis"]
            )
        
        # Analyze frames
        ctx.logger.info(f"🔄 Analyzing {len(msg.frames_base64)} frames...")
        
        # For now, we'll analyze the first frame as a test
        # In production, you'd want to analyze all frames
        first_frame_data = msg.frames_base64[0]
        
        # Extract base64 data (remove data:image/jpeg;base64, prefix)
        if first_frame_data.startswith('data:'):
            base64_data = first_frame_data.split(',')[1]
        else:
            base64_data = first_frame_data
        
        # Decode and save temporarily for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_file.write(base64.b64decode(base64_data))
            temp_path = temp_file.name
        
        try:
            # Analyze the frame
            analysis_result = analyze_image_for_bias(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Process the analysis result
            bias_instances = []
            if "bias_detections" in analysis_result:
                for bias in analysis_result["bias_detections"]:
                    if isinstance(bias, dict):
                        bias_instances.append({
                            "bias_type": bias.get("type", "representation"),
                            "severity": bias.get("severity", "medium"),
                            "examples": bias.get("evidence", []),
                            "context": bias.get("description", ""),
                            "impact": f"This visual bias affects {', '.join(bias.get('affected_groups', ['certain groups']))} and may perpetuate harmful stereotypes or exclusionary practices.",
                            "confidence": 0.8
                        })
            
            # Calculate diversity metrics from analysis
            people_detected = analysis_result.get("people_detected", {})
            demographics = people_detected.get("visible_demographics", {})
            
            gender_dist = demographics.get("gender", {})
            total_people = people_detected.get("total_count", 0)
            
            gender_distribution = {}
            if total_people > 0:
                for gender, count in gender_dist.items():
                    gender_distribution[gender] = count / total_people
            else:
                gender_distribution = {"male": 0.5, "female": 0.5, "non-binary": 0.0, "unknown": 0.0}
            
            ethnicity_dist = demographics.get("ethnicity", {})
            ethnicity_distribution = {}
            if total_people > 0:
                for ethnicity, count in ethnicity_dist.items():
                    ethnicity_distribution[ethnicity] = count / total_people
            else:
                ethnicity_distribution = {"white": 0.8, "black": 0.1, "asian": 0.05, "hispanic": 0.05, "unknown": 0.0}
            
            age_dist = demographics.get("age_groups", {})
            age_distribution = {}
            if total_people > 0:
                for age_group, count in age_dist.items():
                    age_distribution[age_group] = count / total_people
            else:
                age_distribution = {"young_adult": 0.4, "middle_aged": 0.5, "senior": 0.1, "unknown": 0.0}
            
            # Calculate body type diversity
            body_types = demographics.get("body_types", {})
            unique_body_types = len([bt for bt, count in body_types.items() if count > 0])
            body_type_diversity = min(1.0, unique_body_types / 4.0)  # Normalize to 0-1
            
            # Calculate power dynamics score
            spatial_analysis = analysis_result.get("spatial_analysis", {})
            power_positioning = spatial_analysis.get("power_positioning", "")
            power_dynamics_score = 0.5  # Default neutral
            if 'balanced' in power_positioning.lower() or 'equal' in power_positioning.lower():
                power_dynamics_score = 0.8
            elif 'dominant' in power_positioning.lower() or 'central' in power_positioning.lower():
                power_dynamics_score = 0.3
            
            diversity_metrics = {
                "gender_distribution": gender_distribution,
                "apparent_ethnicity": ethnicity_distribution,
                "age_distribution": age_distribution,
                "body_type_diversity": body_type_diversity,
                "power_dynamics_score": power_dynamics_score
            }
            
            # Calculate visual score
            overall_assessment = analysis_result.get("overall_assessment", {})
            visual_score = overall_assessment.get("bias_score", 5.0)
            
            # Generate recommendations
            recommendations = []
            if bias_instances:
                recommendations.append("Review visual content for bias patterns")
                recommendations.append("Consider diverse representation")
            else:
                recommendations.append("Content appears balanced")
            
            # Create comprehensive report
            report = VisualBiasReport(
                request_id=msg.request_id,
                bias_detected=len(bias_instances) > 0,
                bias_instances=bias_instances,
                overall_visual_score=visual_score,
                diversity_metrics=diversity_metrics,
                recommendations=recommendations
            )
            
            # Log the full report for console display
            ctx.logger.info("=" * 80)
            ctx.logger.info("📊 VISUAL BIAS ANALYSIS REPORT:")
            ctx.logger.info("=" * 80)
            ctx.logger.info(json.dumps({
                "request_id": report.request_id,
                "bias_detected": report.bias_detected,
                "bias_count": len(report.bias_instances) if report.bias_instances else 0,
                "visual_score": report.overall_visual_score,
                "diversity_metrics": report.diversity_metrics,
                "recommendations": report.recommendations,
                "claude_analysis": analysis_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, indent=2))
            ctx.logger.info("=" * 80)
            
            # CRITICAL FIX: Store report for Scoring Agent retrieval via REST GET
            report_dict = {
                "request_id": report.request_id,
                "agent_name": report.agent_name,
                "bias_detected": report.bias_detected,
                "bias_instances": report.bias_instances,
                "overall_visual_score": report.overall_visual_score,
                "diversity_metrics": report.diversity_metrics,
                "recommendations": report.recommendations,
                "timestamp": report.timestamp
            }
            report_key = f"visual_report_{msg.request_id}"
            ctx.storage.set(report_key, report_dict)
            ctx.logger.info(f"💾 Visual report stored with key: {report_key}")
            ctx.logger.info(f"🌐 Scoring Agent can retrieve via: GET /report/{msg.request_id}")
            
            return report
            
        except Exception as e:
            ctx.logger.error(f"❌ Error analyzing frame: {e}")
            # Clean up temp file if it exists
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return VisualBiasReport(
                request_id=msg.request_id,
                bias_detected=False,
                bias_instances=[],
                overall_visual_score=5.0,
                diversity_metrics={
                    "gender_distribution": {"male": 0.5, "female": 0.5, "non-binary": 0.0, "unknown": 0.0},
                    "apparent_ethnicity": {"white": 0.8, "black": 0.1, "asian": 0.05, "hispanic": 0.05, "unknown": 0.0},
                    "age_distribution": {"young_adult": 0.4, "middle_aged": 0.5, "senior": 0.1, "unknown": 0.0},
                    "body_type_diversity": 0.5,
                    "power_dynamics_score": 0.5
                },
                recommendations=["Error occurred during analysis"]
            )
    
    except Exception as e:
        ctx.logger.error(f"❌ Error processing visual analysis request: {e}")
        return VisualBiasReport(
            request_id=msg.request_id,
            bias_detected=False,
            bias_instances=[],
            overall_visual_score=5.0,
            diversity_metrics={
                "gender_distribution": {"male": 0.5, "female": 0.5, "non-binary": 0.0, "unknown": 0.0},
                "apparent_ethnicity": {"white": 0.8, "black": 0.1, "asian": 0.05, "hispanic": 0.05, "unknown": 0.0},
                "age_distribution": {"young_adult": 0.4, "middle_aged": 0.5, "senior": 0.1, "unknown": 0.0},
                "body_type_diversity": 0.5,
                "power_dynamics_score": 0.5
            },
            recommendations=["System error occurred"]
        )


@visual_bias_agent.on_rest_get("/report/{request_id}", BiasAnalysisComplete)
async def get_visual_report(ctx: Context, request_id: str) -> BiasAnalysisComplete:
    """
    REST GET endpoint for Scoring Agent to retrieve stored visual bias report.
    
    Usage: GET http://localhost:8102/report/{request_id}
    """
    ctx.logger.info(f"📨 REST GET request for report: {request_id}")
    
    # Retrieve from storage
    report_key = f"visual_report_{request_id}"
    report_data = ctx.storage.get(report_key)
    
    if report_data:
        ctx.logger.info(f"✅ Visual bias report found and returned")
        # Convert dict back to BiasAnalysisComplete
        return BiasAnalysisComplete(
            request_id=request_id,
            sender_agent="visual_bias_agent",
            report=report_data
        )
    else:
        ctx.logger.warning(f"⚠️ Visual bias report not found for request_id: {request_id}")
        # Return a "not found" response
        return BiasAnalysisComplete(
            request_id=request_id,
            sender_agent="visual_bias_agent",
            report={
                "request_id": request_id,
                "agent_name": "visual_bias_agent",
                "error": "Report not found. Analysis may still be in progress or request_id is invalid.",
                "bias_detected": False,
                "overall_visual_score": 0.0,
                "recommendations": []
            }
        )


@visual_bias_agent.on_rest_post("/analyze", EmbeddingPackage, BiasAnalysisComplete)
async def handle_visual_analysis_rest(ctx: Context, request: EmbeddingPackage):
    """
    REST endpoint handler for visual bias analysis.
    """
    ctx.logger.info(f"🌐 REST request received for visual analysis: {request.request_id}")
    
    # Process the analysis using the same handler
    result = await handle_visual_analysis(ctx, "rest_client", request)
    
    # Create report dict for storage and response
    report_dict = {
        "request_id": result.request_id,
        "agent_name": result.agent_name,
        "bias_detected": result.bias_detected,
        "bias_instances": result.bias_instances,
        "overall_visual_score": result.overall_visual_score,
        "diversity_metrics": result.diversity_metrics,
        "recommendations": result.recommendations,
        "claude_analysis": getattr(result, 'claude_analysis', {}),
        "timestamp": result.timestamp
    }
    
    # Store report for Scoring Agent retrieval
    report_key = f"visual_report_{request.request_id}"
    ctx.storage.set(report_key, report_dict)
    ctx.logger.info(f"💾 Visual report stored with key: {report_key}")
    ctx.logger.info(f"🌐 Scoring Agent can retrieve via: GET /report/{request.request_id}")
    
    # Convert to BiasAnalysisComplete format
    return BiasAnalysisComplete(
        request_id=request.request_id,
        sender_agent="visual_bias_agent",
        report=report_dict
    )


if __name__ == "__main__":
    visual_bias_agent.run()