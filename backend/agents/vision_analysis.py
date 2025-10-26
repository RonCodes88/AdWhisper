"""
Vision Analysis Module using Claude Vision API

Based on Fetch.ai's image analysis example:
https://innovationlab.fetch.ai/resources/docs/next/examples/chat-protocol/image-analysis-agent

This module provides real visual bias detection using Anthropic's Claude Vision API.
"""

import json
import os
import base64
import re
from typing import Dict, Any, List
import requests


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
                                    "type": "gender" if "gender" in bias_str.lower() or "women" in bias_str.lower() or "drama" in bias_str.lower() else "contextual",
                                    "severity": "high" if any(word in bias_str.lower() for word in ["sexist", "misogyn", "critical"]) else "medium",
                                    "description": bias_str,
                                    "evidence": analysis_result.get("text_detected", {}).get("text_content", "").split("\n") if analysis_result.get("text_detected") else [],
                                    "affected_groups": ["Women"] if "women" in bias_str.lower() or "gender" in bias_str.lower() else ["General audience"]
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
    from PIL import Image
    import numpy as np
    
    try:
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        height, width, _ = img_array.shape
        brightness = float(np.mean(img_array))
        
        return {
            "fallback": True,
            "error": "ANTHROPIC_API_KEY not configured",
            "message": "Set ANTHROPIC_API_KEY environment variable to enable real bias detection",
            "basic_analysis": {
                "image_size": f"{width}x{height}",
                "brightness": brightness,
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

{_get_bias_analysis_prompt()}

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
                
                # FIX: Handle nested bias_detections in video_analysis OR bias_analysis
                if "video_analysis" in analysis_result and "bias_detections" in analysis_result["video_analysis"]:
                    print(f"🔧 Moving nested bias_detections from video_analysis to top level...")
                    analysis_result["bias_detections"] = analysis_result["video_analysis"]["bias_detections"]
                elif "bias_analysis" in analysis_result and "bias_detections" in analysis_result["bias_analysis"]:
                    print(f"🔧 Moving nested bias_detections from bias_analysis to top level...")
                    analysis_result["bias_detections"] = analysis_result["bias_analysis"]["bias_detections"]
                    # Also extract other useful data from bias_analysis
                    if "text_analysis" in analysis_result["bias_analysis"]:
                        analysis_result["text_detected"] = analysis_result["bias_analysis"]["text_analysis"]
                    if "representation" in analysis_result["bias_analysis"]:
                        analysis_result["people_detected"] = {
                            "total_count": sum(analysis_result["bias_analysis"]["representation"].get("gender_distribution", {}).values()),
                            "visible_demographics": analysis_result["bias_analysis"]["representation"]
                        }
                    if "gender_dynamics" in analysis_result["bias_analysis"]:
                        analysis_result["gender_dynamics"] = analysis_result["bias_analysis"]["gender_dynamics"]
                
                # FIX: Convert string bias_detections to objects if Claude ignored schema
                if "bias_detections" in analysis_result:
                    bias_detections = analysis_result["bias_detections"]
                    if isinstance(bias_detections, list) and len(bias_detections) > 0:
                        # Check if first item is a string (Claude ignored schema)
                        if isinstance(bias_detections[0], str):
                            print(f"⚠️  CONVERTING: Claude returned strings, converting to objects...")
                            fixed_detections = []
                            
                            # Extract context from video_analysis OR bias_analysis
                            video_data = analysis_result.get("video_analysis", {})
                            bias_analysis_data = analysis_result.get("bias_analysis", {})
                            demo = video_data.get("demographic_representation", {}) or bias_analysis_data.get("representation", {})
                            text_data = bias_analysis_data.get("text_analysis", {})
                            gender_dynamics = bias_analysis_data.get("gender_dynamics", {})
                            
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
                                if text_data and "visible_text" in text_data:
                                    evidence.append(f"Text: {text_data['visible_text']}")
                                if text_data and "potential_biases" in text_data:
                                    evidence.extend(text_data["potential_biases"])
                                
                                # Add demographic evidence
                                if demo:
                                    if "gender_distribution" in demo:
                                        evidence.append(f"Gender distribution: {demo['gender_distribution']}")
                                    if "ethnicity" in demo:
                                        evidence.append(f"Ethnicity: {demo['ethnicity']}")
                                    if "age_range" in demo:
                                        evidence.append(f"Age: {demo['age_range']}")
                                
                                # Add gender dynamics evidence
                                if gender_dynamics and "stereotypes_detected" in gender_dynamics:
                                    evidence.extend(gender_dynamics["stereotypes_detected"])
                                
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


def _get_bias_analysis_prompt() -> str:
    """Get the core bias analysis prompt (reusable)."""
    return """Please provide a detailed analysis covering:

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

**FORMAT REQUIREMENT**: 
You MUST respond with ONLY a valid JSON object matching the structure shown above. 
Do NOT include any text before or after the JSON.
Do NOT wrap the JSON in markdown code blocks.
Do NOT add explanations or commentary.
Just return the raw JSON object."""

