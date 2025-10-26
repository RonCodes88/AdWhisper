#!/usr/bin/env python3
"""
Test script for Visual Bias Agent
Tests the visual bias detection functionality using extracted frames
"""

import os
import sys
import json
import base64
import requests
import asyncio
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

# Import the visual bias agent functions
from agents.visual_bias_agent import (
    analyze_base64_frames,
    analyze_multiple_frames,
    encode_image_to_base64,
    _fallback_analysis
)
from uagents import Context

class MockContext:
    """Mock context for testing"""
    def __init__(self):
        self.logger = self
        
    def info(self, msg):
        print(f"ℹ️  {msg}")
        
    def warning(self, msg):
        print(f"⚠️  {msg}")
        
    def error(self, msg):
        print(f"❌ {msg}")

def test_with_extracted_frames():
    """Test using the extracted frames in the backend/extracted_frames directory"""
    print("🧪 Testing Visual Bias Agent with Extracted Frames")
    print("=" * 60)
    
    # Check if frames exist
    frames_dir = backend_dir / "extracted_frames"
    if not frames_dir.exists():
        print("❌ No extracted_frames directory found!")
        return
    
    # Get all frame files
    frame_files = list(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print("❌ No frame files found!")
        return
    
    print(f"📁 Found {len(frame_files)} frame files:")
    for frame_file in sorted(frame_files):
        print(f"   - {frame_file.name}")
    
    # Test with first few frames
    test_frames = sorted(frame_files)[:3]  # Test with first 3 frames
    print(f"\n🎬 Testing with {len(test_frames)} frames...")
    
    # Create mock context
    ctx = MockContext()
    
    # Test the analyze_multiple_frames function
    try:
        print("\n🔄 Running analysis...")
        result = asyncio.run(analyze_multiple_frames(ctx, [str(f) for f in test_frames]))
        
        print("\n📊 Analysis Results:")
        print("=" * 40)
        print(json.dumps(result, indent=2))
        
        # Check if bias detections were found
        bias_detections = result.get('bias_detections', [])
        print(f"\n🎯 Bias Detections Found: {len(bias_detections)}")
        
        for i, bias in enumerate(bias_detections, 1):
            print(f"   {i}. Type: {bias.get('type', 'unknown')}")
            print(f"      Severity: {bias.get('severity', 'unknown')}")
            print(f"      Description: {bias.get('description', 'No description')[:100]}...")
        
        # Check overall assessment
        overall = result.get('overall_assessment', {})
        if overall:
            print(f"\n📈 Overall Assessment:")
            print(f"   Bias Score: {overall.get('bias_score', 'N/A')}/10")
            print(f"   Diversity Score: {overall.get('diversity_score', 'N/A')}/10")
            print(f"   Main Concerns: {overall.get('main_concerns', [])}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def test_rest_endpoint():
    """Test the REST endpoint directly"""
    print("\n🌐 Testing REST Endpoint")
    print("=" * 40)
    
    # Check if agent is running
    try:
        response = requests.get("http://localhost:8102/", timeout=5)
        print("✅ Visual Bias Agent is running!")
    except requests.exceptions.ConnectionError:
        print("❌ Visual Bias Agent is not running!")
        print("   Start it with: python agents/visual_bias_agent.py")
        return
    except Exception as e:
        print(f"❌ Error checking agent status: {e}")
        return
    
    # Prepare test data
    frames_dir = backend_dir / "extracted_frames"
    frame_files = list(frames_dir.glob("frame_*.jpg"))
    
    if not frame_files:
        print("❌ No frame files found for testing!")
        return
    
    # Encode first frame as base64
    test_frame = frame_files[0]
    print(f"📸 Using frame: {test_frame.name}")
    
    with open(test_frame, 'rb') as f:
        frame_data = base64.b64encode(f.read()).decode('utf-8')
        mime_type = 'image/jpeg'
        base64_frame = f"data:{mime_type};base64,{frame_data}"
    
    # Prepare request payload
    payload = {
        "request_id": "test_visual_analysis_001",
        "content_type": "video",  # Required field
        "frames_base64": [base64_frame],
        "visual_embedding": None,  # Optional
        "metadata": {
            "test": True,
            "source": "test_script"
        }
    }
    
    print(f"📤 Sending request to REST endpoint...")
    print(f"   URL: http://localhost:8102/analyze")
    print(f"   Payload size: {len(json.dumps(payload))} bytes")
    
    try:
        response = requests.post(
            "http://localhost:8102/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minutes timeout
        )
        
        print(f"📨 Response received: HTTP {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print("\n📊 Results:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (120s)")
    except Exception as e:
        print(f"❌ Error during REST request: {e}")

def test_environment_setup():
    """Test environment setup"""
    print("🔧 Testing Environment Setup")
    print("=" * 40)
    
    # Check environment variables
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "YOUR_ANTHROPIC_API_KEY_HERE":
        print("✅ ANTHROPIC_API_KEY is set")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set - will use fallback mode")
        print("   Set it with: export ANTHROPIC_API_KEY='your_key_here'")
    
    # Check model engine
    model_engine = os.getenv("MODEL_ENGINE", "claude-3-5-haiku-latest")
    print(f"🤖 Model Engine: {model_engine}")
    
    # Check max tokens
    max_tokens = os.getenv("MAX_TOKENS", "4096")
    print(f"📝 Max Tokens: {max_tokens}")
    
    # Check if frames directory exists
    frames_dir = backend_dir / "extracted_frames"
    if frames_dir.exists():
        frame_count = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"📁 Extracted frames: {frame_count} files")
    else:
        print("❌ No extracted_frames directory found")

def main():
    """Main test function"""
    print("🚀 Visual Bias Agent Test Suite")
    print("=" * 50)
    
    # Test 1: Environment setup
    test_environment_setup()
    
    # Test 2: Direct function testing
    test_with_extracted_frames()
    
    # Test 3: REST endpoint testing
    test_rest_endpoint()
    
    print("\n🎉 Test suite completed!")
    print("\n📋 Next Steps:")
    print("   1. Set ANTHROPIC_API_KEY environment variable for real analysis")
    print("   2. Start the visual bias agent: python agents/visual_bias_agent.py")
    print("   3. Test through the frontend interface")
    print("   4. Check logs in backend/logs/Visual Bias Agent.log")

if __name__ == "__main__":
    main()
