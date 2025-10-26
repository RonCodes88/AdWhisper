#!/usr/bin/env python3
"""
Quick Test for Visual Bias Agent
Simple test using the extracted frames
"""

import os
import sys
import json
import base64
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

def quick_test():
    """Quick test of the visual bias agent"""
    print("🧪 Quick Visual Bias Agent Test")
    print("=" * 40)
    
    # Check environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and api_key != "YOUR_ANTHROPIC_API_KEY_HERE":
        print("✅ Claude API key is configured")
        api_available = True
    else:
        print("⚠️  Claude API key not set - using fallback mode")
        api_available = False
    
    # Check frames
    frames_dir = backend_dir / "extracted_frames"
    if not frames_dir.exists():
        print("❌ No extracted_frames directory found!")
        return
    
    frame_files = list(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        print("❌ No frame files found!")
        return
    
    print(f"📁 Found {len(frame_files)} frame files")
    
    # Test with first frame
    test_frame = frame_files[0]
    print(f"🎬 Testing with: {test_frame.name}")
    
    # Encode frame to base64
    with open(test_frame, 'rb') as f:
        frame_data = base64.b64encode(f.read()).decode('utf-8')
        base64_frame = f"data:image/jpeg;base64,{frame_data}"
    
    print(f"📦 Frame encoded: {len(base64_frame)} characters")
    
    # Test the analyze_base64_frames function directly
    try:
        from agents.visual_bias_agent import analyze_base64_frames
        import asyncio
        
        class MockContext:
            def __init__(self):
                self.logger = self
            def info(self, msg): print(f"ℹ️  {msg}")
            def warning(self, msg): print(f"⚠️  {msg}")
            def error(self, msg): print(f"❌ {msg}")
        
        print("\n🔄 Running analysis...")
        ctx = MockContext()
        result = asyncio.run(analyze_base64_frames(ctx, [base64_frame]))
        
        print("\n📊 Analysis Results:")
        print(json.dumps(result, indent=2))
        
        # Check for bias detections
        bias_detections = result.get('bias_detections', [])
        print(f"\n🎯 Bias Detections: {len(bias_detections)}")
        
        if bias_detections:
            for i, bias in enumerate(bias_detections, 1):
                print(f"   {i}. {bias.get('type', 'unknown')} - {bias.get('severity', 'unknown')}")
                print(f"      {bias.get('description', 'No description')[:80]}...")
        
        # Check overall assessment
        overall = result.get('overall_assessment', {})
        if overall:
            print(f"\n📈 Scores:")
            print(f"   Bias: {overall.get('bias_score', 'N/A')}/10")
            print(f"   Diversity: {overall.get('diversity_score', 'N/A')}/10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
