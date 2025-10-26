#!/bin/bash
# Simple curl test for Visual Bias Agent REST endpoint

echo "🧪 Testing Visual Bias Agent REST Endpoint"
echo "=========================================="

# Check if agent is running
if ! curl -s http://localhost:8102/ > /dev/null; then
    echo "❌ Visual Bias Agent is not running!"
    echo "   Start it with: python agents/visual_bias_agent.py"
    exit 1
fi

echo "✅ Visual Bias Agent is running"

# Create a simple test payload (you'll need to replace with actual base64 frame)
echo "📤 Sending test request..."

curl -X POST http://localhost:8102/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "curl_test_001",
    "content_type": "video",
    "frames_base64": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="],
    "visual_embedding": null,
    "metadata": {
      "test": true,
      "source": "curl_test"
    }
  }' \
  --max-time 120

echo ""
echo "🎉 Test completed!"
