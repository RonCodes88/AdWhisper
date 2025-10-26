"""
Test script to verify agent communication using ctx.send_and_receive
"""
import requests
import json
import time
from datetime import datetime

def test_ingestion_agent():
    """Send a test request to the ingestion agent"""

    url = "http://localhost:8100/analyze"

    # Simple test content (no YouTube URL to avoid long processing)
    payload = {
        "request_id": f"test_{int(time.time())}",
        "content_type": "text",
        "text_content": "Looking for young energetic rockstars to join our team!",
        "metadata": {
            "source": "test",
            "test": True
        }
    }

    print("=" * 70)
    print("🧪 Testing Agent Communication")
    print("=" * 70)
    print(f"📝 Request ID: {payload['request_id']}")
    print(f"🎯 Endpoint: {url}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        print("⏱️  Sending request...")
        start_time = time.time()

        response = requests.post(url, json=payload, timeout=10)

        elapsed = time.time() - start_time

        print(f"✅ Response received in {elapsed:.2f}s")
        print(f"📊 Status Code: {response.status_code}")
        print(f"📄 Response:")
        print(json.dumps(response.json(), indent=2))
        print()

        if response.status_code == 200:
            print("✅ Test PASSED - Ingestion agent responded successfully")
            print("ℹ️  Check the agent logs to verify:")
            print("   - Text Bias Agent received and processed the text")
            print("   - Scoring Agent received the analysis results")
        else:
            print("❌ Test FAILED - Unexpected status code")

    except requests.exceptions.Timeout:
        print("❌ Test FAILED - Request timed out")
    except Exception as e:
        print(f"❌ Test FAILED - Error: {e}")

    print("=" * 70)

if __name__ == "__main__":
    # Wait a bit for agents to fully start
    print("⏳ Waiting 3 seconds for agents to start...")
    time.time()

    test_ingestion_agent()
