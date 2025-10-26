#!/usr/bin/env python3
"""
Test script to verify ingestion agent is working properly
"""

import requests
import json
import uuid

# Test video URL
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

def test_ingestion_agent():
    """Test the ingestion agent REST endpoint"""

    request_id = str(uuid.uuid4())

    payload = {
        "request_id": request_id,
        "content_type": "video",
        "text_content": None,
        "image_url": None,
        "video_url": TEST_VIDEO_URL,
        "metadata": {
            "source": "test_script",
            "test": True
        },
        "timestamp": ""
    }

    print("="*70)
    print("Testing Ingestion Agent")
    print("="*70)
    print(f"Request ID: {request_id}")
    print(f"Video URL: {TEST_VIDEO_URL}")
    print(f"Endpoint: http://localhost:8100/submit")
    print()

    try:
        # Try /analyze endpoint (the working one, not the reserved /submit)
        print("Sending request...")
        response = requests.post(
            "http://localhost:8100/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))

        if response.status_code == 200:
            print("\n✅ SUCCESS - Ingestion agent is working!")
            return True
        else:
            print(f"\n⚠️ WARNING - Got status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError as e:
        print(f"❌ ERROR - Cannot connect to ingestion agent")
        print(f"   Make sure it's running: python agents/ingestion_agent.py")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR - {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_ingestion_agent()
