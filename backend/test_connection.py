"""
Simple test to verify Frontend → FastAPI → Ingestion Agent connection
"""

import requests
import time

print("""
╔══════════════════════════════════════════════════════════════╗
║         🧪 Testing Frontend → Ingestion Agent Flow           ║
╚══════════════════════════════════════════════════════════════╝
""")

# Test 1: Check if FastAPI is running
print("\n1️⃣ Checking if FastAPI server is running...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ FastAPI server is running on port 8000")
    else:
        print(f"   ⚠️ FastAPI returned status: {response.status_code}")
        exit(1)
except:
    print("   ❌ FastAPI server is NOT running!")
    print("   Start it with: cd backend && python main.py")
    exit(1)

# Test 2: Check if Ingestion Agent is running
print("\n2️⃣ Checking if Ingestion Agent is running...")
try:
    # Test the REST endpoint directly
    test_payload = {
        "request_id": "test_123",
        "content_type": "video",
        "text_content": "Test content",
        "video_url": "https://example.com/test.mp4",
        "metadata": {"test": True}
    }
    response = requests.post(
        "http://localhost:8100/analyze",
        json=test_payload,
        timeout=5
    )
    if response.status_code == 200:
        print("   ✅ Ingestion Agent is running on port 8100")
        print(f"   📨 Response: {response.json()}")
    else:
        print(f"   ⚠️ Ingestion Agent returned status: {response.status_code}")
except Exception as e:
    print("   ❌ Ingestion Agent is NOT running!")
    print(f"   Error: {e}")
    print("   Start it with: cd backend/agents && python ingestion_agent.py")
    print("\n   ⚠️ Continuing without agent (FastAPI will work but won't process through agents)...")

# Test 3: Send a test YouTube URL through the full pipeline
print("\n3️⃣ Testing complete flow with YouTube URL...")
test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
print(f"   🔗 Test URL: {test_url}")

try:
    response = requests.post(
        "http://localhost:8000/api/analyze-youtube",
        json={"youtube_url": test_url},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("   ✅ Request successful!")
        print(f"   📝 Request ID: {result.get('request_id')}")
        print(f"   📊 Status: {result.get('status')}")
        print(f"   💬 Message: {result.get('message')}")
        print(f"   🎯 Bias Score: {result.get('bias_score')}")
    else:
        print(f"   ❌ Request failed with status: {response.status_code}")
        print(f"   Error: {response.text}")
        
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

print("\n" + "="*60)
print("✅ TEST COMPLETE!")
print("\nCheck the terminal logs:")
print("  • Terminal 1 (Ingestion Agent): Should show '🌐 REST request received'")
print("  • Terminal 2 (FastAPI): Should show '✅ Ingestion Agent received request'")

