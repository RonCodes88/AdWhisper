"""
Quick script to check if the backend is running and responding
"""

import requests
import sys

print("\n" + "="*70)
print("🔍 CHECKING BACKEND STATUS")
print("="*70)

# Check FastAPI server
print("\n1️⃣ Checking FastAPI Server (http://localhost:8000)...")
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("   ✅ FastAPI is running!")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ⚠️ FastAPI responded with status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ FastAPI is NOT running")
    print("   💡 Start it with: cd backend && ./adwhisper/bin/python main.py")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check Ingestion Agent
print("\n2️⃣ Checking Ingestion Agent (http://localhost:8100)...")
try:
    # The agent doesn't have a health endpoint, so we check if it's listening
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 8100))
    sock.close()
    
    if result == 0:
        print("   ✅ Ingestion Agent port is open (agent likely running)")
    else:
        print("   ⚠️ Ingestion Agent port is not accessible")
        print("   💡 Start it with: cd backend && ./adwhisper/bin/python agents/ingestion_agent.py")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check ChromaDB
print("\n3️⃣ Checking ChromaDB...")
try:
    sys.path.append('.')
    from chroma import ChromaDB
    db = ChromaDB()
    count = db.collection.count()
    print(f"   ✅ ChromaDB is accessible")
    print(f"   Documents stored: {count}")
except Exception as e:
    print(f"   ⚠️ ChromaDB error: {e}")

# Check Frontend
print("\n4️⃣ Checking Frontend (http://localhost:3000)...")
try:
    response = requests.get("http://localhost:3000", timeout=2)
    if response.status_code == 200:
        print("   ✅ Frontend is running!")
    else:
        print(f"   ⚠️ Frontend responded with status: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ Frontend is NOT running")
    print("   💡 Start it with: cd frontend && npm run dev")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
To run the full stack:

Terminal 1 - Backend API:
  cd backend
  ./adwhisper/bin/python main.py

Terminal 2 - Ingestion Agent (optional):
  cd backend
  ./adwhisper/bin/python agents/ingestion_agent.py

Terminal 3 - Frontend:
  cd frontend
  npm run dev

Then visit: http://localhost:3000/upload
""")
print("="*70 + "\n")

