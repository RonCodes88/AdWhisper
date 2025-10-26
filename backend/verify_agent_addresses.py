"""
Agent Address Verification Script

This script verifies that the hardcoded agent addresses in ingestion_agent.py
match the actual runtime addresses of the text and visual bias agents.
"""

from uagents import Agent

# Agent configurations matching your actual agents
text_bias_agent = Agent(
    name="text_bias_agent",
    seed="ad_bias_text_agent_unique_seed_2024",
    port=8101,
    endpoint=["http://localhost:8101/submit"],
    mailbox=True
)

visual_bias_agent = Agent(
    name="visual_bias_agent",
    seed="ad_bias_visual_agent_unique_seed_2024",
    port=8102,
    endpoint=["http://localhost:8102/submit"],
    mailbox=True
)

# Expected addresses from ingestion_agent.py
EXPECTED_TEXT_ADDRESS = "agent1q2f7k0hv7p63y9fjux702n68kyp3gdadljlfal4xpawylnxf2pvzjsppdlv"
EXPECTED_VISUAL_ADDRESS = "agent1qtnatq0rhrj2pauyg2a8dgf56uqkf6tw3757z806w6c57zkw9nry2my2933"

print("="*70)
print("🔍 Agent Address Verification")
print("="*70)

print("\n📝 Text Bias Agent:")
print(f"   Expected: {EXPECTED_TEXT_ADDRESS}")
print(f"   Actual:   {text_bias_agent.address}")
text_match = str(text_bias_agent.address) == EXPECTED_TEXT_ADDRESS
print(f"   Status:   {'✅ MATCH' if text_match else '❌ MISMATCH'}")

print("\n👁️  Visual Bias Agent:")
print(f"   Expected: {EXPECTED_VISUAL_ADDRESS}")
print(f"   Actual:   {visual_bias_agent.address}")
visual_match = str(visual_bias_agent.address) == EXPECTED_VISUAL_ADDRESS
print(f"   Status:   {'✅ MATCH' if visual_match else '❌ MISMATCH'}")

print("\n" + "="*70)
if text_match and visual_match:
    print("✅ ALL ADDRESSES MATCH! Agent communication should work correctly.")
    print("\n📋 To test communication:")
    print("   1. Terminal 1: python backend/agents/text_bias_agent.py")
    print("   2. Terminal 2: python backend/agents/visual_bias_agent.py")
    print("   3. Terminal 3: python backend/agents/ingestion_agent.py")
    print("   4. Send a test request to ingestion agent")
else:
    print("❌ ADDRESS MISMATCH DETECTED!")
    print("\n🔧 To fix this, update ingestion_agent.py with these addresses:")
    print(f"\n   TEXT_BIAS_AGENT_ADDRESS = \"{text_bias_agent.address}\"")
    print(f"   VISUAL_BIAS_AGENT_ADDRESS = \"{visual_bias_agent.address}\"")

print("="*70)
