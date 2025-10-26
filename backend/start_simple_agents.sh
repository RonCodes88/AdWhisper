#!/bin/bash

# AdWhisper Simplified Multi-Agent System Startup Script
# Starts all simplified agents in the background

set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

PYTHON="./adwhisper/bin/python"
LOGS_DIR="./logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🚀 Starting AdWhisper Simplified Agent System           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to start an agent
start_agent() {
    local agent_file="$1"
    local agent_name="$2"
    local port="$3"
    local log_file="$LOGS_DIR/${agent_name}.log"

    echo "📍 Starting ${agent_name} on port ${port}..."

    # Kill any existing process on this port
    lsof -ti:${port} | xargs kill -9 2>/dev/null || true

    # Start the agent in background
    nohup $PYTHON "agents/${agent_file}" > "$log_file" 2>&1 &
    local pid=$!

    echo "   ✅ ${agent_name} started (PID: $pid)"
    echo "   📝 Logs: $log_file"
    echo ""

    # Give the agent time to start
    sleep 2
}

# Start all simplified agents
echo "🔧 Starting simplified agent architecture..."
echo "   • NO ChromaDB in Text/Visual agents"
echo "   • ONLY ChromaDB in Scoring agent (RAG)"
echo "   • Clean uAgents message passing"
echo ""

start_agent "simple_ingestion_agent.py" "Simple Ingestion Agent" "8100"
start_agent "simple_text_bias_agent.py" "Simple Text Bias Agent" "8101"
start_agent "simple_visual_bias_agent.py" "Simple Visual Bias Agent" "8102"
start_agent "simple_scoring_agent.py" "Simple Scoring Agent" "8103"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ✅ All Simplified Agents Started                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Agent Endpoints:"
echo "  • Ingestion Agent:     http://localhost:8100/analyze"
echo "  • Text Bias Agent:     http://localhost:8101/submit"
echo "  • Visual Bias Agent:   http://localhost:8102/submit"
echo "  • Scoring Agent:       http://localhost:8103/submit"
echo ""
echo "Agent Flow:"
echo "  1. Ingestion → Extracts YouTube + Routes to agents"
echo "  2. Text Agent → Analyzes text (NO ChromaDB)"
echo "  3. Visual Agent → Analyzes frames (NO ChromaDB)"
echo "  4. Scoring Agent → Aggregates + ChromaDB RAG + Final report"
echo ""
echo "Logs Location: $LOGS_DIR/"
echo ""
echo "To stop all agents:"
echo "  ./stop_agents.sh"
echo ""
echo "To view logs:"
echo "  tail -f $LOGS_DIR/<agent_name>.log"
echo ""
