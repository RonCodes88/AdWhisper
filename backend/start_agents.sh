#!/bin/bash

# AdWhisper Multi-Agent System Startup Script
# This script starts all agents in the background

set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

PYTHON="./adwhisper/bin/python"
LOGS_DIR="./logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          🚀 Starting AdWhisper Agent System                  ║"
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

# Start all agents
start_agent "ingestion_agent.py" "Ingestion Agent" "8100"
start_agent "text_bias_agent.py" "Text Bias Agent" "8101"
start_agent "visual_bias_agent.py" "Visual Bias Agent" "8102"
start_agent "scoring_agent.py" "Scoring Agent" "8103"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ✅ All Agents Started Successfully                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Agent Endpoints:"
echo "  • Ingestion Agent:     http://localhost:8100/analyze"
echo "  • Text Bias Agent:     http://localhost:8101/submit"
echo "  • Visual Bias Agent:   http://localhost:8102/submit"
echo "  • Scoring Agent:       http://localhost:8103/submit"
echo ""
echo "Logs Location: $LOGS_DIR/"
echo ""
echo "To stop all agents:"
echo "  ./stop_agents.sh"
echo ""
echo "To view logs:"
echo "  tail -f $LOGS_DIR/<agent_name>.log"
echo ""
