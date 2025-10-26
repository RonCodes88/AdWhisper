#!/bin/bash

# AdWhisper Multi-Agent System Stop Script
# This script stops all running agents

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          🛑 Stopping AdWhisper Agent System                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to stop agent on a specific port
stop_agent() {
    local port="$1"
    local agent_name="$2"

    echo "🛑 Stopping ${agent_name} (port ${port})..."

    # Find and kill process on this port
    local pids=$(lsof -ti:${port} 2>/dev/null || true)

    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo "   ✅ ${agent_name} stopped"
    else
        echo "   ℹ️  ${agent_name} not running"
    fi
}

# Stop all agents
stop_agent "8100" "Ingestion Agent"
stop_agent "8101" "Text Bias Agent"
stop_agent "8102" "Visual Bias Agent"
stop_agent "8103" "Scoring Agent"

echo ""
echo "✅ All agents stopped"
echo ""
