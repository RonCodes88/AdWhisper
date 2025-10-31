#!/bin/bash

# AdWhisper Backend Setup Script
# This script automates the initial setup process

echo "=================================="
echo "AdWhisper Backend Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"
echo ""

# Create .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env and add your API keys"
else
    echo "ℹ️  .env file already exists, skipping..."
fi

echo ""

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads
echo "✅ Uploads directory created"
echo ""

# Seed ChromaDB
echo "💾 Seeding ChromaDB with initial bias patterns..."
python seed_chromadb.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to seed ChromaDB"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   nano .env"
echo ""
echo "2. Start the agents (in separate terminals):"
echo "   Terminal 1: python agents/ingestion_agent.py"
echo "   Terminal 2: python agents/text_bias_agent.py"
echo "   Terminal 3: python agents/visual_bias_agent.py"
echo "   Terminal 4: python agents/scoring_agent.py"
echo "   Terminal 5: python main.py"
echo ""
echo "3. Copy agent addresses from logs and update .env"
echo ""
echo "4. Access the API:"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs (Swagger UI)"
echo ""
echo "For more information, see README.md"
echo "=================================="

