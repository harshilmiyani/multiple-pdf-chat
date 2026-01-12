#!/bin/bash

# RAG PDF Chatbot Setup Script
# This script sets up the development environment

set -e  # Exit on error

echo "🚀 Setting up RAG PDF Chatbot..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version OK: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Note: This is a client-side only application
# Users will enter their API key in the Streamlit UI
echo "ℹ️  Note: This is a client-side only application."
echo "   Users will enter their Google API key in the browser UI."

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p faiss_index

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your Google API key from: https://ai.google.dev/"
echo "2. Run: streamlit run app.py"
echo "3. Enter your API key in the sidebar when the app opens"
echo ""
echo "Note: Your API key is stored only in your browser session (client-side only)"
echo ""
