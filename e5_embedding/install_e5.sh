#!/bin/bash

# Quick sync script with verbose output to see what's happening

# Pre-install diagnostic
echo "=========================================="
echo "Python environment check:"
echo "=========================================="
which python
python --version
which pip
pip --version
echo ""

echo "=========================================="
echo "Installing E5-large dependencies..."
echo "=========================================="

# Verbose output for troubleshooting
pip install -v \
    torch>=2.0.0 \
    sentence-transformers>=2.2.0 \
    langchain>=0.1.0 \
    langchain-community>=0.0.10 \
    langchain-text-splitters>=0.0.1 \
    chromadb>=0.4.0 \
    huggingface-hub>=0.16.0

echo ""
echo "=========================================="
echo "Post-install verification..."
echo "=========================================="

python -c "import torch, sentence_transformers, langchain, chromadb; print('✅ All core packages installed successfully!')" 2>/dev/null || echo "⚠️  Some packages may be missing or import errors encountered"

echo ""
echo "=========================================="
echo "Verify sentence-transformers models can be loaded..."
echo "=========================================="
python -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers loaded successfully')" 2>/dev/null || echo "⚠️  Error importing sentence-transformers"

echo ""
echo "=========================================="
echo "Installation summary:"
echo "=========================================="
pip list | grep -E "(torch|sentence-transformers|langchain|chromadb)"

echo ""
echo "=========================================="
echo "For conda installation, use the conda_environment.yml file"
echo "=========================================="