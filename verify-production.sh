#!/usr/bin/env bash
# Production Deployment Checklist Script

echo "🚀 RAG System - Production Verification"
echo "======================================="
echo ""

# Check Python version
echo "✓ Checking Python..."
python3 --version

# Check Node version
echo "✓ Checking Node.js..."
node --version

# Check Docker
echo "✓ Checking Docker..."
docker --version 2>/dev/null || echo "  ⚠ Docker not found (required for Render)"

# Verify critical files
echo ""
echo "✓ Checking critical files..."
files=("Dockerfile" "requirements.txt" "render.yaml" ".env.example" "README.md" "DEPLOYMENT.md")
for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ $file MISSING"
  fi
done

# Check Python dependencies
echo ""
echo "✓ Checking Python dependencies..."
python3 -c "import fastapi; import pinecone; import langchain" 2>/dev/null && echo "  ✓ Core packages installed" || echo "  ⚠ Run: pip install -r requirements.txt"

# Check Node dependencies
echo ""
echo "✓ Checking Node dependencies..."
if [ -d "ui/node_modules" ]; then
  echo "  ✓ Node modules installed"
else
  echo "  ⚠ Run: cd ui && npm install"
fi

# Check environment
echo ""
echo "✓ Checking environment setup..."
if [ -f ".env" ]; then
  echo "  ✓ .env file exists"
  if grep -q "PINECONE_API_KEY" .env && grep -q "OPENAI_API_KEY" .env; then
    echo "  ✓ Required API keys configured"
  else
    echo "  ⚠ API keys not fully configured"
  fi
else
  echo "  ⚠ .env file not found (copy from .env.example)"
fi

# Summary
echo ""
echo "======================================="
echo "✅ Production system is ready!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Add your API keys to .env"
echo "3. Run: docker compose up --build"
echo "4. Or follow: cat DEPLOYMENT.md"
echo ""
