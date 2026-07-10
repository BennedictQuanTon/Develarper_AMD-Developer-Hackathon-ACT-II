#!/usr/bin/env bash
# =============================================================================
# scripts/setup_uv.sh — Full project setup using uv
# Designed for teammates setting up the project for the first time
#
# Usage:
#   bash scripts/setup_uv.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "=============================================="
echo "  AMD Hackathon — Project Setup (UV)"
echo "=============================================="
echo ""

# --- Check uv installation ---
info "Checking for uv installation..."
if ! command -v uv &>/dev/null; then
    error "uv is not installed. Please install it first (e.g., visit https://astral.sh/uv)."
fi
info "uv is installed ✓"

# --- Check Python ---
info "Checking Python version..."
if ! command -v uv python -v &>/dev/null; then
    error "Python3 was not found. Please install Python 3.10+ first."
fi
PY_VER=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER found ✓"

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    info "Creating virtual environment (.venv) using uv..."
    uv venv
else
    warn ".venv already exists, skipping creation."
fi

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    error "Could not locate virtual environment activation script."
fi
info "Virtual environment activated ✓"

# --- Install torch CPU-only first (to avoid pulling CUDA wheels ~2.5 GB) ---
info "Installing torch CPU-only (~700 MB, this may take a few minutes)..."
uv pip install --quiet torch --extra-index-url https://download.pytorch.org/whl/cpu

# --- Install requirements.txt ---
info "Installing dependencies from requirements.txt..."
uv pip install --quiet -r requirements.txt

# --- Install llama-cpp-python (CPU wheel, avoiding C++ compilation) ---
info "Installing llama-cpp-python (CPU wheel)..."
uv pip install --quiet llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# --- Pre-cache sentence-transformer model ---
info "Pre-caching sentence-transformer model (all-MiniLM-L6-v2 ~90 MB)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('  Model cached ✓')
" 2>/dev/null || warn "Could not pre-cache model automatically. It will download on first run."

# --- Download Qwen2.5-1.5B GGUF model ---
info "Checking Qwen2.5-1.5B model weights..."
MODEL_PATH="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
if [ -f "$MODEL_PATH" ]; then
    warn "Model weights already exist at $MODEL_PATH, skipping download."
else
    info "Downloading Qwen2.5-1.5B model (~1 GB)..."
    bash scripts/download_model.sh
fi

# --- Setup .env ---
if [ ! -f ".env" ]; then
    info "Creating .env file from .env.example..."
    cp .env.example .env
    warn "⚠️  Please open the .env file and fill in your FIREWORKS_API_KEY!"
else
    warn ".env already exists, skipping."
fi

# --- Create output folder ---
mkdir -p output

echo ""
echo "=============================================="
echo -e "  ${GREEN}✅ Setup complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Fill in your API key in .env (if not done already)"
echo "  2. Run the project:  bash scripts/run.sh"
echo "  3. Run the tests:    python scripts/test_local.py"
echo ""
