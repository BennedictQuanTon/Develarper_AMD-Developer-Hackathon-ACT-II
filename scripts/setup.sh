#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — Cài đặt toàn bộ dự án từ đầu
# Dùng cho teammate fork về lần đầu tiên
#
# Cách chạy:
#   bash scripts/setup.sh
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
echo "  AMD Hackathon — Project Setup"
echo "=============================================="
echo ""

# --- Kiểm tra Python ---
info "Kiểm tra Python version..."
if ! command -v python3 &>/dev/null; then
    error "Python3 không tìm thấy. Cài Python 3.10+ trước."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER ✓"

# --- Tạo virtual environment ---
if [ ! -d ".venv" ]; then
    info "Tạo virtual environment (.venv)..."
    python3 -m venv .venv
else
    warn ".venv đã tồn tại, bỏ qua bước tạo."
fi

# Activate venv
source .venv/bin/activate
info "Virtual environment activated ✓"

# --- Upgrade pip ---
info "Upgrade pip..."
pip install --quiet --upgrade pip

# --- Cài torch CPU-only trước (tránh pull CUDA wheels 2.5 GB) ---
info "Cài torch CPU-only (~700 MB, có thể mất vài phút)..."
pip install --quiet torch --extra-index-url https://download.pytorch.org/whl/cpu

# --- Cài requirements.txt ---
info "Cài dependencies từ requirements.txt..."
pip install --quiet -r requirements.txt

# --- Cài llama-cpp-python (CPU wheel, không cần compile C++) ---
info "Cài llama-cpp-python (CPU wheel)..."
pip install --quiet llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# --- Pre-cache sentence-transformer model ---
info "Pre-cache sentence-transformer model (all-MiniLM-L6-v2 ~90 MB)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('  Model cached ✓')
" 2>/dev/null

# --- Download Qwen2.5-1.5B GGUF model ---
info "Kiểm tra model weights Qwen2.5-1.5B..."
MODEL_PATH="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
if [ -f "$MODEL_PATH" ]; then
    warn "Model đã tồn tại tại $MODEL_PATH, bỏ qua download."
else
    info "Download Qwen2.5-1.5B (~1 GB)..."
    bash scripts/download_model.sh
fi

# --- Setup .env ---
if [ ! -f ".env" ]; then
    info "Tạo .env từ .env.example..."
    cp .env.example .env
    warn "⚠️  Mở file .env và điền FIREWORKS_API_KEY của bạn vào!"
else
    warn ".env đã tồn tại, bỏ qua."
fi

# --- Tạo thư mục output ---
mkdir -p output

echo ""
echo "=============================================="
echo -e "  ${GREEN}✅ Setup hoàn tất!${NC}"
echo "=============================================="
echo ""
echo "Bước tiếp theo:"
echo "  1. Điền API key vào .env  (nếu chưa làm)"
echo "  2. Chạy dự án:   bash scripts/run.sh"
echo "  3. Chạy test:    bash scripts/test_local.sh"
echo ""
