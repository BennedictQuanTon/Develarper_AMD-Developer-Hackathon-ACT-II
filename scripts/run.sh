#!/usr/bin/env bash
# =============================================================================
# scripts/run.sh — Chạy agent với input/output tùy chỉnh
#
# Cách dùng:
#   bash scripts/run.sh                          # dùng đường dẫn mặc định
#   bash scripts/run.sh my_tasks.json            # chỉ định input file
#   bash scripts/run.sh my_tasks.json out.json   # chỉ định cả input và output
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Đường dẫn mặc định ---
INPUT_FILE="${1:-tests/fixtures/practice_tasks.json}"
OUTPUT_FILE="${2:-output/results.json}"

echo ""
echo "=============================================="
echo "  AMD Hackathon Agent — Run"
echo "=============================================="
echo "  Input  : $INPUT_FILE"
echo "  Output : $OUTPUT_FILE"
echo "=============================================="
echo ""

# --- Kiểm tra .env ---
if [ ! -f ".env" ]; then
    error ".env không tồn tại. Chạy 'bash scripts/setup.sh' trước."
fi

# --- Kiểm tra model weights ---
MODEL_PATH=$(grep "LOCAL_MODEL_PATH" .env | cut -d= -f2 | xargs)
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    error "Model không tìm thấy tại '$MODEL_PATH'. Chạy 'bash scripts/download_model.sh' trước."
fi

# --- Kiểm tra input file ---
if [ ! -f "$INPUT_FILE" ]; then
    error "Input file không tồn tại: $INPUT_FILE"
fi

# --- Activate venv nếu có ---
if [ -d ".venv" ]; then
    source .venv/bin/activate
    info "Virtual environment activated ✓"
fi

# --- Tạo thư mục output ---
mkdir -p "$(dirname "$OUTPUT_FILE")"

# --- Chạy agent ---
info "Bắt đầu chạy agent..."
echo ""
INPUT_PATH="$INPUT_FILE" \
OUTPUT_PATH="$OUTPUT_FILE" \
PYTHONPATH=. python main.py

echo ""
if [ -f "$OUTPUT_FILE" ]; then
    TASK_COUNT=$(python3 -c "import json; d=json.load(open('$OUTPUT_FILE')); print(len(d))")
    info "✅ Hoàn tất! $TASK_COUNT tasks → $OUTPUT_FILE"
    echo ""
    echo "--- Kết quả (preview) ---"
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)
for r in results:
    ans = r['answer'][:80].replace('\n', ' ')
    print(f\"  [{r['task_id']}]: {ans}{'...' if len(r['answer']) > 80 else ''}\")
"
else
    error "Output file không được tạo ra. Kiểm tra logs bên trên."
fi
echo ""
