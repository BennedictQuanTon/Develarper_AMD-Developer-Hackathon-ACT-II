#!/usr/bin/env bash
# =============================================================================
# scripts/run_uv.sh — Run the agent with custom input/output using uv
#
# Usage:
#   bash scripts/run_uv.sh                          # uses default paths
#   bash scripts/run_uv.sh my_tasks.json            # specifies input file
#   bash scripts/run_uv.sh my_tasks.json out.json   # specifies both input and output
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# --- Default Paths ---
INPUT_FILE="${1:-tests/fixtures/practice_tasks.json}"
OUTPUT_FILE="${2:-output/results.json}"

echo ""
echo "=============================================="
echo "  AMD Hackathon Agent — Run (UV)"
echo "=============================================="
echo "  Input  : $INPUT_FILE"
echo "  Output : $OUTPUT_FILE"
echo "=============================================="
echo ""

# --- Check .env ---
if [ ! -f ".env" ]; then
    error ".env file does not exist. Run 'bash scripts/setup_uv.sh' first."
fi

# --- Check model weights ---
MODEL_PATH=$(grep "LOCAL_MODEL_PATH" .env | cut -d= -f2 | xargs)
MODEL_PATH="${MODEL_PATH:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    error "Model not found at '$MODEL_PATH'. Run 'bash scripts/download_model.sh' first."
fi

# --- Check input file ---
if [ ! -f "$INPUT_FILE" ]; then
    error "Input file does not exist: $INPUT_FILE"
fi

# --- Activate venv if present ---
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    fi
    info "Virtual environment activated ✓"
fi

# --- Create output folder ---
mkdir -p "$(dirname "$OUTPUT_FILE")"

# --- Run agent ---
info "Starting agent execution..."
echo ""
INPUT_PATH="$INPUT_FILE" \
OUTPUT_PATH="$OUTPUT_FILE" \
PYTHONPATH=. uv run python main.py

echo ""
if [ -f "$OUTPUT_FILE" ]; then
    TASK_COUNT=$(uv run python -c "import json; d=json.load(open('$OUTPUT_FILE')); print(len(d))")
    info "✅ Run complete! $TASK_COUNT tasks processed → $OUTPUT_FILE"
    echo ""
    echo "--- Results preview ---"
    uv run python -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)
for r in results:
    ans = r['answer'][:80].replace('\n', ' ')
    print(f\"  [{r['task_id']}]: {ans}{'...' if len(r['answer']) > 80 else ''}\")
"
else
    error "Output file was not created. Please check the log messages above."
fi
echo ""
