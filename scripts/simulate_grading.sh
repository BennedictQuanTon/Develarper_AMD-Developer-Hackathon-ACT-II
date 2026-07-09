#!/usr/bin/env bash
# scripts/simulate_grading.sh
# Simulates the hackathon grading environment locally.
# Usage: bash scripts/simulate_grading.sh [IMAGE_TAG]
#
# Prerequisites:
#   1. Docker is running
#   2. Model weights are in ./models/ (run scripts/download_model.sh first)
#   3. Image is built: docker build -t amd-agent:latest .
#      For AMD submission: docker buildx build --platform linux/amd64 -t <username>/develarper-agent:latest .

set -euo pipefail

IMAGE="${1:-amd-agent:latest}"
INPUT_DIR="$(pwd)/tests/fixtures"
OUTPUT_DIR="$(pwd)/output_test"

echo "=== AMD Hackathon Grading Simulation ==="
echo "Image      : $IMAGE"
echo "Input      : $INPUT_DIR/sample_tasks.json"
echo "Output     : $OUTPUT_DIR/results.json"
echo ""

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Wipe previous output
rm -f "$OUTPUT_DIR/results.json"

echo "[1/3] Running container (--memory=4g --cpus=2 --network=none)..."
docker run --rm \
    --memory=4g \
    --cpus=2 \
    --network=none \
    -v "$INPUT_DIR:/input:ro" \
    -v "$OUTPUT_DIR:/output" \
    -e INPUT_PATH=/input/sample_tasks.json \
    -e OUTPUT_PATH=/output/results.json \
    -e FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-PLACEHOLDER}" \
    -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}" \
    -e ALLOWED_MODELS="${ALLOWED_MODELS:-minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4}" \
    "$IMAGE"

echo ""
echo "[2/3] Validating output schema..."
python3 - <<'EOF'
import json, sys
with open("output_test/results.json") as f:
    results = json.load(f)
assert isinstance(results, list), "results.json must be a JSON array"
for r in results:
    assert "task_id" in r, f"Missing task_id in: {r}"
    assert "answer" in r, f"Missing answer in: {r}"
    assert isinstance(r["answer"], str), f"answer must be string: {r}"
print(f"  ✓ {len(results)} tasks present, schema valid")
EOF

echo ""
echo "[3/3] Results preview:"
python3 -c "
import json
with open('output_test/results.json') as f:
    results = json.load(f)
for r in results:
    print(f'  {r[\"task_id\"]}: {r[\"answer\"][:80]!r}')
"
echo ""
echo "=== Simulation complete ==="
