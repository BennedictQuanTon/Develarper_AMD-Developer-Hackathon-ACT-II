#!/usr/bin/env python3
"""
scripts/test_local.py — Chạy 8 practice tasks từ CONTEXT.md, local SLM only.
Hiển thị: route, câu trả lời, số token (input + output) từng task.
Lưu kết quả ra output/practice_results.json đúng format cuộc thi.

Cách chạy:
    PYTHONPATH=. python scripts/test_local.py
    PYTHONPATH=. python scripts/test_local.py tests/fixtures/sample_tasks.json
"""

import json
import logging
import os
import sys
import time

# Suppress noisy logs — chỉ show ERROR
logging.basicConfig(level=logging.ERROR)
os.environ.setdefault("LOCAL_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
os.environ.setdefault("LOCAL_N_CTX", "2048")
os.environ.setdefault("LOCAL_N_THREADS", "2")
os.environ.setdefault("LOCAL_N_GPU_LAYERS", "0")

# Load .env nếu có
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from agent.ast_eval import evaluate_math_expression  # noqa: E402
from agent.classifier import ROUTE_API_CODE, ROUTE_API_LOGIC, ROUTE_API_MATH, classify  # noqa: E402
from engines.local_slm import LocalSLMEngine  # noqa: E402
from handlers._base import load_prompt_template  # noqa: E402

# ---------------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------------
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/practice_tasks.json"
OUTPUT_FILE = "output/practice_results.json"

if not os.path.exists(INPUT_FILE):
    print(f"❌ Input file không tìm thấy: {INPUT_FILE}")
    sys.exit(1)

with open(INPUT_FILE, encoding="utf-8") as f:
    tasks = json.load(f)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  AMD Hackathon — Local Test Runner")
print(f"  Input: {INPUT_FILE}")
print("=" * 70)
print()
print("⏳ Đang load Qwen2.5-3B + SemanticClassifier...")
t0 = time.time()
engine = LocalSLMEngine.get_instance()
load_time = time.time() - t0
print(f"✅ Models loaded trong {load_time:.1f}s")
print()

# ---------------------------------------------------------------------------
# Load system prompts
# ---------------------------------------------------------------------------
prompts = {
    "factual": load_prompt_template("factual.txt"),
    "sentiment": load_prompt_template("sentiment.txt"),
    "ner": load_prompt_template("ner.txt"),
    "summarization": load_prompt_template("summarization.txt"),
    "code": load_prompt_template("remote_code.txt"),
    "math": load_prompt_template("remote_math.txt"),
    "logic": load_prompt_template("remote_logic.txt"),
}


def count_tokens(text: str) -> int:
    """Đếm token chính xác dùng llama.cpp tokenizer của Qwen2.5."""
    try:
        return len(engine.model.tokenize(text.encode("utf-8")))
    except Exception:
        # Fallback: ước tính (~1.3 tokens per word)
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Run tasks
# ---------------------------------------------------------------------------
results = []
total_input_tokens = 0
total_output_tokens = 0

for i, task in enumerate(tasks, 1):
    tid = task["task_id"]
    prompt = task["prompt"]
    t_start = time.time()

    # --- Layer 1b: AST deterministic math ---
    ast_result = evaluate_math_expression(prompt)
    if ast_result:
        route = "AST_EVAL (0 tokens)"
        answer = ast_result
        in_tok = 0
        out_tok = 0
    else:
        route = classify(prompt)

        # Chọn system prompt phù hợp
        if route == "LOCAL_SENTIMENT":
            sys_p = prompts["sentiment"]
            max_t = 20
        elif route == "LOCAL_NER":
            sys_p = prompts["ner"]
            max_t = 300
        elif route == ROUTE_API_MATH:
            sys_p = prompts["math"]
            max_t = 150
        elif route == ROUTE_API_CODE:
            sys_p = prompts["code"]
            max_t = 400
        elif route == ROUTE_API_LOGIC:
            sys_p = prompts["logic"]
            max_t = 200
        else:  # LOCAL_GENERAL, API_LONG, fallback
            p_lower = prompt.lower()
            if any(w in p_lower for w in ["summarize", "summary", "tldr", "in exactly one sentence", "condense"]):
                sys_p = prompts["summarization"]
                max_t = 250
            else:
                sys_p = prompts["factual"]
                max_t = 150

        # Format full prompt như Qwen2.5 chat template để count đúng input tokens
        formatted = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        in_tok = count_tokens(formatted)

        answer = engine.generate(prompt, sys_p, max_tokens=max_t)
        out_tok = count_tokens(answer)

    elapsed = time.time() - t_start
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    results.append({"task_id": tid, "answer": answer})

    # --- Print kết quả ---
    print(f"{'─' * 70}")
    print(f"[{i}/8] {tid}  │  route: {route}  │  ⏱ {elapsed:.1f}s")
    print(f"       📥 input: {in_tok} tokens   📤 output: {out_tok} tokens   " f"📊 total: {in_tok + out_tok} tokens")
    print(f"PROMPT : {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"ANSWER : {answer[:200]}{'...' if len(answer) > 200 else ''}")

# ---------------------------------------------------------------------------
# Tổng kết
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  TỔNG KẾT TOKEN USAGE")
print("=" * 70)
print(f"  📥 Tổng input tokens  : {total_input_tokens:,}")
print(f"  📤 Tổng output tokens : {total_output_tokens:,}")
print(f"  📊 Tổng cộng          : {total_input_tokens + total_output_tokens:,}")
print()
print("  ⚠️  Ghi chú: đây là LOCAL tokens (0 Fireworks tokens)")
print("  ⚠️  Fireworks chỉ tính token khi gọi qua FIREWORKS_BASE_URL")
print("=" * 70)

# ---------------------------------------------------------------------------
# Ghi output file đúng format cuộc thi
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print()
print(f"✅ Đã lưu {len(results)} kết quả → {OUTPUT_FILE}")
print()
print("--- Output JSON (format cuộc thi) ---")
print(json.dumps(results, ensure_ascii=False, indent=2))
