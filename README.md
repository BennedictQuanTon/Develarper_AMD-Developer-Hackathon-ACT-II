# AMD AI Developer Hackathon 2026 — Track 1: Token-Efficient Agent

**Team**: Develarper

> **Hackathon Goal**: Build a containerized AI agent covering 8 capability domains, pass the LLM-Judge accuracy gate (≥80%), and **minimize total Fireworks API tokens** for leaderboard ranking.

---

## Architecture

The agent uses a **4-layer hybrid routing engine** — each layer intercepts tasks at the lowest possible token cost before escalating to the next:

```
Input Task
    │
    ▼
[L1a] SHA-256 Semantic Cache          → duplicate task  → 0 tokens
    │ MISS
    ▼
[L1b] AST Math Evaluator              → pure expression → 0 tokens (deterministic)
    │ NOT PURE MATH
    ▼
[L2]  Semantic Embedding Classifier   → 0 tokens, ~10ms (all-MiniLM-L6-v2, local)
    │
    ├─► LOCAL_SENTIMENT / LOCAL_NER / LOCAL_GENERAL
    │       ▼
    │   [L3] Qwen2.5-1.5B Q4_K_M via llama.cpp  →  0 Fireworks tokens
    │       │ __ESCALATE__ signal
    │       ▼
    └─► API_MATH / API_CODE / API_LOGIC / API_LONG_CONTEXT
            ▼
        [L4] Fireworks API — category-aware model selection
                          + prompt compression (strip filler phrases)
                          + per-category max_tokens budget
```

### 8 Capability Domains

| Domain | Route | Engine | Token Budget |
|---|---|---|---|
| Factual Knowledge | `LOCAL_GENERAL` | Qwen2.5-1.5B | 150 (local) |
| Math – pure expression | `AST_EVAL` | AST (deterministic) | 0 |
| Math – word problem | `API_MATH` | `gemma-4-31b-it` | 50 |
| Sentiment Classification | `LOCAL_SENTIMENT` | Qwen2.5-1.5B | 20 (local) |
| Text Summarization (≤6k) | `LOCAL_GENERAL` | Qwen2.5-1.5B | 250 (local) |
| Text Summarization (>6k) | `API_LONG_CONTEXT` | `gemma-4-26b-a4b-it` | 200 |
| Named Entity Recognition | `LOCAL_NER` | Qwen2.5-1.5B | 300 (local) |
| Code Debugging | `API_CODE` | `kimi-k2p7-code` | 400 |
| Logical Reasoning | `API_LOGIC` | `gemma-4-31b-it` | 150 |
| Code Generation | `API_CODE` | `kimi-k2p7-code` | 500 |

### Semantic Classifier (L2)

Layer 2 uses **`all-MiniLM-L6-v2`** (sentence-transformers) for zero-shot semantic classification:
- Encodes the prompt into a vector embedding (~10ms, CPU-only)
- Compares cosine similarity against pre-computed label anchor embeddings for all 6 routes
- Picks the route with highest semantic similarity — no regex, no hardcoded keywords
- Model size: ~90 MB | RAM: ~200 MB | Runs entirely local → **0 Fireworks tokens**

### Prompt Compression

Before every remote API call, the prompt goes through two transforms:
1. **Filler strip** — removes phrases like *"Can you please explain..."*, *"I would like you to..."*
2. **Output suffix** — appends a concise constraint (e.g., `" Output ONLY the final numeric answer."`)

This reduces input + output tokens on every remote call.

---

## Project Structure

```
├── agent/
│   ├── schemas.py        # Pydantic Task & Result models
│   ├── cache.py          # SHA-256 semantic dedup cache (thread-safe)
│   ├── ast_eval.py       # Safe deterministic math evaluator (AST whitelist)
│   ├── classifier.py     # Semantic embedding classifier (all-MiniLM-L6-v2)
│   ├── router.py         # AgentRouter — orchestrates all 4 layers
│   └── watchdog.py       # Daemon thread: fires at 570s, flushes partial output
│
├── engines/
│   ├── local_slm.py      # llama-cpp-python wrapper (Qwen2.5-1.5B Q4_K_M)
│   └── remote_llm.py     # Async Fireworks API client (aiohttp + tenacity retry)
│
├── handlers/             # One handler file per capability domain
│   ├── _base.py          # Shared load_prompt_template utility
│   ├── factual.py        # → local SLM
│   ├── sentiment.py      # → local SLM (Positive / Negative / Neutral)
│   ├── ner.py            # → local SLM (JSON list output)
│   ├── summarization.py  # → local SLM
│   ├── math_handler.py   # → remote gemma-4-31b-it   (max 50 tokens)
│   ├── debug.py          # → remote kimi-k2p7-code   (max 400 tokens)
│   ├── code_gen.py       # → remote kimi-k2p7-code   (max 500 tokens)
│   ├── logic.py          # → remote gemma-4-31b-it   (max 150 tokens)
│   └── remote_handlers.py # RemoteGeneralHandler (escalation fallback)
│
├── prompts/              # System prompt templates (.txt)
├── models/               # Bundled GGUF weights (~1 GB, not tracked in git)
├── tests/
│   ├── fixtures/
│   │   ├── practice_tasks.json    # 8 practice tasks from CONTEXT.md
│   │   ├── sample_tasks.json      # 8 sample tasks (one per domain)
│   │   └── expected_results.json  # Baseline expected answers
│   ├── test_ast_eval.py
│   ├── test_cache.py
│   ├── test_classifier.py
│   ├── test_remote_llm.py
│   ├── test_router.py
│   └── test_integration.py
│
├── scripts/
│   ├── setup.sh             # 🚀 First-time setup (install everything)
│   ├── run.sh               # ▶️  Run agent with custom input/output
│   ├── test_local.py        # 🧪 Run practice tasks locally + token stats
│   ├── download_model.sh    # Download GGUF weights from HuggingFace
│   └── simulate_grading.sh  # Docker run with 4GB RAM / 2 CPU constraints
│
├── output/               # Generated results (git-ignored)
├── main.py               # Entrypoint
├── Dockerfile
├── .env.example          # Template — copy to .env and fill in credentials
└── requirements.txt
```

---

## Getting Started

> **Lần đầu fork về?** Chỉ cần 1 lệnh:

```bash
bash scripts/setup.sh
```

Script này tự động:
- Tạo virtual environment (`.venv`)
- Cài `torch` CPU-only (tránh CUDA wheels 2.5 GB)
- Cài tất cả dependencies từ `requirements.txt`
- Cài `llama-cpp-python` (CPU wheel, không cần C++ compiler)
- Pre-cache `all-MiniLM-L6-v2` (~90 MB)
- Download `Qwen2.5-1.5B Q4_K_M` GGUF (~1 GB)
- Tạo `.env` từ `.env.example`

Sau đó điền API key vào `.env`:
```bash
FIREWORKS_API_KEY=<your_key>
```

---

## Chạy dự án

### Chạy với input/output tùy chỉnh

```bash
# Dùng practice tasks mặc định
bash scripts/run.sh

# Chỉ định input file
bash scripts/run.sh path/to/tasks.json

# Chỉ định cả input và output
bash scripts/run.sh path/to/tasks.json path/to/results.json
```

### Chạy test local + xem token stats

```bash
PYTHONPATH=. python scripts/test_local.py
```

Output mẫu:
```
[1/8] practice-01  │  route: LOCAL_GENERAL  │  ⏱ 1.2s
      📥 input: 114 tokens   📤 output: 16 tokens   📊 total: 130 tokens
ANSWER : The capital of Australia is Canberra...

======================================================================
  TỔNG KẾT TOKEN USAGE
======================================================================
  📥 Tổng input tokens  : 1,107
  📤 Tổng output tokens : 489
  📊 Tổng cộng          : 1,596

  ⚠️  Đây là LOCAL tokens (0 Fireworks tokens)
```

### Chạy unit tests

```bash
# Unit tests — không cần model (mocked)
PYTHONPATH=. pytest tests/test_ast_eval.py tests/test_cache.py \
    tests/test_remote_llm.py tests/test_router.py -v

# Full integration test (loads local SLM)
PYTHONPATH=. python tests/test_integration.py
```

---

## Docker Build & Grading Simulation

> Download model weights trước khi build: `bash scripts/download_model.sh`

```bash
# 1. Build image (bắt buộc dùng --platform linux/amd64 để submit)
docker buildx build --platform linux/amd64 \
    -t <your-dockerhub-username>/develarper-agent:latest .

# 2. Kiểm tra image size (phải < 10 GB compressed)
docker images <your-dockerhub-username>/develarper-agent:latest

# 3. Simulate grading environment (4 GB RAM, 2 CPUs, no network)
FIREWORKS_API_KEY=your_key bash scripts/simulate_grading.sh

# 4. Xem kết quả
cat output_test/results.json

# 5. Push lên Docker Hub khi sẵn sàng submit
docker push <your-dockerhub-username>/develarper-agent:latest
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | Yes (grading) | API key — injected by harness lúc chấm |
| `FIREWORKS_BASE_URL` | Yes (grading) | Proxy URL cho token counting — harness inject |
| `ALLOWED_MODELS` | Yes (grading) | Comma-separated model IDs — harness inject |
| `LOCAL_MODEL_PATH` | Optional | Đường dẫn GGUF (default: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`) |
| `LOCAL_N_GPU_LAYERS` | Optional | GPU layers (`0` = CPU-only, `-1` = auto Metal trên Mac) |
| `LOCAL_N_THREADS` | Optional | CPU threads (default: `2`) |
| `LOCAL_N_CTX` | Optional | Context window size (default: `2048`) |
| `INPUT_PATH` | Runtime | Path đọc `tasks.json` (default: `/input/tasks.json`) |
| `OUTPUT_PATH` | Runtime | Path ghi `results.json` (default: `/output/results.json`) |

> **Lưu ý:** Harness sẽ inject `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` lúc chấm — không cần hardcode trong image.

---

## Grading Constraints

| Constraint | Limit | Approach |
|---|---|---|
| RAM | 4 GB | Qwen2.5 ~1 GB + MiniLM ~200 MB ≈ ~1.5 GB total ✅ |
| CPUs | 2 vCPUs | `n_threads=2` trong llama.cpp |
| Image size | 10 GB compressed | ~2–3 GB estimated ✅ |
| Runtime | 10 phút | Watchdog fires at 570s, flush partial results |
| Architecture | linux/amd64 | Build với `--platform linux/amd64` |

---

## Scoring Strategy

```
Minimize: Σ (input_tokens + output_tokens) sent via FIREWORKS_BASE_URL
Subject to: accuracy ≥ 80% (binary gate — phải pass trước)
```

- **Local execution = 0 Fireworks tokens** → maximize local handling
- **Semantic classifier** (MiniLM) → routing chính xác hơn → ít misroute → ít API call thừa
- **Prompt compression** → strip filler phrases + output suffix → giảm tokens mỗi remote call
- **Per-category `max_tokens` budgets** → giới hạn output dài không cần thiết
- **Semantic cache** → dedup identical/similar prompts

---

## Development Notes

- **Python version**: 3.10 (Docker) / 3.11+ (host dev)
- **Classifier**: `all-MiniLM-L6-v2` via `sentence-transformers` — zero-shot, no API calls
- **Local SLM**: `Qwen2.5-1.5B-Instruct Q4_K_M` via `llama-cpp-python`
- **Linting**: `ruff check .`
- **Type checking**: `mypy .`
- **Pre-commit**: `pre-commit run --all-files`
- Không cần API key để chạy local SLM tasks và unit tests
