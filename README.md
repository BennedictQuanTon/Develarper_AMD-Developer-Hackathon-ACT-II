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
[L2]  LLM Classifier (local Qwen)        → 0 tokens, grammar-constrained label
    │
    ├─► LOCAL_SENTIMENT / LOCAL_NER / LOCAL_GENERAL
    │       ▼
    │   [L3] Qwen2.5-3B Q4_K_M via llama.cpp  →  0 Fireworks tokens
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
| Factual Knowledge | `LOCAL_GENERAL` | Qwen2.5-3B | 150 (local) |
| Math – pure expression | `AST_EVAL` | AST (deterministic) | 0 |
| Math – word problem | `API_MATH` | `kimi-k2p7-code` / `minimax-m3` | 768 |
| Sentiment Classification | `LOCAL_SENTIMENT` | Qwen2.5-3B | 20 (local) |
| Text Summarization (≤6k chars) | `LOCAL_GENERAL` | Qwen2.5-3B | 250 (local) |
| Text Summarization (>6k chars) | `API_LONG_CONTEXT` | `gemma-4-26b-a4b-it` | 200 |
| Named Entity Recognition | `LOCAL_NER` | Qwen2.5-3B | 300 (local) |
| Code Debugging | `API_CODE` (local-first) | Qwen2.5-3B → `kimi-k2p7-code` | 400 |
| Logical Reasoning | `API_LOGIC` | `kimi-k2p7-code` / `minimax-m3` | 768 |
| Code Generation | `API_CODE` (local-first) | Qwen2.5-3B → `kimi-k2p7-code` | 500 |

> **Local-first code strategy**: Code debugging and code generation tasks use a difficulty classifier (`code_utils.py`). Easy/medium tasks attempt Qwen2.5-3B locally first, validated with `ast.parse` + completeness checks — only falling back to the remote API if local output is invalid. Hard tasks go directly to the remote API to avoid wasting wall-clock time.

### LLM Classifier (L2)

Layer 2 uses the **same local Qwen2.5-3B GGUF model** as the local handlers to pick the routing category — no separate embedding model or neural head:
- **Grammar-constrained output**: a GBNF grammar forces the model to emit exactly one of the valid route labels (e.g. `LOCAL_SENTIMENT`, `API_CODE`). There is no free-form prose to parse, so mislabels from a stray token are impossible.
- **Zero tokens**: runs entirely on the bundled model, 0 Fireworks API tokens.
- **Concurrency-safe**: the classifier and the local handlers share one llama.cpp context, serialized by a lock inside `LocalSLMEngine`; `router.py` offloads every local call (classify + handlers) to worker threads so the asyncio event loop stays responsive while remote API tasks run concurrently.
- **Long-context override**: prompts longer than 6,000 chars skip the model and route directly to `API_LONG_CONTEXT` to avoid CPU OOM.

### Remote Model Selection

The agent dynamically selects the best model from `ALLOWED_MODELS` (injected at runtime by the harness) using per-category priority preferences:

| Category | Preferred Models (in priority order) | Fallback |
|---|---|---|
| `API_CODE` | `kimi-k2p7-code` → `gemma-4-31b-it` | First available |
| `API_MATH` | `kimi-k2p7-code` → `minimax-m3` | First available |
| `API_LOGIC` | `kimi-k2p7-code` → `minimax-m3` | First available |
| `API_LONG_CONTEXT` | `gemma-4-26b-a4b-it` → `gemma-4-31b-it-nvfp4` | First available |
| `LOCAL_GENERAL` (escalation) | `minimax-m3` → `kimi-k2p7-code` | First available |
| `LOCAL_SENTIMENT` (escalation) | `minimax-m3` → `kimi-k2p7-code` | First available |
| `LOCAL_NER` (escalation) | `minimax-m3` → `kimi-k2p7-code` | First available |

### Prompt Compression

Before every remote API call, the prompt goes through two transforms:
1. **Filler strip** — removes phrases like *"Can you please explain..."*, *"I would like you to..."*
2. **Output suffix** — appends a concise constraint per category (e.g., `" Return ONLY raw code."` for code tasks)

This reduces input + output tokens on every remote call.

---

## Project Structure

```
├── agent/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic Task & Result models
│   ├── cache.py              # SHA-256 semantic dedup cache (thread-safe)
│   ├── ast_eval.py           # Safe deterministic math evaluator (AST whitelist)
│   ├── classifier.py         # LLM classifier (local Qwen, GBNF grammar-constrained)
│   ├── router.py             # AgentRouter — orchestrates all 4 layers
│   └── watchdog.py           # Daemon thread: fires at 570s, flushes partial output
│
├── engines/
│   ├── __init__.py
│   ├── local_slm.py          # llama-cpp-python wrapper (Qwen2.5-3B Q4_K_M)
│   └── remote_llm.py         # Async Fireworks API client (aiohttp + tenacity retry)
│
├── handlers/                 # One handler file per capability domain
│   ├── __init__.py
│   ├── _base.py              # Shared load_prompt_template utility
│   ├── code_utils.py         # Code difficulty classifier + extract/validate helpers
│   ├── factual.py            # → local SLM
│   ├── sentiment.py          # → local SLM (Positive / Negative / Neutral)
│   ├── ner.py                # → local SLM (JSON list output)
│   ├── summarization.py      # → local SLM
│   ├── math_handler.py       # → remote (CoT + numeric extraction, max 768 tokens)
│   ├── debug.py              # → local-first, API fallback (max 400 tokens)
│   ├── code_gen.py           # → local-first, API fallback (max 500 tokens)
│   ├── logic.py              # → remote (max 768 tokens)
│   ├── local_handlers.py     # Backwards-compat composite LocalGeneralHandler
│   └── remote_handlers.py    # RemoteGeneralHandler (escalation fallback)
│
├── prompts/                  # System prompt templates (.txt)
│   ├── factual.txt
│   ├── sentiment.txt
│   ├── ner.txt
│   ├── summarization.txt
│   ├── remote_math.txt       # CoT with few-shot examples → ANSWER: <number>
│   ├── remote_logic.txt      # Direct answer only, no explanation
│   ├── remote_code.txt       # Raw Python code output only
│   ├── remote_general.txt    # Escalation fallback prompt
│   ├── local_code_gen.txt    # Local code generation (no markdown)
│   └── local_code_debug.txt  # Local debug (corrected code only)
│
├── models/                   # Bundled GGUF weights (~1 GB, not tracked in git)
│
├── tests/
│   ├── fixtures/
│   │   ├── task.json                 # 3,235 consolidated tasks (classifier training data)
│   │   ├── expected_results.json     # Baseline expected answers for sample tasks
│   │   ├── sample_tasks.json         # Practice tasks from the hackathon guide
│   │   └── test_cases_60.json        # 60-task evaluation subset
│   ├── results/                      # Saved test run outputs
│   ├── eval_200_tasks.json           # 200-task evaluation dataset
│   ├── eval_results.json             # Evaluation results (200 tasks)
│   ├── evaluation_report.md          # Detailed performance report (93% accuracy)
│   ├── test_ast_eval.py
│   ├── test_cache.py
│   ├── test_classifier.py
│   ├── test_local_slm.py
│   ├── test_remote_llm.py
│   ├── test_router.py
│   └── test_integration.py
│
├── scripts/
│   ├── setup.sh                # 🚀 First-time setup (install everything)
│   ├── run.sh                  # ▶️  Run agent with custom input/output
│   ├── test_local.py           # 🧪 Run practice tasks locally + token stats
│   ├── prompt_benchmark.py     # 📊 Benchmark 5 prompting strategies (sentiment + summarization)
│   ├── download_model.sh       # Download GGUF weights from HuggingFace
│   └── simulate_grading.sh    # Docker run with 4GB RAM / 2 CPU constraints
│
├── output/                   # Generated results (git-ignored)
├── main.py                   # Entrypoint — async task processing with watchdog
├── Dockerfile                # Python 3.12-slim + uv package manager
├── entrypoint.sh             # Loads .env if present, then runs main.py
├── .env.example              # Template — copy to .env and fill in credentials
├── pyproject.toml            # ruff + mypy + pytest configuration
├── requirements.txt
└── requirements-dev.txt
```

---

## Getting Started

> **Lần đầu fork về?** Chỉ cần 1 lệnh:

```bash
bash scripts/setup.sh
```

Script này tự động:
- Tạo virtual environment (`.venv`)
- Cài tất cả dependencies từ `requirements.txt`
- Cài `llama-cpp-python` (CPU wheel, không cần C++ compiler)
- Download `Qwen2.5-3B Q4_K_M` GGUF (~2 GB)
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

### Benchmark prompting strategies

```bash
PYTHONPATH=. python scripts/prompt_benchmark.py
```

Benchmarks 5 strategies (baseline, zero-shot strict, few-shot, chain-of-thought, self-consistency 3×) across sentiment and summarization — outputs CSV + markdown summary to `output/`.

### Chạy unit tests

```bash
# Unit tests — không cần model (mocked)
PYTHONPATH=. pytest tests/test_ast_eval.py tests/test_cache.py \
    tests/test_remote_llm.py tests/test_router.py -v

# Classifier test (loads local GGUF model)
PYTHONPATH=. pytest tests/test_classifier.py -v

# Local SLM test (requires GGUF model)
PYTHONPATH=. pytest tests/test_local_slm.py -v

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

**Docker image features:**
- Uses `entrypoint.sh` (loads `.env` if present, then runs `main.py`)
- Bundles `Qwen2.5-3B Q4_K_M` GGUF (~2 GB) in `/app/models/` — the only model needed (used for both classification and local handlers)
- No torch / sentence-transformers / embedding model — smaller image, faster cold start

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | Yes (grading) | API key — injected by harness lúc chấm |
| `FIREWORKS_BASE_URL` | Yes (grading) | Proxy URL cho token counting — harness inject |
| `ALLOWED_MODELS` | Yes (grading) | Comma-separated model IDs — harness inject |
| `LOCAL_MODEL_PATH` | Optional | Đường dẫn GGUF (default: `models/qwen2.5-3b-instruct-q4_k_m.gguf`) |
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
| RAM | 4 GB | Qwen2.5 ~2 GB + MiniLM ~200 MB ≈ ~2.2 GB total ✅ |
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
- **Local-first code strategy** → easy/medium code tasks solved locally with AST validation → only hard tasks or failed validations use API tokens
- **Supervised PyTorch Classifier** → 100.00% routing accuracy → zero misroutes → zero unnecessary API token waste
- **Prompt compression** → strip filler phrases + output suffix → giảm tokens mỗi remote call
- **Per-category `max_tokens` budgets** → giới hạn output dài không cần thiết
- **Semantic cache** → dedup identical/similar prompts
- **Category-aware model selection** → pick best available model per task type from `ALLOWED_MODELS`
- **Escalation model preferences** → local escalations prefer `minimax-m3` for cost efficiency

---

## Evaluation Results

Local evaluation on 200 tasks (100 Factual + 100 Summarization) achieved:

| Metric | Value |
|---|---|
| **Global Accuracy** | 93.00% (186/200) |
| **Avg Latency** | 505.1 ms |
| Factual Knowledge | 89.00% accuracy, 598.6 ms avg |
| Text Summarization | 97.00% accuracy, 411.5 ms avg |

See [`tests/evaluation_report.md`](tests/evaluation_report.md) for detailed analysis.

---

## Development Notes

- **Python version**: 3.12 (Docker) / 3.11+ (host dev)
- **Package manager**: `uv` (in Docker), `pip` (host dev)
- **Classifier**: local Qwen2.5-3B with a GBNF grammar constraining output to one of the route labels (0 tokens, no separate embedding model / PyTorch head)
- **Local SLM**: `Qwen2.5-3B-Instruct Q4_K_M` via `llama-cpp-python`
- **Remote API**: `aiohttp` + `tenacity` retry (3 attempts, exponential backoff)
- **Math prompting**: CoT with few-shot examples, handles fractions/decimals, answer extraction via regex
- **Linting**: `ruff check .` (target: Python 3.11, line-length: 150)
- **Type checking**: `mypy .` (strict mode, Python 3.12)
- **Pre-commit**: `pre-commit run --all-files` (ruff + ruff-format + mypy + file checks)
- Không cần API key để chạy local SLM tasks và unit tests
