# AMD AI Developer Hackathon 2026 — Track 1: Token-Efficient Agent

**Team**: Develarper

> **Hackathon Goal**: Build a containerized AI agent covering 8 capability domains, pass the LLM-Judge accuracy gate, and **minimize total Fireworks API tokens** for leaderboard ranking.

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
[L2]  Weighted Scoring Classifier     → 0 tokens, <1ms  (pure re module)
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
│   ├── classifier.py     # Weighted scoring router (pure re, 0 deps)
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
├── models/               # Bundled GGUF weights (~986 MB, not tracked in git)
├── tests/
│   ├── fixtures/
│   │   ├── sample_tasks.json      # 8 sample tasks (one per domain)
│   │   └── expected_results.json  # Baseline expected answers
│   ├── test_ast_eval.py
│   ├── test_cache.py
│   ├── test_classifier.py
│   ├── test_remote_llm.py   # Tests compress_prompt + model selection
│   ├── test_router.py       # Tests all 4 routing layers end-to-end
│   └── test_integration.py  # Full pipeline with mocked API
├── scripts/
│   ├── download_model.sh    # Download GGUF weights from HuggingFace
│   └── simulate_grading.sh  # Docker run with 4GB RAM / 2 CPU constraints
├── main.py                  # Entrypoint
├── Dockerfile
├── .env.example             # Template — copy to .env and fill in credentials
└── requirements.txt
```

---

## Getting Started (For Teammates)

> Follow these steps in order after cloning the repo.

### Step 0: Clone the repo

```bash
git clone https://github.com/<your-org>/Develarper_AMD-Developer-Hackathon-ACT-II.git
cd Develarper_AMD-Developer-Hackathon-ACT-II
```

### Step 1: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install llama-cpp-python (precompiled CPU wheel — no C++ compiler needed)
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> **Apple Silicon (M1/M2/M3)** or **CUDA GPU**: You can install the Metal/CUDA wheel instead for faster inference during development. But make sure `LOCAL_N_GPU_LAYERS=0` in `.env` for Docker (CPU-only inside container).

### Step 3: Download the local SLM weights

```bash
bash scripts/download_model.sh
```

This downloads `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1 GB) into the `models/` directory.

> `models/` is in `.gitignore` — weights are **never committed to git**.

### Step 4: Configure environment variables

```bash
cp .env.example .env
```

Then open `.env` and fill in:

```
FIREWORKS_API_KEY=<your_key>       # ← Required for remote API calls (Phase 5+)
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
```

> Without `FIREWORKS_API_KEY`, local SLM tasks (Sentiment, NER, Factual, Summarization) still work. Only remote escalation calls will fail.

### Step 5: Run the test suite

```bash
# Run all unit + router tests (no model loading — instant)
PYTHONPATH=. pytest tests/test_ast_eval.py tests/test_cache.py \
    tests/test_classifier.py tests/test_remote_llm.py tests/test_router.py -v

# Run the full integration test (mocked API, loads local SLM)
PYTHONPATH=. python tests/test_integration.py
```

Expected output: **33 passed, 0 warnings** for unit tests.

### Step 6: Run locally against fixture tasks

```bash
mkdir -p output

INPUT_PATH=tests/fixtures/sample_tasks.json \
OUTPUT_PATH=output/results.json \
PYTHONPATH=. python main.py

cat output/results.json
```

---

## Docker Build & Grading Simulation

> Download the model weights first (Step 3 above) before building.

```bash
# 1. Build image (use --platform linux/amd64 for AMD submission)
docker buildx build --platform linux/amd64 \
    -t <your-dockerhub-username>/develarper-agent:latest .

# 2. Check image size (must be < 10 GB compressed)
docker images <your-dockerhub-username>/develarper-agent:latest

# 3. Simulate the grading environment (4 GB RAM, 2 CPUs, no network)
FIREWORKS_API_KEY=your_key bash scripts/simulate_grading.sh

# 4. View results
cat output_test/results.json

# 5. Push to Docker Hub when ready to submit
docker push <your-dockerhub-username>/develarper-agent:latest
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | Phase 5+ | API key for Fireworks AI (injected by harness at judging) |
| `FIREWORKS_BASE_URL` | Phase 5+ | Proxy URL for token counting (harness injects this) |
| `ALLOWED_MODELS` | Phase 5+ | Comma-separated list of approved model IDs |
| `LOCAL_MODEL_PATH` | Optional | Path to GGUF file (default: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`) |
| `LOCAL_N_GPU_LAYERS` | Optional | GPU layers for llama.cpp (`0` = CPU-only, `-1` = auto Metal on Mac) |
| `LOCAL_N_THREADS` | Optional | CPU threads for llama.cpp (default: `2`) |
| `LOCAL_N_CTX` | Optional | Context window size (default: `2048`) |
| `INPUT_PATH` | Runtime | Path to input `tasks.json` (default: `/input/tasks.json`) |
| `OUTPUT_PATH` | Runtime | Path to write `results.json` (default: `/output/results.json`) |

> **Security**: Never commit `.env` to git. It is already in `.gitignore`.

---

## Grading Constraints

| Constraint | Limit | Our Approach |
|---|---|---|
| RAM | 4 GB | SLM uses ~1.1 GB; total peak ~2.0 GB ✅ |
| CPUs | 2 vCPUs | `n_threads=2` in llama.cpp |
| Image size | 10 GB compressed | ~1.1–1.5 GB estimated ✅ |
| Runtime | 10 minutes | Watchdog fires at 570s, flushes partial results |
| Architecture | linux/amd64 | Build with `--platform linux/amd64` |

---

## Scoring Strategy

```
Minimize: Σ (input_tokens + output_tokens) sent via FIREWORKS_BASE_URL
Subject to: accuracy ≥ threshold (binary gate — must pass first)
```

- **Local execution = 0 Fireworks tokens** → maximize this
- **Prompt compression** reduces tokens on every remote call
- **Per-category `max_tokens` budgets** prevent over-generation
- **Semantic cache** deduplicates identical/near-identical prompts

---

## Development Notes

- **Python version**: 3.10 (Docker) / 3.11+ (host dev)
- **Linting**: `ruff` — run `ruff check .` before committing
- **Type checking**: `mypy` — run `mypy .`
- **Pre-commit hooks**: `pre-commit run --all-files`
- **No API key needed** to run unit tests and test local SLM flows
