FROM python:3.12-slim

# Build for AMD64 (hackathon requirement)
# Local build: docker buildx build --platform linux/amd64 -t <username>/develarper-agent:latest .

WORKDIR /app

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Enable system-wide package installation for uv
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# Install remaining deps (llama-cpp-python installed separately via CPU wheel).
# torch + sentence-transformers were removed: classification now uses the local
# Qwen GGUF model directly, so no embedding model / PyTorch head is needed.
RUN uv pip install --no-cache -r requirements.txt

# Install llama-cpp-python via precompiled CPU wheel (avoids C++ compilation)
RUN uv pip install --no-cache llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Bundle GGUF model weights (~2 GB)
COPY models/ /app/models/

# System prompt templates
COPY prompts/ /app/prompts/

# Application packages
COPY agent/ /app/agent/
COPY engines/ /app/engines/
COPY handlers/ /app/handlers/
COPY main.py /app/
COPY entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh && mkdir -p /input /output

# Env defaults (harness will override FIREWORKS_* at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_N_GPU_LAYERS=0 \
    LOCAL_N_THREADS=2 \
    LOCAL_N_CTX=2048

ENTRYPOINT ["/app/entrypoint.sh"]
