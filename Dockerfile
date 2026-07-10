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

# Step 1: Install CPU-only torch first to avoid pulling 2.5 GB CUDA wheels
RUN uv pip install torch \
        --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining deps (sentence-transformers will reuse the torch above)
RUN uv pip install -r requirements.txt

# Step 3: Install llama-cpp-python via precompiled CPU wheel (avoids C++ compilation)
RUN uv pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Step 4: Pre-cache sentence-transformer model weights at build time
# Prevents runtime download within the 10-minute container limit
RUN python -c "
from sentence_transformers import SentenceTransformer
print('Pre-caching all-MiniLM-L6-v2...')
SentenceTransformer('all-MiniLM-L6-v2')
print('Model cached successfully.')
"

# Bundle GGUF model weights (~986 MB)
# Make sure to run scripts/download_model.sh before building
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
    LOCAL_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    LOCAL_N_GPU_LAYERS=0 \
    LOCAL_N_THREADS=2 \
    LOCAL_N_CTX=2048

CMD ["python", "/app/main.py"]
