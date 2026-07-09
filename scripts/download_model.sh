#!/bin/bash
set -e

mkdir -p models

MODEL_PATH="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)..."
    if command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli..."
        huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models --local-dir-use-symlinks False
    else
        echo "huggingface-cli not found, falling back to curl..."
        curl -L -o "$MODEL_PATH" "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    fi
    echo "Download completed."
else
    echo "Model already exists at $MODEL_PATH."
fi
