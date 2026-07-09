#!/bin/bash
set -e

# Load .env file if it exists at /app/.env
if [ -f /app/.env ]; then
    echo "Loading environment variables from /app/.env"
    export $(grep -v '^#' /app/.env | xargs)
fi

# Execute main Python orchestrator
exec python /app/main.py
