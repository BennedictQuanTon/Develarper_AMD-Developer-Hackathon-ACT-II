# handlers/_base.py
"""Shared utilities for all handlers."""
import logging
import os

logger = logging.getLogger(__name__)


def load_prompt_template(filename: str) -> str:
    """Load a system-prompt template from the prompts/ directory."""
    # Walk up from handlers/ to project root, then into prompts/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "prompts", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    logger.warning("Prompt template not found: %s", path)
    return ""
