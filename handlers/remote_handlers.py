# handlers/remote_handlers.py
"""
Fallback remote handlers for escalation and long-context tasks.
These are NOT the primary handlers — they handle overflow from local SLM.
"""

import logging

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class RemoteGeneralHandler:
    """Escalation fallback — handles any category that local SLM couldn't handle."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_general.txt")

    async def handle(self, prompt: str, category: str = "LOCAL_GENERAL") -> str:
        # Category-specific token budgets
        max_tokens = {
            "LOCAL_GENERAL": 80,
            "LOCAL_SENTIMENT": 60,  # label + one-sentence reason
            "LOCAL_NER": 200,
            "API_LONG_CONTEXT": 200,
        }.get(category, 80)

        # Preserve schema constraints for structured-output categories
        if category == "LOCAL_NER":
            # Must keep JSON list format intact — remote_general.txt is unstructured prose
            system_prompt = load_prompt_template("ner.txt")
        elif category == "LOCAL_SENTIMENT":
            system_prompt = (
                "Classify the sentiment of the text as Positive, Negative, or Neutral. "
                "Then provide exactly one sentence explaining your reasoning. "
                "Format: <LABEL>. <one sentence reason>  "
                "Example: Neutral. The packaging was damaged but the product itself works perfectly."
            )
        else:
            system_prompt = self.system_prompt

        return await self.engine.generate(
            prompt=prompt,
            category=category,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
