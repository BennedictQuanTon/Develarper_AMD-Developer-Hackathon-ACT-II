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
        # Set max_tokens based on category
        max_tokens = {
            "LOCAL_GENERAL":    80,
            "LOCAL_SENTIMENT":  20,
            "LOCAL_NER":       200,
            "API_LONG_CONTEXT": 200,
        }.get(category, 80)

        return await self.engine.generate(
            prompt=prompt,
            category=category,
            system_prompt=self.system_prompt,
            max_tokens=max_tokens,
        )
