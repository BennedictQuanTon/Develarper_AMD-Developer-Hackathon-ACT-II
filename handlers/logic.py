# handlers/logic.py
import logging

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class LogicHandler:
    """Routes logical reasoning / puzzle tasks to the remote LLM."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_logic.txt")

    async def handle(self, prompt: str) -> str:
        return await self.engine.generate(
            prompt=prompt,
            category="API_LOGIC",
            system_prompt=self.system_prompt,
            max_tokens=150,  # Token budget: concise final answer
        )
