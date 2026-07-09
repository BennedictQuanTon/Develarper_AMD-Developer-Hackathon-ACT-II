# handlers/math_handler.py
import logging
from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class MathHandler:
    """Routes complex math word problems to the remote LLM."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_math.txt")

    async def handle(self, prompt: str) -> str:
        return await self.engine.generate(
            prompt=prompt,
            category="API_MATH",
            system_prompt=self.system_prompt,
            max_tokens=50,  # Token budget: output only the numeric answer
        )
