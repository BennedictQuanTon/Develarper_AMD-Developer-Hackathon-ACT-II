# handlers/code_gen.py
import logging

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class CodeGenHandler:
    """Routes code generation tasks to a code-specialist remote model."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_code.txt")

    async def handle(self, prompt: str) -> str:
        return await self.engine.generate(
            prompt=prompt,
            category="API_CODE",
            system_prompt=self.system_prompt,
            max_tokens=500,  # Token budget for code generation tasks
        )
