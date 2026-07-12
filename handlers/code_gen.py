# handlers/code_gen.py
"""Code generation handler — direct remote API routing for speed and correctness."""

import logging

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template
from handlers.code_utils import extract_code

logger = logging.getLogger(__name__)


class CodeGenHandler:
    """Generates Python code using the remote model."""

    def __init__(self) -> None:
        self.remote = RemoteLLMEngine()
        self.remote_prompt = load_prompt_template("remote_code.txt")

    async def handle(self, prompt: str) -> str:
        raw = await self.remote.generate(
            prompt=prompt,
            category="API_CODE",
            system_prompt=self.remote_prompt,
            max_tokens=500,
        )
        return extract_code(raw)
