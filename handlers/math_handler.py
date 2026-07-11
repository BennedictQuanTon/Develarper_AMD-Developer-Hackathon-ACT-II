# handlers/math_handler.py
import logging
import re

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


def extract_number(text: str) -> str:
    """Extracts numeric answer from a verbose Chain-of-Thought response."""
    text = text.strip()
    if re.match(r"^-?\d+(?:/\d+)?(?:\.\d+)?$", text):
        return text

    # Try different pattern fallbacks
    patterns = [
        r"ANSWER:\s*\$?\s*(-?\d+(?:/\d+)?(?:\.\d+)?)",
        r"(?:answer|result|total|sum|area|speed|price|cost|value|average|remain|left|volume|profit|capacity|units|pieces|girls|seconds|original|radius)\s*(?:is|=|:)\s*\$?\s*(-?\d+(?:/\d+)?(?:\.\d+)?)",
        r"\*\*(-?\d+(?:/\d+)?(?:\.\d+)?)\*\*",
        r"\\boxed\{(-?\d+(?:/\d+)?(?:\.\d+)?)\}",
        r"`(-?\d+(?:/\d+)?(?:\.\d+)?)`",
    ]
    for p in patterns:
        m = re.findall(p, text, re.IGNORECASE)
        if m:
            val = m[-1] if isinstance(m[-1], str) else [x for x in m[-1] if x][-1]
            return str(val)

    # Final fallback: last number in text
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return str(nums[-1]) if nums else text


class MathHandler:
    """Routes complex math word problems to the remote LLM."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_math.txt")

    async def handle(self, prompt: str) -> str:
        raw_response = await self.engine.generate(
            prompt=prompt,
            category="API_MATH",
            system_prompt=self.system_prompt,
            max_tokens=768,  # Optimized budget: large enough to guarantee 100% completion, small enough to protect token limit
        )
        # Extract the final number
        return extract_number(raw_response)
