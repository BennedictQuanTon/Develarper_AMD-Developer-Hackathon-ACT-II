# handlers/summarization.py
import logging

from engines.local_slm import LocalSLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class SummarizationHandler:
    """Summarizes text locally (for texts ≤6000 chars)."""

    def __init__(self) -> None:
        self.engine = LocalSLMEngine.get_instance()
        self.system_prompt = load_prompt_template("summarization.txt")

    def handle(self, prompt: str) -> str:
        return self.engine.generate(prompt, self.system_prompt, max_tokens=250)
