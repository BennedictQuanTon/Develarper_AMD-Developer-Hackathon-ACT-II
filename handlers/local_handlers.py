# handlers/local_handlers.py
"""
Kept for backwards compatibility. All logic now lives in individual handler files.
"""

from handlers._base import load_prompt_template
from handlers.factual import FactualHandler
from handlers.ner import NERHandler as LocalNERHandler
from handlers.sentiment import SentimentHandler as LocalSentimentHandler
from handlers.summarization import SummarizationHandler


class LocalGeneralHandler:
    """Composite handler for factual questions and short summarization."""

    def __init__(self) -> None:
        self._factual = FactualHandler()
        self._summary = SummarizationHandler()

    def handle(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ["summarize", "summary", "tldr"]):
            return self._summary.handle(prompt)
        return self._factual.handle(prompt)


__all__ = [
    "load_prompt_template",
    "LocalSentimentHandler",
    "LocalNERHandler",
    "LocalGeneralHandler",
]
