# handlers/sentiment.py
import logging
from engines.local_slm import LocalSLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)

_VALID = {"Positive", "Negative", "Neutral"}


class SentimentHandler:
    """Classifies sentiment locally. Returns Positive/Negative/Neutral or __ESCALATE__."""

    def __init__(self) -> None:
        self.engine = LocalSLMEngine.get_instance()
        self.system_prompt = load_prompt_template("sentiment.txt")

    def handle(self, prompt: str) -> str:
        res = self.engine.generate(prompt, self.system_prompt, max_tokens=20)
        if res == "__ESCALATE__":
            return "__ESCALATE__"
        cleaned = res.strip().title()
        for label in _VALID:
            if label in cleaned:
                return label
        logger.info("SentimentHandler: non-standard output '%s'. Escalating.", res)
        return "__ESCALATE__"
