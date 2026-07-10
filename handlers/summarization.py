# handlers/summarization.py
import logging
import re

from engines.local_slm import LocalSLMEngine

logger = logging.getLogger(__name__)

_FILLER_RE = re.compile(
    r"\b(please|kindly|summarize the following text|summarize the following" r"|i want a summary of|the following text)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = "Summarize the text in under 3 concise sentences. " "Output ONLY the final summary. No intro. No structural wrappers."

_WORD_COUNT_THRESHOLD = 1200


class SummarizationHandler:
    """Summarizes text locally for inputs under the word-count threshold.

    Word count gate:
      < 1200 words → local SLM (max_tokens=90, temperature=0.1)
      ≥ 1200 words → returns '__ESCALATE__' for async remote handling in router
    """

    def __init__(self) -> None:
        self.engine = LocalSLMEngine.get_instance()

    def _clean(self, text: str) -> str:
        """Strip filler phrases to trim input token volume."""
        text = _FILLER_RE.sub("", text)
        return " ".join(text.split())

    def handle(self, prompt: str) -> str:
        cleaned = self._clean(prompt)
        word_count = len(cleaned.split())
        if word_count >= _WORD_COUNT_THRESHOLD:
            logger.info(
                "SummarizationHandler: word_count=%d ≥ threshold=%d → __ESCALATE__",
                word_count,
                _WORD_COUNT_THRESHOLD,
            )
            return "__ESCALATE__"
        return self.engine.generate(
            cleaned,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=90,
            temperature=0.1,
        )
