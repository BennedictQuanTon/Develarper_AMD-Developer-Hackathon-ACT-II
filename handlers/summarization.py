# handlers/summarization.py
import logging
import re

from engines.local_slm import LocalSLMEngine

logger = logging.getLogger(__name__)

_FILLER_RE = re.compile(
    r"\b(please|kindly|summarize the following text|summarize the following" r"|i want a summary of|the following text)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT_PROSE = "Summarize the text in under 3 concise sentences. " "Output ONLY the final summary. No intro. No structural wrappers."

_SYSTEM_PROMPT_BULLETS = (
    "Summarize the text in EXACTLY 3 bullet points. "
    "Each bullet point must be 15 words or fewer. "
    "Start each bullet with '• '. No intro, no headers, no extra text."
)

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
                "SummarizationHandler: word_count=%d >= threshold=%d → __ESCALATE__",
                word_count,
                _WORD_COUNT_THRESHOLD,
            )
            return "__ESCALATE__"

        # Detect bullet-format directive in the original prompt
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["bullet", "bullet point", "• ", "points"]):
            system_prompt = _SYSTEM_PROMPT_BULLETS
            max_tok = 120  # 3 bullets × ~15 words × ~1.3 tokens/word
        else:
            system_prompt = _SYSTEM_PROMPT_PROSE
            max_tok = 90

        return self.engine.generate(
            cleaned,
            system_prompt=system_prompt,
            max_tokens=max_tok,
            temperature=0.1,
        )
