# handlers/sentiment.py
import logging

from engines.local_slm import LocalSLMEngine

logger = logging.getLogger(__name__)

_VALID = {"Positive", "Negative", "Neutral"}

_SYSTEM_PROMPT = (
    "You are a zero-filler text classification engine. "
    "Analyze the input text sentiment. Output EXACTLY one word "
    "from these options: Positive, Negative, Neutral. "
    "No preamble. No explanations. No markdown formatting."
)


class SentimentHandler:
    """Classifies sentiment locally. Zero remote tokens guaranteed.

    Decoding is pinned to temperature=0.0 (greedy) and max_tokens=2
    to force single-token output and eliminate conversational framing.
    Parse failures default to 'Neutral' locally — no __ESCALATE__ path.
    """

    def __init__(self) -> None:
        self.engine = LocalSLMEngine.get_instance()

    def handle(self, prompt: str) -> str:
        res = self.engine.generate(
            f"Text: {prompt}\nSentiment:",
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=2,
            temperature=0.0,
        )
        # __ESCALATE__ is unreachable at max_tokens=2 (no length truncation),
        # but guard defensively.
        if res == "__ESCALATE__":
            return "Neutral"
        cleaned = res.strip().title()
        for label in _VALID:
            if label in cleaned:
                return label
        logger.info("SentimentHandler: non-standard output '%s'. Defaulting to Neutral.", res)
        return "Neutral"
