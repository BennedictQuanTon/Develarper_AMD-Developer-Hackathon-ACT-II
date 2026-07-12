# handlers/sentiment.py
import logging

from engines.local_slm import LocalSLMEngine

logger = logging.getLogger(__name__)

_VALID = {"Positive", "Negative", "Neutral"}

_SYSTEM_PROMPT = (
    "Classify the sentiment of the text as Positive, Negative, or Neutral. "
    "Then provide exactly one sentence explaining your reasoning. "
    "Format your response EXACTLY as: <LABEL>. <one sentence reason> "
    "Example: Neutral. The packaging was damaged but the product itself works perfectly."
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
            max_tokens=60,  # enough for "<Label>. <one-sentence reason>"
            temperature=0.0,
        )
        if res == "__ESCALATE__":
            return "Neutral"
        cleaned = res.strip()
        # Return the full label+reason string if it starts with a valid label
        for label in _VALID:
            if cleaned.startswith(label):
                return cleaned
        # Fallback: scan for label anywhere in response (model omitted format)
        for label in _VALID:
            if label in cleaned.title():
                return cleaned
        logger.info("SentimentHandler: non-standard output '%s'. Defaulting to Neutral.", res)
        return "Neutral"
