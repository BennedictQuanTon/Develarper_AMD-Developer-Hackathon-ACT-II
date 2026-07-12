"""
agent/classifier.py — LLM-driven zero-token classifier
======================================================
The local Qwen model chooses the routing category. Output is constrained by a
GBNF grammar so the model can ONLY emit a valid label.
"""

import logging
import os

from engines.local_slm import LocalSLMEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route constants (imported by router.py — do NOT rename)
# ---------------------------------------------------------------------------
ROUTE_LOCAL_SENTIMENT = "LOCAL_SENTIMENT"
ROUTE_LOCAL_NER = "LOCAL_NER"
ROUTE_LOCAL_GENERAL = "LOCAL_GENERAL"  # covers factual + summarization
ROUTE_API_MATH = "API_MATH"
ROUTE_API_CODE = "API_CODE"
ROUTE_API_LOGIC = "API_LOGIC"
ROUTE_API_LONG = "API_LONG_CONTEXT"

_LABELS = [
    ROUTE_LOCAL_SENTIMENT,
    ROUTE_LOCAL_NER,
    ROUTE_LOCAL_GENERAL,
    ROUTE_API_MATH,
    ROUTE_API_CODE,
    ROUTE_API_LOGIC,
]

_LONG_CONTEXT_THRESHOLD = 6000  # chars; above this → API_LONG to avoid CPU OOM

_SYSTEM_PROMPT = (
    "You are a task router. Read the user task and output EXACTLY ONE label "
    "describing its type. Definitions:\n"
    "LOCAL_SENTIMENT = classify sentiment or emotion of text.\n"
    "LOCAL_NER = extract named entities (people, places, organizations, dates).\n"
    "LOCAL_GENERAL = factual knowledge question, definition, or summarization request.\n"
    "API_MATH = arithmetic calculation or math word problem that needs a numeric answer.\n"
    "API_CODE = write, generate, fix, or debug programming code.\n"
    "API_LOGIC = logical reasoning puzzle, deduction, or constraint problem.\n"
    "Examples:\n"
    "Q: What is the capital of France? → LOCAL_GENERAL\n"
    "Q: What is the boiling point of water in degrees Celsius? → LOCAL_GENERAL\n"
    "Q: Who wrote 'One Hundred Years of Solitude'? → LOCAL_GENERAL\n"
    "Q: Classify the sentiment of this review → LOCAL_SENTIMENT\n"
    "Q: Extract named entities from this text → LOCAL_NER\n"
    "Q: Calculate 342 * 12 → API_MATH\n"
    "Q: Write a Python function to sort a list → API_CODE\n"
    "Q: If A is taller than B and B is taller than C, is A taller than C? → API_LOGIC\n"
    "Output only the label, nothing else."
)

# GBNF grammar forces output to be one of the valid route labels.
_GRAMMAR_STR = "root ::= " + " | ".join(f'"{label}"' for label in _LABELS)

_grammar = None


def _get_grammar():
    global _grammar
    if _grammar is None:
        from llama_cpp import LlamaGrammar

        _grammar = LlamaGrammar.from_string(_GRAMMAR_STR)
    return _grammar


def classify(prompt: str) -> str:
    """
    Classify a prompt into one of the routing destinations.
    """
    if len(prompt) > _LONG_CONTEXT_THRESHOLD:
        logger.debug("Classifier: long context (%d chars) → %s", len(prompt), ROUTE_API_LONG)
        return ROUTE_API_LONG

    engine = LocalSLMEngine.get_instance()
    raw = engine.generate(
        prompt=f"Task:\n{prompt}\n\nLabel:",
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=8,
        temperature=0.0,
        grammar=_get_grammar(),
    ).strip()

    if raw in _LABELS:
        logger.debug("Classifier: LLM → %s", raw)
        return raw

    logger.warning("Classifier: invalid LLM label '%s' → fallback LOCAL_GENERAL", raw)
    return ROUTE_LOCAL_GENERAL
