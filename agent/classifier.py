"""
agent/classifier.py — Zero-Shot Semantic Classifier
=====================================================
Uses sentence-transformers (all-MiniLM-L6-v2) to classify prompts by cosine
similarity against pre-computed label anchor embeddings.

Design:
  - SemanticClassifier singleton: model loaded once at first call (~1s warm-up)
  - 6 label anchors (multiple sentences per route → mean pooled embedding)
  - classify() is a drop-in replacement: same public signature as before
  - Two hard structural overrides (prompt length, code block detection)
  - Zero Fireworks API calls — runs entirely local → 0 tokens toward score

Routes (constants unchanged — router.py imports these):
  LOCAL_SENTIMENT, LOCAL_NER, LOCAL_GENERAL
  API_MATH, API_CODE, API_LOGIC, API_LONG_CONTEXT
"""

import logging
import re
from typing import Any, Optional

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_LONG_CONTEXT_THRESHOLD = 6000  # chars; above this → API_LONG to avoid CPU OOM
_CODE_STRUCTURAL_PATTERN = re.compile(r"```|def\s+\w+\s*\(|class\s+\w+\s*:|public\s+static|console\.log")
_CODE_STRUCTURAL_SCORE_THRESHOLD = 0.35  # if code anchor score < this, hard-override wins

# ---------------------------------------------------------------------------
# Label Anchors
# Each route has multiple representative sentences.
# SemanticClassifier will mean-pool them into a single route embedding.
# ---------------------------------------------------------------------------
LABEL_ANCHORS: dict[str, list[str]] = {
    ROUTE_LOCAL_SENTIMENT: [
        "Classify the sentiment of this text as positive, negative, or neutral.",
        "What is the emotional tone of this review or statement?",
        "Analyze whether this feedback is positive, negative, or neutral.",
        "Label the mood or feeling expressed in this sentence.",
        "Is this opinion positive or negative? Justify your classification.",
    ],
    ROUTE_LOCAL_NER: [
        "Extract all named entities: persons, organizations, locations, and dates.",
        "Identify and label all proper nouns and entity types in this passage.",
        "Find all mentions of people, companies, places, and times.",
        "List the entities and their categories found in this text.",
        "Extract named entities and classify them by type.",
    ],
    ROUTE_LOCAL_GENERAL: [
        "Answer this factual knowledge question accurately.",
        "What is the definition or explanation of this concept or term?",
        "Summarize or condense this passage into a shorter form.",
        "What is the capital city, historical fact, or general knowledge being asked?",
        "Provide a brief, factual explanation of what is being asked.",
        "Give a concise summary of the following text.",
    ],
    ROUTE_API_MATH: [
        "Calculate the result of this arithmetic or mathematical problem.",
        "Solve this word problem involving numbers, percentages, or quantities.",
        "Compute the numerical answer to this equation or multi-step calculation.",
        "How many items remain after these numeric operations?",
        "Find the percentage, ratio, or projected value from these numbers.",
    ],
    ROUTE_API_CODE: [
        "Write a Python function that implements this programming specification.",
        "Debug this code snippet and fix the bug or syntax error.",
        "Generate working code to accomplish this programming task.",
        "This function has a bug — find the issue and provide the corrected implementation.",
        "Write a program or script that performs the described operation.",
    ],
    ROUTE_API_LOGIC: [
        "Solve this logical deduction puzzle given the stated constraints.",
        "Determine who satisfies all conditions in this logic problem.",
        "If these statements are true, what can be logically deduced?",
        "Apply deductive reasoning to find the unique answer from these clues.",
        "Given these constraints, which solution satisfies all conditions?",
    ],
}


# ---------------------------------------------------------------------------
# SemanticClassifier — Singleton
# ---------------------------------------------------------------------------
class SemanticClassifier:
    """
    Loads all-MiniLM-L6-v2 once and pre-computes mean anchor embeddings
    for all routes. Subsequent classify() calls are pure cosine similarity.
    """

    _instance: Optional["SemanticClassifier"] = None

    @classmethod
    def get_instance(cls) -> "SemanticClassifier":
        if cls._instance is None:
            logger.info("Initializing SemanticClassifier (first call — loading model)…")
            cls._instance = cls()
            logger.info("SemanticClassifier ready.")
        return cls._instance

    def __init__(self) -> None:
        # Import here so import errors surface early with a clear message
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import util as st_util
        except ImportError as exc:
            raise ImportError("sentence-transformers is not installed. " "Run: pip install sentence-transformers") from exc

        self._util = st_util
        self.model = SentenceTransformer(_MODEL_NAME)

        # Pre-compute mean-pooled anchor embedding per route
        self.route_embeddings: dict[str, Any] = {}
        for route, anchors in LABEL_ANCHORS.items():
            embs = self.model.encode(anchors, convert_to_tensor=True)
            # Mean pool across anchor sentences
            self.route_embeddings[route] = embs.mean(dim=0)

        logger.info(
            "SemanticClassifier: pre-computed embeddings for %d routes.",
            len(self.route_embeddings),
        )

    def classify(self, prompt: str) -> str:
        """Return the best-matching route string for the given prompt."""
        prompt_emb = self.model.encode(prompt, convert_to_tensor=True)

        scores: dict[str, float] = {route: float(self._util.cos_sim(prompt_emb, anchor_emb)) for route, anchor_emb in self.route_embeddings.items()}

        best = max(scores, key=lambda k: scores[k])
        logger.debug(
            "SemanticClassifier scores: %s → best=%s",
            {r: f"{s:.3f}" for r, s in scores.items()},
            best,
        )
        return best

    def get_score(self, prompt: str, route: str) -> float:
        """Return cosine similarity score for a specific route (used by hard overrides)."""
        prompt_emb = self.model.encode(prompt, convert_to_tensor=True)
        return float(self._util.cos_sim(prompt_emb, self.route_embeddings[route]))


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for the old classify()
# ---------------------------------------------------------------------------
def classify(prompt: str) -> str:
    """
    Classify a prompt into one of the 6 routing destinations.

    Override order:
      1. Long context (> 6000 chars)        → ROUTE_API_LONG   (prevent CPU OOM)
      2. Structural code markers detected    → ROUTE_API_CODE   (unless embedding
         already strongly agrees with code)
      3. Semantic embedding (cosine sim)     → winning route

    Returns one of the ROUTE_* constants.
    """
    # --- Override 1: Long context ---
    if len(prompt) > _LONG_CONTEXT_THRESHOLD:
        logger.debug("Classifier: long context (%d chars) → %s", len(prompt), ROUTE_API_LONG)
        return ROUTE_API_LONG

    classifier = SemanticClassifier.get_instance()

    p = prompt.lower()

    # --- Override: Creative writing / story fallback ---
    if re.search(r"\bstory\b|\bpoem\b|\bessay\b|\bwrite a letter\b", p):
        logger.debug("Classifier: creative writing override → %s", ROUTE_LOCAL_GENERAL)
        return ROUTE_LOCAL_GENERAL

    # --- Override: Explicit code / script generation ---
    if re.search(r"\bpython script\b|\bwrite code\b|\bcode snippet\b", p):
        logger.debug("Classifier: code script override → %s", ROUTE_API_CODE)
        return ROUTE_API_CODE

    # --- Override 2: Structural code markers ---
    if _CODE_STRUCTURAL_PATTERN.search(prompt):
        code_score = classifier.get_score(prompt, ROUTE_API_CODE)
        if code_score < _CODE_STRUCTURAL_SCORE_THRESHOLD:
            # Structural signal is strong but embedding doesn't agree → trust structure
            logger.debug(
                "Classifier: structural code override (embedding code score=%.3f < %.2f) → %s",
                code_score,
                _CODE_STRUCTURAL_SCORE_THRESHOLD,
                ROUTE_API_CODE,
            )
            return ROUTE_API_CODE
        # else: embedding already leans toward code anyway, fall through to embedding

    # --- Primary: Embedding-based classification ---
    route = classifier.classify(prompt)
    logger.debug("Classifier: embedding → %s", route)
    return route
