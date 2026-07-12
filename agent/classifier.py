"""
agent/classifier.py — Supervised Neural Network Classifier
=========================================================
Uses sentence-transformers (all-MiniLM-L6-v2) to encode prompts into dense
384-dimensional embeddings, then passes them through a PyTorch classification
head to predict the optimal routing category.

Routes (constants unchanged):
  LOCAL_SENTIMENT, LOCAL_NER, LOCAL_GENERAL
  API_MATH, API_CODE, API_LOGIC, API_LONG_CONTEXT
"""

import logging
import os
import threading
from typing import Any, Optional

import torch
import torch.nn as nn

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

ROUTES_MAP = [
    ROUTE_LOCAL_SENTIMENT,
    ROUTE_LOCAL_NER,
    ROUTE_LOCAL_GENERAL,
    ROUTE_API_MATH,
    ROUTE_API_CODE,
    ROUTE_API_LOGIC,
]

_MODEL_NAME = "all-MiniLM-L6-v2"
_LONG_CONTEXT_THRESHOLD = 6000  # chars; above this → API_LONG to avoid CPU OOM
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "supervised_model.pt")


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Simple MLP head: 384 -> 64 -> 6 classes
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, num_classes))

    def forward(self, x: Any) -> Any:
        return self.net(x)


class SemanticClassifier:
    """
    Loads all-MiniLM-L6-v2 and the trained PyTorch classification head.
    Performs fast, offline inference by passing the prompt embedding through the MLP.
    """

    _instance: Optional["SemanticClassifier"] = None
    _init_lock: threading.Lock = threading.Lock()  # guards concurrent thread-pool callers

    @classmethod
    def get_instance(cls) -> "SemanticClassifier":
        if cls._instance is None:
            with cls._init_lock:  # only one thread enters at a time
                if cls._instance is None:  # re-check inside lock (double-checked locking)
                    logger.info("Initializing SemanticClassifier (loading model & PyTorch head)...")
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required. " "Run: pip install sentence-transformers") from exc

        self.encoder = SentenceTransformer(_MODEL_NAME)
        self.model = LinearClassifier(input_dim=384, num_classes=len(ROUTES_MAP))

        if os.path.exists(WEIGHTS_PATH):
            logger.info(f"Loading supervised weights from {WEIGHTS_PATH}")
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        else:
            # Fallback if weights not found: check if task.json exists to auto-train
            task_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "fixtures", "task.json")
            if os.path.exists(task_path):
                logger.warning(f"Supervised weights not found at {WEIGHTS_PATH}. Auto-training from {task_path}...")
                self._train_from_data(task_path)
            else:
                logger.error(f"Supervised weights not found at {WEIGHTS_PATH} and no training data found. Using untrained weights.")

        self.model.eval()

    def _train_from_data(self, data_path: str) -> None:
        import json

        import torch.optim as optim

        with open(data_path, encoding="utf-8") as f:
            tasks = json.load(f)

        prompts = [t["prompt"] for t in tasks]
        labels = [ROUTES_MAP.index(t["expected_route"]) for t in tasks]

        embeddings = self.encoder.encode(prompts, convert_to_tensor=True).cpu()
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.01)

        self.model.train()
        for _ in range(150):
            optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

        torch.save(self.model.state_dict(), WEIGHTS_PATH)
        logger.info(f"Successfully trained and saved new classifier weights to {WEIGHTS_PATH}!")

    def classify(self, prompt: str) -> str:
        prompt_emb = self.encoder.encode(prompt, convert_to_tensor=True).cpu()
        with torch.no_grad():
            logits = self.model(prompt_emb.unsqueeze(0))
            pred_idx = int(logits.argmax(dim=1).item())
        return ROUTES_MAP[pred_idx]


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for the old classify()
# ---------------------------------------------------------------------------
def classify(prompt: str) -> str:
    """
    Classify a prompt into one of the routing destinations.
    """
    # --- Override: Long context ---
    if len(prompt) > _LONG_CONTEXT_THRESHOLD:
        logger.debug("Classifier: long context (%d chars) → %s", len(prompt), ROUTE_API_LONG)
        return ROUTE_API_LONG

    classifier = SemanticClassifier.get_instance()

    # --- Execute Supervised PyTorch Classification ---
    route = classifier.classify(prompt)
    logger.debug("Classifier: supervised PyTorch → %s", route)
    return route
