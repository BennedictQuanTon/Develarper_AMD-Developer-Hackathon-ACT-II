import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.classifier import (
    ROUTE_API_CODE,
    ROUTE_API_LOGIC,
    ROUTE_API_MATH,
    ROUTE_LOCAL_GENERAL,
    ROUTE_LOCAL_NER,
    ROUTE_LOCAL_SENTIMENT,
)


def fake_classify_generate(prompt: str, **kwargs: Any) -> str:
    p = prompt.lower()
    if "code" in p or "python" in p or "c++" in p or "bug" in p or "script" in p or "error" in p:
        return ROUTE_API_CODE
    if "sentiment" in p or "emotion" in p or "tone" in p:
        return ROUTE_LOCAL_SENTIMENT
    if "entity" in p or "entities" in p or "name" in p:
        return ROUTE_LOCAL_NER
    if "calculate" in p or "solve for x" in p or "derivative" in p or "percentage" in p:
        return ROUTE_API_MATH
    if "taller" in p or "riddle" in p or "constraint" in p or ">" in p or "logic" in p:
        return ROUTE_API_LOGIC
    return ROUTE_LOCAL_GENERAL


@pytest.fixture(autouse=True)
def mock_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Automatically mock the ClassifierEngine and LocalSLMEngine if the model file is not present.
    This allows the test suite to pass in CI environments (like GitHub Actions)
    where the 2GB .gguf file is excluded by .dockerignore/.gitignore.
    """
    model_path = os.environ.get("LOCAL_MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
    if not os.path.exists(model_path):
        fake_classifier = MagicMock()
        fake_classifier.generate.side_effect = fake_classify_generate

        fake_local = MagicMock()
        fake_local.generate.return_value = "Mocked Local Generation"

        monkeypatch.setattr("engines.local_slm.ClassifierEngine.get_instance", lambda: fake_classifier)
        monkeypatch.setattr("engines.local_slm.LocalSLMEngine.get_instance", lambda: fake_local)
