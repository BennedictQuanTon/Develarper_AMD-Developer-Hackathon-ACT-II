import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.cache import SemanticCache


@pytest.mark.asyncio
async def test_integration_end_to_end(tmp_path: Path) -> None:
    input_file = os.path.join(os.path.dirname(__file__), "fixtures/sample_tasks.json")
    output_file = str(tmp_path / "results.json")

    os.environ["INPUT_PATH"] = input_file
    os.environ["OUTPUT_PATH"] = output_file
    os.environ["LOCAL_N_GPU_LAYERS"] = "0"

    with (
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
        patch("handlers.factual.LocalSLMEngine.get_instance") as mock_f,
        patch("handlers.sentiment.LocalSLMEngine.get_instance") as mock_s,
        patch("handlers.ner.LocalSLMEngine.get_instance") as mock_n,
        patch("handlers.summarization.LocalSLMEngine.get_instance") as mock_su,
    ):
        mock_remote.return_value = "Mocked Remote Response"

        # Configure local SLM mocks
        fake_factual = MagicMock()
        fake_factual.generate.return_value = "Paris"
        mock_f.return_value = fake_factual

        fake_sentiment = MagicMock()
        fake_sentiment.generate.return_value = "Positive"
        mock_s.return_value = fake_sentiment

        fake_ner = MagicMock()
        fake_ner.generate.return_value = '[{"entity": "Barack Obama", "type": "Person"}, {"entity": "Hawaii", "type": "Location"}]'
        mock_n.return_value = fake_ner

        fake_summary = MagicMock()
        fake_summary.generate.return_value = "AI simulates human intelligence in machines."
        mock_su.return_value = fake_summary

        import main

        main.INPUT_PATH = input_file
        main.OUTPUT_PATH = output_file
        main._completed.clear()
        main._cache = SemanticCache()
        from agent.router import AgentRouter

        main._router = AgentRouter(cache=main._cache)

        await main.main()

    assert os.path.exists(output_file)
    with open(output_file) as f:
        results = json.load(f)

    assert len(results) == 8
    result_map = {r["task_id"]: r["answer"] for r in results}

    # Layer 1b: AST Math should be resolved deterministically (0 tokens)
    assert result_map["math_1"] == "4104"

    # All task IDs must be present
    for tid in ["factual_1", "sentiment_1", "summarization_1", "ner_1", "debug_1", "logic_1", "code_gen_1"]:
        assert tid in result_map
        assert result_map[tid], f"{tid} answer is empty"


@pytest.mark.asyncio
async def test_sentiment_zero_remote_calls() -> None:
    """Sentiment must resolve locally — RemoteLLMEngine.generate must never be called.

    Patches:
    - engines.local_slm.LocalSLMEngine.get_instance: prevents real model loading
      in every handler (FactualHandler, NERHandler, SummarizationHandler, etc.)
    - agent.classifier.classify: forces ROUTE_LOCAL_SENTIMENT so the dispatch
      path is deterministic regardless of classifier weights.
    - engines.remote_llm.RemoteLLMEngine.generate: monitored for zero calls.
    """
    from agent.classifier import ROUTE_LOCAL_SENTIMENT

    with (
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
        patch("engines.local_slm.LocalSLMEngine.get_instance") as mock_slm_factory,
        patch("agent.router.classify", return_value=ROUTE_LOCAL_SENTIMENT),
    ):
        mock_remote.return_value = "Mocked remote — should NOT be reached"

        fake_slm = MagicMock()
        fake_slm.generate.return_value = "Positive"
        mock_slm_factory.return_value = fake_slm

        from agent.cache import SemanticCache
        from agent.router import AgentRouter

        router = AgentRouter(cache=SemanticCache())
        result = await router.route(
            "test_sent_001",
            "The product arrived on time and works perfectly as advertised.",
        )

        assert result in {"Positive", "Negative", "Neutral"}
        mock_remote.assert_not_called()
