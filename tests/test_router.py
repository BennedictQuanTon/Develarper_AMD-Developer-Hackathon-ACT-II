from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.cache import SemanticCache
from agent.router import AgentRouter


@pytest.fixture()
def router() -> AgentRouter:
    cache = SemanticCache()
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance") as mock_slm,
        patch("handlers.sentiment.LocalSLMEngine.get_instance") as mock_slm2,
        patch("handlers.ner.LocalSLMEngine.get_instance") as mock_slm3,
        patch("handlers.summarization.LocalSLMEngine.get_instance") as mock_slm4,
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as _,
    ):
        fake_engine = MagicMock()
        fake_engine.generate.return_value = "Local answer"
        mock_slm.return_value = fake_engine
        mock_slm2.return_value = fake_engine
        mock_slm3.return_value = "[]"
        mock_slm4.return_value = fake_engine
        return AgentRouter(cache=cache)


# ---------------------------------------------------------------------------
# Layer 1a: Cache
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cache_hit() -> None:
    cache = SemanticCache()
    cache.set("What is 2+2?", "4")
    with patch("agent.router.AgentRouter._dispatch", new_callable=AsyncMock) as mock_dispatch:
        with (
            patch("handlers.factual.LocalSLMEngine.get_instance"),
            patch("handlers.sentiment.LocalSLMEngine.get_instance"),
            patch("handlers.ner.LocalSLMEngine.get_instance"),
            patch("handlers.summarization.LocalSLMEngine.get_instance"),
            patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock),
        ):
            router = AgentRouter(cache=cache)
            result = await router.route("task_1", "What is 2+2?")
            mock_dispatch.assert_not_called()
            assert result == "4"


# ---------------------------------------------------------------------------
# Layer 1b: AST Math (bypasses model entirely)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ast_math_bypass() -> None:
    cache = SemanticCache()
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance"),
        patch("handlers.sentiment.LocalSLMEngine.get_instance"),
        patch("handlers.ner.LocalSLMEngine.get_instance"),
        patch("handlers.summarization.LocalSLMEngine.get_instance"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
    ):
        router = AgentRouter(cache=cache)
        result = await router.route("math_task", "Calculate 342 * 12")
        assert result == "4104"
        mock_remote.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2/3: Classifier → local routes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_route_sentiment_local() -> None:
    cache = SemanticCache()
    fake_engine = MagicMock()
    fake_engine.generate.return_value = "Positive"
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.sentiment.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.ner.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.summarization.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("agent.router.classify", return_value="LOCAL_SENTIMENT"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock),
    ):
        router = AgentRouter(cache=cache)
        result = await router.route("s1", "Classify the sentiment: 'I love this!'")
        assert result == "Positive"


@pytest.mark.asyncio
async def test_route_ner_local() -> None:
    cache = SemanticCache()
    fake_engine = MagicMock()
    fake_engine.generate.return_value = '[{"entity": "Obama", "type": "Person"}]'
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.sentiment.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.ner.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("handlers.summarization.LocalSLMEngine.get_instance", return_value=fake_engine),
        patch("agent.router.classify", return_value="LOCAL_NER"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock),
    ):
        router = AgentRouter(cache=cache)
        result = await router.route("n1", "Extract named entities: 'Barack Obama visited France.'")
        assert "Obama" in result


# ---------------------------------------------------------------------------
# Layer 4: Remote routes (mocked)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_route_api_code() -> None:
    cache = SemanticCache()
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance"),
        patch("handlers.sentiment.LocalSLMEngine.get_instance"),
        patch("handlers.ner.LocalSLMEngine.get_instance"),
        patch("handlers.summarization.LocalSLMEngine.get_instance"),
        patch("agent.router.classify", return_value="API_CODE"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
    ):
        mock_remote.return_value = "def add(a, b): return a + b"
        router = AgentRouter(cache=cache)
        result = await router.route("c1", "Write a Python function to add two numbers")
        mock_remote.assert_called_once()
        assert "def add" in result


@pytest.mark.asyncio
async def test_route_api_logic() -> None:
    cache = SemanticCache()
    with (
        patch("handlers.factual.LocalSLMEngine.get_instance"),
        patch("handlers.sentiment.LocalSLMEngine.get_instance"),
        patch("handlers.ner.LocalSLMEngine.get_instance"),
        patch("handlers.summarization.LocalSLMEngine.get_instance"),
        patch("agent.router.classify", return_value="API_LOGIC"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
    ):
        mock_remote.return_value = "Yes"
        router = AgentRouter(cache=cache)
        result = await router.route("l1", "If John is taller than Mary, and Mary is taller than Sue, is John taller than Sue?")
        mock_remote.assert_called_once()
        assert result == "Yes"


# ---------------------------------------------------------------------------
# Local sentiment: unrecognised output → 'Neutral' locally, zero remote calls
# (PI-DEV-009: __ESCALATE__ path removed from SentimentHandler)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sentiment_unknown_output_defaults_neutral() -> None:
    """When the local SLM returns an unrecognised string, SentimentHandler must
    default to 'Neutral' without touching the remote API."""
    cache = SemanticCache()
    with (
        patch("engines.local_slm.LocalSLMEngine.get_instance") as mock_slm,
        patch("agent.router.classify", return_value="LOCAL_SENTIMENT"),
        patch("engines.remote_llm.RemoteLLMEngine.generate", new_callable=AsyncMock) as mock_remote,
    ):
        fake_engine = MagicMock()
        fake_engine.generate.return_value = "I'm not sure about the emotion here."
        mock_slm.return_value = fake_engine
        mock_remote.return_value = "should not be reached"

        router = AgentRouter(cache=cache)
        result = await router.route("esc1", "What is the sentiment of this review?")

        mock_remote.assert_not_called()
        assert result == "Neutral"
