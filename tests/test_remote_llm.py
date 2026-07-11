from unittest.mock import AsyncMock, patch

import pytest

from engines.remote_llm import RemoteLLMEngine, compress_prompt, select_remote_model


# ---------------------------------------------------------------------------
# compress_prompt
# ---------------------------------------------------------------------------
def test_compress_strips_filler() -> None:
    raw = "Can you please explain the photosynthesis process?"
    result = compress_prompt(raw, "LOCAL_GENERAL")
    assert "can you" not in result.lower()
    assert "photosynthesis" in result.lower()


def test_compress_appends_suffix_math() -> None:
    result = compress_prompt("What is 2 + 2?", "API_MATH")
    assert result == "What is 2 + 2?"


def test_compress_appends_suffix_code() -> None:
    result = compress_prompt("Write a function", "API_CODE")
    assert "Return ONLY raw code" in result


def test_compress_no_suffix_for_general() -> None:
    result = compress_prompt("What is the capital of France?", "LOCAL_GENERAL")
    # No suffix for LOCAL_GENERAL
    assert "ONLY" not in result


# ---------------------------------------------------------------------------
# select_remote_model — priority list
# ---------------------------------------------------------------------------
def test_model_selection_code_prefers_kimi() -> None:
    with patch("engines.remote_llm.ALLOWED_MODELS", ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it"]):
        assert select_remote_model("API_CODE") == "kimi-k2p7-code"


def test_model_selection_math_prefers_kimi() -> None:
    with patch("engines.remote_llm.ALLOWED_MODELS", ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it"]):
        assert select_remote_model("API_MATH") == "kimi-k2p7-code"


def test_model_selection_logic_prefers_kimi_then_minimax() -> None:
    with patch("engines.remote_llm.ALLOWED_MODELS", ["gemma-4-31b-it"]):
        # Preferred logic models are kimi and minimax; neither is allowed.
        # Fallback logic should pick the first available gemma model.
        assert select_remote_model("API_LOGIC") == "gemma-4-31b-it"

    with patch("engines.remote_llm.ALLOWED_MODELS", ["minimax-m3", "kimi-k2p7-code"]):
        assert select_remote_model("API_LOGIC") == "kimi-k2p7-code"


def test_model_selection_fallback_empty() -> None:
    with patch("engines.remote_llm.ALLOWED_MODELS", []):
        assert select_remote_model("API_MATH") == "gemma-4-31b-it"


def test_model_selection_long_context() -> None:
    with patch("engines.remote_llm.ALLOWED_MODELS", ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]):
        assert select_remote_model("API_LONG_CONTEXT") == "gemma-4-26b-a4b-it"


# ---------------------------------------------------------------------------
# RemoteLLMEngine.generate (mocked)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_remote_llm_generate_mocked() -> None:
    engine = RemoteLLMEngine()
    mock_response = {"choices": [{"message": {"content": "42"}}]}
    with patch("engines.remote_llm.ALLOWED_MODELS", ["gemma-4-31b-it"]):
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp

            result = await engine.generate("What is 6 * 7?", "API_MATH")
            assert result == "42"


@pytest.mark.asyncio
async def test_remote_llm_generate_completions_mocked() -> None:
    engine = RemoteLLMEngine()
    mock_response = {"choices": [{"message": {"content": "99"}}]}
    with patch("engines.remote_llm.select_remote_model", return_value="minimax-m3"):
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value = mock_resp

            result = await engine.generate("Solve logic riddle", "API_LOGIC")
            assert result == "99"
