import logging
import os
import re
from collections.abc import Iterator
from typing import Any, SupportsIndex, overload

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class DynamicAllowedModels(list[str]):
    def _get_list(self) -> list[str]:
        return [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._get_list())

    def __len__(self) -> int:
        return len(self._get_list())

    def __bool__(self) -> bool:
        return bool(self._get_list())

    @overload
    def __getitem__(self, index: SupportsIndex) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> str | list[str]:
        return self._get_list()[index]

    def __contains__(self, item: object) -> bool:
        return item in self._get_list()


ALLOWED_MODELS = DynamicAllowedModels()

# Priority preferences per category — first matching model in ALLOWED_MODELS wins
CATEGORY_MODEL_PREFERENCE: dict[str, list[str]] = {
    "API_CODE": ["kimi-k2p7-code", "gemma-4-31b-it"],
    "API_MATH": ["kimi-k2p7-code", "minimax-m3"],
    "API_LOGIC": ["kimi-k2p7-code", "minimax-m3"],
    "API_LONG_CONTEXT": ["gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4"],
}

# ---------------------------------------------------------------------------
# Prompt compression — strip filler phrases + append output suffix
# ---------------------------------------------------------------------------
_FILLER_RE = re.compile(
    r"(please\s+(tell me|explain|provide|help me|describe)\s*"
    r"|can you\s+(please\s+)?"
    r"|i (would|'d) like (you to\s+)?"
    r"|i want (you to\s+)?"
    r"|could you\s+|i need (you to\s+)?)",
    re.IGNORECASE,
)

_OUTPUT_SUFFIX: dict[str, str] = {
    "API_CODE": " Return ONLY raw code. No markdown, no explanation.",
    "API_MATH": "",  # Handled natively in CoT system prompt
    "API_LOGIC": "", # Handled natively in Direct system prompt
    "API_LONG_CONTEXT": " Summarize in 3 sentences max.",
}


def compress_prompt(prompt: str, category: str) -> str:
    """Strip filler phrases and append a concise output instruction."""
    compressed = _FILLER_RE.sub("", prompt).strip()
    suffix = _OUTPUT_SUFFIX.get(category, "")
    return compressed + suffix


def select_remote_model(category: str) -> str:
    """Select the best available model for the given category."""
    preferred = CATEGORY_MODEL_PREFERENCE.get(category, [])
    for model in preferred:
        for allowed in ALLOWED_MODELS:
            if allowed == model or allowed.endswith(f"/{model}"):
                return allowed
    # Fallback: any gemma in allowed list, then minimax, then first available
    for model in ALLOWED_MODELS:
        if "gemma" in model.lower():
            return model
    for model in ALLOWED_MODELS:
        if "minimax" in model.lower():
            return model
    if ALLOWED_MODELS:
        return ALLOWED_MODELS[0]
    # Hard default when nothing is configured
    return "gemma-4-31b-it"


# ---------------------------------------------------------------------------
# Remote LLM Engine
# ---------------------------------------------------------------------------
class RemoteLLMEngine:
    def __init__(self) -> None:
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        raw_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

        # Normalise: ensure the URL ends at /v1
        if raw_url.endswith("/v1"):
            base = raw_url
        elif "/v1" in raw_url:
            base = raw_url.rstrip("/")
        else:
            base = raw_url.rstrip("/") + "/v1"

        self.base_url = base

        self.model_prefix = "accounts/fireworks/models/"

        if not self.api_key:
            logger.warning("FIREWORKS_API_KEY is not set. Remote API calls will fail with 401. Set the key in your .env file before running Phase 5.")
        logger.info("Remote LLM Engine base URL → %s", self.base_url)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientResponseError, aiohttp.ClientConnectorError)),
    )
    async def generate(
        self,
        prompt: str,
        category: str,
        system_prompt: str = "",
        max_tokens: int = 150,
        temperature: float = 0.2,
    ) -> str:
        # Compress prompt before sending
        compressed = compress_prompt(prompt, category)
        model_name = select_remote_model(category)
        if model_name.startswith(self.model_prefix):
            model = model_name
        else:
            model = f"{self.model_prefix}{model_name}"
        logger.info("Remote [%s] → model=%s max_tokens=%d", category, model, max_tokens)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Dynamically switch between Chat Completions and raw Completions
        is_chat = any(x in model_name.lower() for x in ["-it", "kimi", "minimax"])
        endpoint = f"{self.base_url}/chat/completions" if is_chat else f"{self.base_url}/completions"

        if is_chat:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": compressed})
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        else:
            if system_prompt:
                prompt_text = f"{system_prompt}\n\nUser: {compressed}\nAssistant:"
            else:
                prompt_text = f"User: {compressed}\nAssistant:"
            payload = {
                "model": model,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error("Fireworks API error (status %d): %s", response.status, text)
                    response.raise_for_status()

                data = await response.json()
                if is_chat:
                    return str(data["choices"][0]["message"]["content"]).strip()
                else:
                    return str(data["choices"][0]["text"]).strip()
