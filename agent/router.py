# agent/router.py
"""
4-Layer AgentRouter
    Layer 1a: Semantic Cache (0 tokens)
    Layer 1b: AST Math Evaluator (0 tokens, deterministic)
    Layer 2:  Weighted Scoring Classifier (~0ms, no model)
    Layer 3:  Local SLM — Qwen2.5-1.5B via llama.cpp
    Layer 4:  Remote Fireworks API (category-aware model + prompt compression)
"""
import logging
from typing import Callable

from agent.ast_eval import evaluate_math_expression
from agent.cache import SemanticCache
from agent.classifier import (
    ROUTE_API_CODE,
    ROUTE_API_LOGIC,
    ROUTE_API_LONG,
    ROUTE_API_MATH,
    ROUTE_LOCAL_GENERAL,
    ROUTE_LOCAL_NER,
    ROUTE_LOCAL_SENTIMENT,
    classify,
)

# Handlers — local
from handlers.factual import FactualHandler
from handlers.summarization import SummarizationHandler
from handlers.sentiment import SentimentHandler
from handlers.ner import NERHandler

# Handlers — remote
from handlers.math_handler import MathHandler
from handlers.debug import DebugHandler
from handlers.code_gen import CodeGenHandler
from handlers.logic import LogicHandler

# Escalation fallback (remote, general-purpose)
from handlers.remote_handlers import RemoteGeneralHandler

logger = logging.getLogger(__name__)


class AgentRouter:
    """Orchestrates all 4 routing layers for a single prompt."""

    def __init__(self, cache: SemanticCache) -> None:
        self.cache = cache

        # Instantiate all handlers once (singleton SLM loaded once)
        self._factual = FactualHandler()
        self._summarization = SummarizationHandler()
        self._sentiment = SentimentHandler()
        self._ner = NERHandler()

        self._math = MathHandler()
        self._debug = DebugHandler()
        self._code_gen = CodeGenHandler()
        self._logic = LogicHandler()
        self._remote_general = RemoteGeneralHandler()

    async def route(self, task_id: str, prompt: str) -> str:
        # -------------------------------------------------------------------
        # Layer 1a: Cache
        # -------------------------------------------------------------------
        cached = self.cache.get(prompt)
        if cached is not None:
            logger.info("[%s] Cache hit", task_id)
            return cached

        # -------------------------------------------------------------------
        # Layer 1b: AST Math (deterministic, 0 tokens)
        # -------------------------------------------------------------------
        math_result = evaluate_math_expression(prompt)
        if math_result is not None:
            logger.info("[%s] AST math: %s", task_id, math_result)
            self.cache.set(prompt, math_result)
            return math_result

        # -------------------------------------------------------------------
        # Layer 2: Weighted Classifier
        # -------------------------------------------------------------------
        route = classify(prompt)
        logger.info("[%s] Route → %s", task_id, route)

        # -------------------------------------------------------------------
        # Layer 3 / 4: Dispatch
        # -------------------------------------------------------------------
        answer = await self._dispatch(task_id, prompt, route)
        self.cache.set(prompt, answer)
        return answer

    async def _dispatch(self, task_id: str, prompt: str, route: str) -> str:
        try:
            # ---- Local routes ----
            if route == ROUTE_LOCAL_SENTIMENT:
                res = self._sentiment.handle(prompt)
                if res == "__ESCALATE__":
                    logger.info("[%s] Sentiment escalated → remote", task_id)
                    res = await self._remote_general.handle(
                        prompt, category=ROUTE_LOCAL_SENTIMENT
                    )
                return res

            if route == ROUTE_LOCAL_NER:
                res = self._ner.handle(prompt)
                if res == "__ESCALATE__":
                    logger.info("[%s] NER escalated → remote", task_id)
                    res = await self._remote_general.handle(
                        prompt, category=ROUTE_LOCAL_NER
                    )
                return res

            if route == ROUTE_LOCAL_GENERAL:
                p = prompt.lower()
                if any(w in p for w in ["summarize", "summary", "tldr"]):
                    res = self._summarization.handle(prompt)
                else:
                    res = self._factual.handle(prompt)
                if res == "__ESCALATE__":
                    logger.info("[%s] Local general escalated → remote", task_id)
                    res = await self._remote_general.handle(
                        prompt, category=ROUTE_LOCAL_GENERAL
                    )
                return res

            # ---- Remote routes ----
            if route == ROUTE_API_MATH:
                return await self._math.handle(prompt)

            if route == ROUTE_API_CODE:
                p = prompt.lower()
                if any(w in p for w in ["debug", "fix", "bug", "error", "syntax"]):
                    return await self._debug.handle(prompt)
                return await self._code_gen.handle(prompt)

            if route == ROUTE_API_LOGIC:
                return await self._logic.handle(prompt)

            if route == ROUTE_API_LONG:
                return await self._remote_general.handle(
                    prompt, category=ROUTE_API_LONG
                )

            # Unexpected fallback
            logger.warning("[%s] Unknown route '%s' → remote general", task_id, route)
            return await self._remote_general.handle(prompt)

        except Exception as exc:
            logger.error("[%s] Dispatch error: %s", task_id, exc)
            return f"Error: processing failed ({type(exc).__name__})."
