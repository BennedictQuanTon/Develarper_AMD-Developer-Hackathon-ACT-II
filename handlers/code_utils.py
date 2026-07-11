# handlers/code_utils.py
"""Shared helpers for code_gen and debug handlers.

Three responsibilities, all zero-token:
  1. classify_code_difficulty — easy/medium/hard routing for local-first vs API
  2. extract_code             — pull raw code out of a model response
  3. validate_code            — ast.parse + completeness checks (drives fallback)
"""

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def classify_code_difficulty(prompt: str) -> str:
    """Easy / medium / hard — decides local-first vs direct API.

    Heuristics only, no model calls, no tokens.
    """
    low = prompt.lower()
    length = len(prompt)

    hard_kw = (
        "dynamic programming",
        "dijkstra",
        "graph",
        "tree",
        "trie",
        "heap",
        "linked list",
        "lru",
        "backtracking",
        "topological",
        "n-queens",
        "knapsack",
        "memoization",
        "bfs",
        "dfs",
        "binary search tree",
        "cycle detection",
        "floyd",
        "strongly connected",
        "minimax",
    )
    if any(kw in low for kw in hard_kw):
        return HARD
    if length > 400:
        return HARD

    fn_count = len(re.findall(r"\bdef\s+\w+\s*\(", prompt))
    if fn_count >= 2 or "class " in low:
        return HARD

    easy_kw = ("simple", "basic", "straightforward")
    if any(kw in low for kw in easy_kw):
        return EASY
    if length < 120:
        return EASY

    return MEDIUM


def extract_code(text: str) -> str:
    """Pull code out of a model response (strip fences, skip prose)."""
    text = text.strip()
    matches = _FENCE_RE.findall(text)
    if matches:
        return str(max(matches, key=len)).strip()

    # No fences — strip leading prose until we hit a code-like line.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s and (s.startswith(("def ", "class ", "import ", "from ", "if __", "#")) or re.match(r"^[a-zA-Z_]\w*\s*=", s)):
            return "\n".join(lines[i:]).strip()

    return text


def validate_code(code: str) -> dict[str, Any]:
    """Run syntax + completeness checks. Returns {is_valid, reason}."""
    if not code or not code.strip():
        return {"is_valid": False, "reason": "empty"}

    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"is_valid": False, "reason": f"SyntaxError line {e.lineno}: {e.msg}"}

    meaningful = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(meaningful) < 2:
        return {"is_valid": False, "reason": "too short"}

    opens = code.count("(") + code.count("[") + code.count("{")
    closes = code.count(")") + code.count("]") + code.count("}")
    if abs(opens - closes) > 1:
        return {"is_valid": False, "reason": "unbalanced brackets"}

    last = meaningful[-1].strip()
    if last.endswith("...") or last.endswith("# TODO"):
        return {"is_valid": False, "reason": "truncated"}

    return {"is_valid": True, "reason": ""}
