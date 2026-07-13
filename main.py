"""
main.py — Entrypoint
Reads /input/tasks.json, routes each task through AgentRouter, writes /output/results.json.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import sys

from dotenv import load_dotenv

from agent.cache import SemanticCache
from agent.router import AgentRouter
from agent.schemas import Task
from agent.watchdog import Watchdog

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Global state — lightweight; expensive models are deferred into main()
_cache: SemanticCache | None = None
_router: AgentRouter | None = None
_completed: list[dict[str, str]] = []
_lock: asyncio.Lock | None = None


def _get_results() -> list[dict[str, str]]:
    return list(_completed)


async def _process(task: Task) -> None:
    assert _router is not None and _lock is not None
    try:
        answer = await asyncio.wait_for(_router.route(task.task_id, task.prompt), timeout=25.0)
    except TimeoutError:
        logger.error("Task %s timed out after 25s", task.task_id)
        answer = "Error: Execution timed out."
    except Exception as exc:
        logger.error("Task %s failed: %s", task.task_id, exc)
        answer = f"Error: {exc}"

    async with _lock:
        _completed.append({"task_id": task.task_id, "answer": answer})
    logger.info("Done: %s", task.task_id)


async def main() -> None:
    global _cache, _router, _lock

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # -----------------------------------------------------------------------
    # Initialization block — runs INSIDE asyncio.run(), after loop is live.
    # Guards allow tests to pre-inject mocks before calling main().
    # -----------------------------------------------------------------------
    if _router is None:
        logger.info("Initializing agent components (GGUF + classifier)...")
        _cache = SemanticCache()
        _router = AgentRouter(cache=_cache, executor=executor)
        logger.info("Initialization complete — starting task processing.")
    if _lock is None:
        _lock = asyncio.Lock()

    if not os.path.exists(INPUT_PATH):
        logger.error("Input file not found: %s", INPUT_PATH)
        sys.exit(1)

    try:
        with open(INPUT_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.error("Failed to parse input: %s", exc)
        sys.exit(1)

    tasks: list[Task] = []
    for item in raw:
        try:
            tasks.append(Task.model_validate(item))
        except Exception as exc:
            logger.warning("Skipping invalid task %s: %s", item, exc)

    if not tasks:
        logger.error("No valid tasks found.")
        sys.exit(1)

    watchdog = Watchdog(
        timeout_seconds=570.0,
        tasks=[t.model_dump() for t in tasks],
        results_path=OUTPUT_PATH,
        get_current_results=_get_results,
    )
    watchdog.start()

    try:
        await asyncio.gather(*[_process(t) for t in tasks])

        final = _get_results()
        temp_path = OUTPUT_PATH + ".tmp"
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, OUTPUT_PATH)
        logger.info("All %d tasks done → %s", len(final), OUTPUT_PATH)
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
    finally:
        watchdog.stop()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
