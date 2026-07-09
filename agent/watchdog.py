import json
import logging
import os
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(self, timeout_seconds: float, tasks: list[dict], results_path: str, get_current_results: Callable[[], list[dict]]) -> None:
        self.timeout_seconds = timeout_seconds
        self.tasks = tasks
        self.results_path = results_path
        self.get_current_results = get_current_results
        self.timer: threading.Timer | None = None

    def start(self) -> None:
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        logger.info(f"Watchdog started with {self.timeout_seconds} seconds timeout.")

    def stop(self) -> None:
        if self.timer:
            self.timer.cancel()
            logger.info("Watchdog stopped.")

    def _timeout_handler(self) -> None:
        logger.error(f"Watchdog timeout reached ({self.timeout_seconds}s). Force saving partial results.")
        try:
            completed_results = self.get_current_results()
            completed_ids = {r["task_id"] for r in completed_results}

            final_results = list(completed_results)
            for task in self.tasks:
                tid = task.get("task_id")
                if tid and tid not in completed_ids:
                    final_results.append({"task_id": tid, "answer": "Error: Execution timed out. Default fallback response."})

            # Atomic write
            temp_path = self.results_path + ".tmp"
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.results_path)
            logger.error("Successfully wrote partial results to output path. Exiting container.")
        except Exception as e:
            logger.critical(f"Watchdog failed to write output: {e}")
        finally:
            sys.exit(0)
