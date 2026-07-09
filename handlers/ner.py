# handlers/ner.py
import json
import logging
from engines.local_slm import LocalSLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


class NERHandler:
    """Extracts named entities locally as a JSON list. Falls back to __ESCALATE__."""

    def __init__(self) -> None:
        self.engine = LocalSLMEngine.get_instance()
        self.system_prompt = load_prompt_template("ner.txt")

    def handle(self, prompt: str) -> str:
        res = self.engine.generate(prompt, self.system_prompt, max_tokens=300)
        if res == "__ESCALATE__":
            return "__ESCALATE__"
        try:
            json_str = res
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
            if not isinstance(parsed, list):
                logger.warning("NERHandler: response is not a list. Escalating.")
                return "__ESCALATE__"
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as exc:
            logger.warning("NERHandler: JSON parse failed — %s. Escalating.", exc)
            return "__ESCALATE__"
