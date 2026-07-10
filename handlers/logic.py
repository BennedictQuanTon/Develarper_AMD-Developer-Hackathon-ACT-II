# handlers/logic.py
import logging
import re

from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template

logger = logging.getLogger(__name__)


def extract_logic(text: str) -> str:
    """Cleans up the final logic answer based on strict format."""
    text = text.strip()
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip('.')
    if len(text.split()) <= 3:
        return text
        
    bold = re.findall(r'\*\*([^*]+?)\*\*', text)
    if bold:
        return bold[-1].strip()
        
    lines = text.strip().split('\n')
    last = lines[-1].strip()
    if re.match(r'^(yes|no)\.?$', last, re.IGNORECASE):
        return last.rstrip('.')
        
    if re.search(r'\byes\b', text, re.I) and not re.search(r'\bno\b', text, re.I):
        return "Yes"
    if re.search(r'\bno\b', text, re.I) and not re.search(r'\byes\b', text, re.I):
        return "No"
        
    ans = re.findall(r'(?:answer|conclusion|therefore)\s*(?:is|:)\s*["\']?(.+?)(?:["\']?\s*(?:\.|$|!))', text, re.I)
    if ans and len(ans[-1].split()) <= 6:
        return ans[-1].strip()
        
    return last if len(last.split()) <= 6 else text[:60]


class LogicHandler:
    """Routes logical reasoning / puzzle tasks to the remote LLM."""

    def __init__(self) -> None:
        self.engine = RemoteLLMEngine()
        self.system_prompt = load_prompt_template("remote_logic.txt")

    async def handle(self, prompt: str) -> str:
        raw_response = await self.engine.generate(
            prompt=prompt,
            category="API_LOGIC",
            system_prompt=self.system_prompt,
            max_tokens=200,  # Token budget: concise final answer
        )
        return extract_logic(raw_response)
