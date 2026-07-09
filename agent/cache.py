import hashlib
import re
import threading
from typing import Dict, Optional

class SemanticCache:
    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _normalize(self, text: str) -> str:
        # Lowercase, strip punctuation, strip whitespaces
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_hash(self, text: str) -> str:
        normalized = self._normalize(text)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        h = self._get_hash(prompt)
        with self._lock:
            return self._cache.get(h)

    def set(self, prompt: str, answer: str) -> None:
        h = self._get_hash(prompt)
        with self._lock:
            self._cache[h] = answer

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
