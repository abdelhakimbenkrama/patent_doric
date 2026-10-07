"""On-disk JSON cache so re-running an idea search spends no SerpApi credits or Groq tokens.

Entries never expire; delete the cache directory (config.CACHE_DIR) to clear it.
"""

import hashlib
import json
from typing import Any

from .config import CACHE_DIR


def _path(namespace: str, key: Any):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return CACHE_DIR / namespace / f"{digest}.json"


def get(namespace: str, key: Any) -> Any | None:
    path = _path(namespace, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(namespace: str, key: Any, value: Any) -> None:
    path = _path(namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # caching is best-effort; never fail a request over it
