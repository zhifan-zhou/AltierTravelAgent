"""Simple in-memory cache for flight search results (MVP)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


class CacheService:
    """Trivial in-memory cache with TTL. Replace with Redis for production."""

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[datetime, object]] = {}

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if datetime.now() - ts > timedelta(seconds=self._ttl):
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (datetime.now(), value)

    def clear(self) -> None:
        self._store.clear()
