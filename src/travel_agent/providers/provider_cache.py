"""In-memory cache for flight provider search results."""

from __future__ import annotations

from datetime import datetime, timedelta


class ProviderCache:
    """Simple in-memory cache with TTL. Keyed on search parameters."""

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[datetime, object]] = {}

    @staticmethod
    def _make_key(
        provider_name: str, origin: str, destination: str,
        departure_date: str, cabin: str, passengers: int,
    ) -> str:
        return f"{provider_name}|{origin}|{destination}|{departure_date}|{cabin}|{passengers}"

    def get(
        self, provider_name: str, origin: str, destination: str,
        departure_date: str, cabin: str, passengers: int,
    ) -> object | None:
        key = self._make_key(provider_name, origin, destination, departure_date, cabin, passengers)
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if datetime.now() - ts > timedelta(seconds=self._ttl):
            del self._store[key]
            return None
        return value

    def set(
        self, provider_name: str, origin: str, destination: str,
        departure_date: str, cabin: str, passengers: int,
        value: object,
    ) -> None:
        key = self._make_key(provider_name, origin, destination, departure_date, cabin, passengers)
        self._store[key] = (datetime.now(), value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)
