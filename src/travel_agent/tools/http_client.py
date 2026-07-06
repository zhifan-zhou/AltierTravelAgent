"""Small robust HTTP client with retry, TTL cache, and rate limiting."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx


class HttpClientError(RuntimeError):
    """Sanitized external API error; request headers are never included."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._items: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._items.get(key)
            if not entry:
                return None
            if entry.expires_at <= self._clock():
                self._items.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: dict[str, Any], ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = _CacheEntry(value=value, expires_at=self._clock() + ttl_seconds)


class RateLimiter:
    """Process-local minimum interval limiter; deliberately lightweight."""

    def __init__(self, min_interval_seconds: float = 0.0):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.min_interval_seconds:
            return
        with self._lock:
            delay = self.min_interval_seconds - (time.monotonic() - self._last_call)
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()


class HttpClient:
    """Synchronous JSON client so current tool interfaces remain compatible."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 8.0,
        retries: int = 2,
        backoff_seconds: float = 0.15,
        cache: TTLCache | None = None,
        rate_limiter: RateLimiter | None = None,
        user_agent: str = "AltierTravelAgent/0.2 (+https://github.com/zhifan-zhou/AltierTravelAgent)",
    ):
        self.transport = transport
        self.timeout = min(max(float(timeout), 0.1), 10.0)
        self.retries = min(max(int(retries), 0), 2)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.cache = cache or TTLCache()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.user_agent = user_agent

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        cache_ttl: float = 0.0,
    ) -> dict[str, Any]:
        request_timeout = min(max(float(timeout or self.timeout), 0.1), 10.0)
        safe_headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        safe_headers.update(headers or {})
        request = httpx.Request("GET", url, params=params)
        cache_key = str(request.url)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        last_code = "network_error"
        for attempt in range(self.retries + 1):
            self.rate_limiter.wait()
            try:
                with httpx.Client(
                    transport=self.transport,
                    timeout=request_timeout,
                    follow_redirects=True,
                ) as client:
                    response = client.get(url, params=params, headers=safe_headers)
                if response.status_code == 429:
                    raise HttpClientError("rate_limited", "External API rate limited the request")
                if response.status_code >= 500:
                    raise HttpClientError("upstream_error", f"External API returned HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise HttpClientError("http_error", f"External API returned HTTP {response.status_code}")
                payload = response.json()
                if not isinstance(payload, dict):
                    raise HttpClientError("invalid_response", "External API returned a non-object JSON payload")
                self.cache.set(cache_key, payload, cache_ttl)
                return payload
            except HttpClientError as exc:
                last_code = exc.code
                if exc.code == "http_error" or attempt >= self.retries:
                    raise
            except (httpx.HTTPError, ValueError) as exc:
                last_code = "invalid_response" if isinstance(exc, ValueError) else "network_error"
                if attempt >= self.retries:
                    raise HttpClientError(last_code, f"External API request failed: {exc.__class__.__name__}") from exc
            if self.backoff_seconds:
                time.sleep(self.backoff_seconds * (2**attempt))
        raise HttpClientError(last_code, "External API request failed")
