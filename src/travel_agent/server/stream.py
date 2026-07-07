"""SSE helpers for streaming only user-visible response chunks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from travel_agent.planning.models import UserResponse
from travel_agent.rendering.response_streamer import ResponseStreamer


def sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def user_response_token_events(
    response: UserResponse,
    *,
    streamer: ResponseStreamer | None = None,
) -> Iterator[str]:
    active_streamer = streamer or ResponseStreamer()
    for chunk in active_streamer.stream_response(response):
        yield sse_event("token", {"text": chunk})
