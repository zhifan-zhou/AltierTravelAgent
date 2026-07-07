"""Chunked fallback streaming for every final user response."""

from __future__ import annotations

from collections.abc import Iterator

from travel_agent.planning.models import UserResponse
from travel_agent.rendering.user_renderer import UserRenderer


class ResponseStreamer:
    def __init__(self, *, chunk_size: int = 32, renderer: UserRenderer | None = None):
        self.chunk_size = max(8, int(chunk_size))
        self.renderer = renderer or UserRenderer()

    def stream_text(self, text: str) -> Iterator[str]:
        for start in range(0, len(text), self.chunk_size):
            yield text[start : start + self.chunk_size]

    def stream_response(self, response: UserResponse) -> Iterator[str]:
        yield from self.stream_text(self.renderer.render(response))
