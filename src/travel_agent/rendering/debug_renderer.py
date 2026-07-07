"""Debug output is separate from the ordinary response renderer."""

from __future__ import annotations


class DebugRenderer:
    def render(self, debug_text: str) -> str:
        return debug_text.strip()
