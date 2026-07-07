"""Ordinary renderer with a strict UserResponse-only boundary."""

from __future__ import annotations

from travel_agent.planning.models import UserResponse


FORBIDDEN_USER_MARKERS = (
    "[debug]",
    "toolrequest",
    "toolresult",
    "contract(",
    "schema update",
    "chain of thought",
    "internal",
    "planning notes",
    "contract diff",
    "tool request",
    "tool result",
    "traceback",
    "route_semantics",
    "llm prompt",
    "raw json",
)


class UserRenderer:
    def render(self, response: UserResponse) -> str:
        if not isinstance(response, UserResponse):
            raise TypeError("UserRenderer accepts UserResponse only")
        text = response.text
        lowered = text.casefold()
        if any(marker in lowered for marker in FORBIDDEN_USER_MARKERS):
            return "这次响应包含不应展示的诊断内容，已安全隐藏。请重试。"
        return text
