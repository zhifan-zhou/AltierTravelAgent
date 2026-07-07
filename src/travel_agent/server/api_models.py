"""Clean API contracts for the v0.4 web prototype."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VERSION = "v0.4"
SERVICE_NAME = "AltierTravelAgent"


FORBIDDEN_WEB_MARKERS = (
    "[debug]",
    "ToolRequest",
    "ToolResult",
    "Contract(",
    "TravelContract(",
    "schema update",
    "chain of thought",
    "internal",
    "traceback",
    "Traceback",
    "route_semantics",
    "LLM prompt",
    "raw JSON",
    "stack trace",
    "Exception(",
)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = VERSION
    service: str = SERVICE_NAME


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = True


class ChatResponse(BaseModel):
    session_id: str
    assistant_response: str
    response_type: str = "general_answer"
    contract_summary: dict[str, Any] = Field(default_factory=dict)
    cards: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str
    contract_summary: dict[str, Any] = Field(default_factory=dict)


class SessionListItem(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    contract_summary: dict[str, Any] = Field(default_factory=dict)


class SessionSnapshot(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    contract_summary: dict[str, Any] = Field(default_factory=dict)
    cards: list[dict[str, Any]] = Field(default_factory=list)


class DeleteSessionResponse(BaseModel):
    status: Literal["deleted"]


class ErrorResponse(BaseModel):
    message: str


def contains_forbidden_marker(value: Any) -> bool:
    """Return True if a user-facing API object contains an internal marker."""
    if isinstance(value, str):
        lowered = value.casefold()
        return any(marker.casefold() in lowered for marker in FORBIDDEN_WEB_MARKERS)
    if isinstance(value, dict):
        return any(contains_forbidden_marker(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(contains_forbidden_marker(item) for item in value)
    return False
