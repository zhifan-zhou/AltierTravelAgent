"""Shared contracts for all non-flight tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from travel_agent.contract.models import TravelRequirementContract


ToolStatus = Literal["ok", "unavailable", "error", "needs_clarification"]


class ToolRequestContext(BaseModel):
    contract: TravelRequirementContract | None = None


class ToolResult(BaseModel):
    """Honest, serializable result envelope used by every tool."""

    tool_name: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    message: str
    source: str | None = None
    fetched_at: datetime | None = None
    is_live: bool = False
    is_mock: bool = False
    error_code: str | None = None
    debug: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Compatibility alias for the v0.1 tool contract."""
        return self.status == "ok"

    @property
    def user_facing_text_zh(self) -> str:
        return self.message

    @property
    def error_message(self) -> str:
        return self.error_code or ""


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any] = {}

    @abstractmethod
    def execute(self, args: dict[str, Any], context: ToolRequestContext) -> ToolResult:
        raise NotImplementedError


def clarification(tool_name: str, message: str, *, error_code: str = "missing_input") -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status="needs_clarification",
        message=message,
        source="tool_router",
        error_code=error_code,
    )


def unavailable(
    tool_name: str,
    message: str,
    *,
    source: str | None,
    error_code: str,
    debug: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status="unavailable",
        message=message,
        source=source,
        error_code=error_code,
        debug=debug,
    )
