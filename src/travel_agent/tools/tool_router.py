"""MCP-ready local tool router for non-flight actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from travel_agent.config import Settings, load_settings
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.schemas import ToolRequest
from travel_agent.services.airport_service import AirportService


class ToolRequestContext(BaseModel):
    contract: TravelRequirementContract | None = None


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    user_facing_text_zh: str
    data: dict[str, Any] = Field(default_factory=dict)
    source: str = "local"
    error_message: str = ""


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any] = {}

    @abstractmethod
    def execute(self, args: dict[str, Any], context: ToolRequestContext) -> ToolResult:
        raise NotImplementedError


class WeatherTool(BaseTool):
    name = "weather"
    description = "Weather lookup placeholder; ready for MCP/weather integration."
    input_schema = {"type": "object", "properties": {"location": {"type": "string"}}}

    def execute(self, args: dict[str, Any], context: ToolRequestContext) -> ToolResult:
        location = _location_from_args_or_contract(args, context)
        if not location:
            return ToolResult(
                tool_name=self.name,
                success=False,
                user_facing_text_zh="你想查哪个城市的天气？",
                source="local_stub",
                error_message="missing location",
            )
        return ToolResult(
            tool_name=self.name,
            success=False,
            user_facing_text_zh=(
                f"你问的是目的地天气。当前查询城市是 {location}。"
                "实时天气工具还未接入，所以我不能给出真实天气；后续可以通过 MCP/weather 接入。"
            ),
            data={"location": location},
            source="local_stub",
            error_message="weather tool not connected",
        )


class AirportLookupTool(BaseTool):
    name = "airport_lookup"
    description = "Resolve airport/city names using local airport data."
    input_schema = {"type": "object", "properties": {"location": {"type": "string"}}}

    def __init__(self, airport_service: AirportService | None = None):
        self.airports = airport_service or AirportService()

    def execute(self, args: dict[str, Any], context: ToolRequestContext) -> ToolResult:
        location = str(args.get("location") or "").strip()
        if not location and context.contract and context.contract.trip.destination_airport:
            location = context.contract.trip.destination_airport
        if not location:
            return ToolResult(
                tool_name=self.name,
                success=False,
                user_facing_text_zh="你想查哪个城市或机场？",
                source="local_airport_data",
                error_message="missing location",
            )
        codes = self.airports.resolve_location(location)
        if not codes:
            return ToolResult(
                tool_name=self.name,
                success=False,
                user_facing_text_zh=f"我暂时没有在机场数据里找到“{location}”。",
                source="local_airport_data",
                error_message="airport not found",
            )
        lines = []
        for code in codes:
            row = self.airports.get(code) or {}
            city = row.get("city_cn") or row.get("city") or code
            name = row.get("name") or code
            lines.append(f"{city}主要机场：{name} ({code})")
        return ToolResult(
            tool_name=self.name,
            success=True,
            user_facing_text_zh="\n".join(lines),
            data={"location": location, "airports": codes},
            source="local_airport_data",
        )


class StubTool(BaseTool):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.input_schema = {}

    def execute(self, args: dict[str, Any], context: ToolRequestContext) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=False,
            user_facing_text_zh=f"当前 CLI 尚未接入 {self.name} 实时工具。后续可以通过 MCP/{self.name} 接入。",
            data={"arguments": args},
            source="local_stub",
            error_message="tool not connected",
        )


class ToolRouter:
    def __init__(self, settings: Settings | None = None, airport_service: AirportService | None = None):
        self.settings = settings or load_settings()
        self.registry: dict[str, BaseTool] = {
            "weather": WeatherTool(),
            "airport_lookup": AirportLookupTool(airport_service),
            "time": StubTool("time", "Time lookup placeholder."),
            "currency": StubTool("currency", "Currency conversion placeholder."),
        }

    def execute(self, request: ToolRequest, context: ToolRequestContext) -> ToolResult:
        if not self.settings.enable_tools:
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                user_facing_text_zh="当前 CLI 工具层未启用。",
                source="tool_router",
                error_message="tools disabled",
            )
        if request.tool_name not in self.settings.enabled_tools:
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                user_facing_text_zh=f"当前 CLI 没有启用 {request.tool_name} 工具，但我可以先记录这个需求。",
                source="tool_router",
                error_message="tool not enabled",
            )
        tool = self.registry.get(request.tool_name)
        if not tool:
            return ToolResult(
                tool_name=request.tool_name,
                success=False,
                user_facing_text_zh=f"当前 CLI 没有接入 {request.tool_name} 工具，但我可以先记录这个需求。",
                source="tool_router",
                error_message="tool not registered",
            )
        return tool.execute(dict(request.arguments or {}), context)


def _location_from_args_or_contract(args: dict[str, Any], context: ToolRequestContext) -> str:
    location = str(args.get("location") or "").strip()
    if location:
        return location
    contract = context.contract
    if not contract or not contract.trip.destination_airport:
        return ""
    row = AirportService().get(contract.trip.destination_airport) or {}
    return row.get("city_cn") or row.get("city") or contract.trip.destination_airport
