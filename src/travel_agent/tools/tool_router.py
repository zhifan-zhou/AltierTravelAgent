"""Unified router for real/free non-flight tool adapters."""

from __future__ import annotations

from travel_agent.config import Settings, load_settings
from travel_agent.llm.schemas import ToolRequest
from travel_agent.services.airport_service import AirportService
from travel_agent.services.currency_service import FrankfurterCurrencyClient
from travel_agent.services.geocoding_service import OpenMeteoGeocodingClient
from travel_agent.services.time_service import LocalTimeService
from travel_agent.services.weather_service import OpenMeteoWeatherClient
from travel_agent.tools.airport_lookup import AirportLookupTool
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, unavailable
from travel_agent.tools.currency import CurrencyTool
from travel_agent.tools.destination_brief import DestinationBriefTool
from travel_agent.tools.http_client import HttpClient, RateLimiter
from travel_agent.tools.time_tool import LocalTimeTool
from travel_agent.tools.weather import WeatherTool


class ToolRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        airport_service: AirportService | None = None,
        *,
        http_client: HttpClient | None = None,
        registry: dict[str, BaseTool] | None = None,
    ):
        self.settings = settings or load_settings()
        airports = airport_service or AirportService()
        http = http_client or HttpClient(rate_limiter=RateLimiter(0.05))
        geocoding = OpenMeteoGeocodingClient(http)
        if registry is not None:
            self.registry = dict(registry)
        else:
            self.registry: dict[str, BaseTool] = {
                "weather": WeatherTool(geocoding, OpenMeteoWeatherClient(http), airports),
                "airport_lookup": AirportLookupTool(airports),
                "time": LocalTimeTool(geocoding, LocalTimeService(), airports),
                "currency": CurrencyTool(FrankfurterCurrencyClient(http)),
                "destination_brief": DestinationBriefTool(http, airports),
            }

    def execute(self, request: ToolRequest, context: ToolRequestContext) -> ToolResult:
        if not self.settings.enable_tools:
            return unavailable(
                request.tool_name,
                "当前 CLI 工具层未启用。",
                source="tool_router",
                error_code="tools_disabled",
            )
        if request.tool_name not in self.settings.enabled_tools:
            return unavailable(
                request.tool_name,
                f"当前 CLI 没有启用 {request.tool_name} 工具。",
                source="tool_router",
                error_code="tool_disabled",
            )
        tool = self.registry.get(request.tool_name)
        if not tool:
            return unavailable(
                request.tool_name,
                f"当前 CLI 没有注册 {request.tool_name} 工具。",
                source="tool_router",
                error_code="tool_not_registered",
            )
        try:
            return tool.execute(dict(request.arguments or {}), context)
        except Exception as exc:  # final safety boundary for third-party adapters
            return ToolResult(
                tool_name=request.tool_name,
                status="error",
                message=f"{request.tool_name} 工具处理失败，当前没有可用结果；不会返回猜测数据。",
                source="tool_router",
                error_code="internal_tool_error",
                debug={"exception": exc.__class__.__name__},
            )


__all__ = ["ToolRequestContext", "ToolResult", "ToolRouter"]
