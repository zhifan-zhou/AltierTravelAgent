"""Current local-time tool using geocoded IANA timezone data."""

from __future__ import annotations

from travel_agent.services.airport_service import AirportService
from travel_agent.services.geocoding_service import OpenMeteoGeocodingClient, select_unambiguous_location
from travel_agent.services.time_service import LocalTimeService
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, clarification, unavailable
from travel_agent.tools.http_client import HttpClientError
from travel_agent.tools.weather import _location_and_country


class LocalTimeTool(BaseTool):
    name = "time"
    description = "Current local time via Open-Meteo timezone data and Python zoneinfo."
    input_schema = {"type": "object", "properties": {"location": {"type": "string"}}}

    def __init__(
        self,
        geocoding: OpenMeteoGeocodingClient,
        service: LocalTimeService | None = None,
        airport_service: AirportService | None = None,
    ):
        self.geocoding = geocoding
        self.service = service or LocalTimeService()
        self.airports = airport_service or AirportService()

    def execute(self, args: dict, context: ToolRequestContext) -> ToolResult:
        location, country = _location_and_country(args, context, self.airports)
        if not location:
            return clarification(self.name, "你想查哪个城市的当前时间？")
        try:
            candidates = self.geocoding.search_location(location, country_code=country)
            selected = select_unambiguous_location(location, candidates, country_code=country)
            if not selected:
                if candidates:
                    return ToolResult(
                        tool_name=self.name,
                        status="needs_clarification",
                        message=f"“{location}”可能对应多个地点，请补充国家或地区后再查当地时间。",
                        source="open_meteo_geocoding",
                        error_code="ambiguous_location",
                    )
                return unavailable(
                    self.name,
                    f"Open-Meteo 暂时找不到“{location}”的时区，无法可靠计算当地时间。",
                    source="open_meteo_geocoding",
                    error_code="location_not_found",
                )
            if not selected.timezone:
                return unavailable(
                    self.name,
                    f"“{location}”没有可用时区信息，当前无法可靠计算当地时间。",
                    source="open_meteo_geocoding",
                    error_code="timezone_unavailable",
                )
            result = self.service.resolve(
                location_name=selected.name,
                timezone=selected.timezone,
                geocoding_source=selected.source,
            )
            return ToolResult(
                tool_name=self.name,
                status="ok",
                data=result.model_dump(mode="json"),
                message=(
                    f"{result.location_name} 当前当地时间是 {result.local_time} "
                    f"（{result.timezone}，UTC{result.utc_offset}）。"
                ),
                source=result.source,
                fetched_at=result.fetched_at,
                is_live=True,
                debug={"geocoding_source": result.geocoding_source},
            )
        except HttpClientError as exc:
            return unavailable(
                self.name,
                "地点时区查询当前不可用，无法可靠计算当地时间；我不会猜测时区。",
                source="open_meteo_geocoding",
                error_code=exc.code,
                debug={"exception": exc.__class__.__name__},
            )
        except ValueError as exc:
            return unavailable(
                self.name,
                "系统缺少该地点的有效时区数据，当前无法计算当地时间。",
                source="python_zoneinfo",
                error_code="timezone_unavailable",
                debug={"exception": exc.__class__.__name__},
            )
