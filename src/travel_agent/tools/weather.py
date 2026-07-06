"""Weather tool using Open-Meteo geocoding and forecast APIs."""

from __future__ import annotations

from travel_agent.services.airport_service import AirportService
from travel_agent.services.geocoding_service import (
    OpenMeteoGeocodingClient,
    select_unambiguous_location,
)
from travel_agent.services.weather_service import OpenMeteoWeatherClient
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, clarification, unavailable
from travel_agent.tools.http_client import HttpClientError


class WeatherTool(BaseTool):
    name = "weather"
    description = "Live weather forecast from Open-Meteo."
    input_schema = {
        "type": "object",
        "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
    }

    def __init__(
        self,
        geocoding: OpenMeteoGeocodingClient,
        weather: OpenMeteoWeatherClient,
        airport_service: AirportService | None = None,
    ):
        self.geocoding = geocoding
        self.weather = weather
        self.airports = airport_service or AirportService()

    def execute(self, args: dict, context: ToolRequestContext) -> ToolResult:
        location, country = _location_and_country(args, context, self.airports)
        if not location:
            return clarification(self.name, "你想查哪个城市的天气？")
        try:
            locations = self.geocoding.search_location(location, country_code=country)
            selected = select_unambiguous_location(location, locations, country_code=country)
            if not selected:
                if not locations:
                    return unavailable(
                        self.name,
                        f"Open-Meteo 暂时找不到“{location}”对应的位置，无法提供天气。",
                        source="open_meteo_geocoding",
                        error_code="location_not_found",
                    )
                choices = "、".join(
                    f"{item.name} ({item.admin1 or item.country or item.country_code or '地区未知'})"
                    for item in locations[:4]
                )
                return ToolResult(
                    tool_name=self.name,
                    status="needs_clarification",
                    message=f"“{location}”有多个可能地点：{choices}。请补充国家或地区。",
                    source="open_meteo_geocoding",
                    error_code="ambiguous_location",
                    debug={"candidate_count": len(locations)},
                )
            forecast = self.weather.get_forecast(
                selected.latitude,
                selected.longitude,
                selected.timezone,
                days=int(args.get("days") or 7),
                location_name=_display_location(selected.name, selected.admin1, selected.country),
            )
            if not forecast.daily and not forecast.current:
                return unavailable(
                    self.name,
                    "Open-Meteo 返回的数据不完整，当前无法提供天气。",
                    source="open_meteo",
                    error_code="invalid_response",
                )
            lines = [f"{forecast.location_name or location}（Open-Meteo forecast）："]
            if forecast.current and forecast.current.temperature is not None:
                lines.append(
                    f"当前约 {forecast.current.temperature:g}°C，{forecast.current.summary or '天气代码未识别'}。"
                )
            for item in forecast.daily[:5]:
                temperature = _temperature_range(item.temperature_min, item.temperature_max)
                precipitation = (
                    f"，降水概率最高 {item.precipitation_probability_max:g}%"
                    if item.precipitation_probability_max is not None
                    else ""
                )
                lines.append(f"- {item.date}：{item.summary or '天气代码未识别'}，{temperature}{precipitation}")
            return ToolResult(
                tool_name=self.name,
                status="ok",
                data=forecast.model_dump(mode="json"),
                message="\n".join(lines),
                source=forecast.source,
                fetched_at=forecast.fetched_at,
                is_live=True,
                debug={"geocoding_source": selected.source, "timezone": selected.timezone},
            )
        except HttpClientError as exc:
            return unavailable(
                self.name,
                "Open-Meteo 当前不可用，无法提供实时天气；我不会用猜测数据代替。",
                source="open_meteo",
                error_code=exc.code,
                debug={"exception": exc.__class__.__name__},
            )
        except (TypeError, ValueError) as exc:
            return unavailable(
                self.name,
                "天气查询参数或返回数据无效，当前无法提供天气。",
                source="open_meteo",
                error_code="invalid_response",
                debug={"exception": exc.__class__.__name__},
            )


def _location_and_country(
    args: dict,
    context: ToolRequestContext,
    airports: AirportService,
) -> tuple[str, str | None]:
    location = str(args.get("location") or "").strip()
    code = ""
    contract = context.contract
    if contract:
        code = contract.trip.destination_airport or ""
    if not location and code:
        row = airports.get(code) or {}
        return str(row.get("city_cn") or row.get("city") or code), row.get("country")
    if location:
        codes = airports.resolve_location(location)
        row = airports.get(codes[0]) if codes else None
        return location, row.get("country") if row else None
    return "", None


def _display_location(name: str, admin1: str | None, country: str | None) -> str:
    parts = [name]
    for value in [admin1, country]:
        if value and value not in parts:
            parts.append(value)
    return ", ".join(parts)


def _temperature_range(low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return "温度数据缺失"
    if low is None:
        return f"最高 {high:g}°C"
    if high is None:
        return f"最低 {low:g}°C"
    return f"{low:g}–{high:g}°C"
