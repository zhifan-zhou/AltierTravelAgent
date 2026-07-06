"""Open-Meteo forecast adapter with typed parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from travel_agent.tools.http_client import HttpClient


class CurrentWeather(BaseModel):
    temperature: float | None = None
    weather_code: int | None = None
    summary: str | None = None
    wind_speed: float | None = None


class DailyWeather(BaseModel):
    date: str
    temperature_max: float | None = None
    temperature_min: float | None = None
    precipitation_probability_max: float | None = None
    weather_code: int | None = None
    summary: str | None = None


class WeatherForecast(BaseModel):
    location_name: str | None = None
    daily: list[DailyWeather] = Field(default_factory=list)
    current: CurrentWeather | None = None
    units: dict[str, Any] = Field(default_factory=dict)
    source: str = "open_meteo"
    fetched_at: datetime
    is_live: bool = True


WMO_SUMMARIES = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴冰雹",
}


class OpenMeteoWeatherClient:
    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, http: HttpClient | None = None):
        self.http = http or HttpClient()

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        timezone: str | None = None,
        days: int = 7,
        *,
        location_name: str | None = None,
    ) -> WeatherForecast:
        payload = self.http.get_json(
            self.endpoint,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone or "auto",
                "forecast_days": min(max(days, 1), 14),
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            },
            cache_ttl=20 * 60,
        )
        current_row = payload.get("current") or {}
        current = CurrentWeather(
            temperature=_number(current_row.get("temperature_2m")),
            weather_code=_integer(current_row.get("weather_code")),
            summary=WMO_SUMMARIES.get(_integer(current_row.get("weather_code"))),
            wind_speed=_number(current_row.get("wind_speed_10m")),
        ) if current_row else None
        daily_payload = payload.get("daily") or {}
        dates = daily_payload.get("time") or []
        daily: list[DailyWeather] = []
        for index, date in enumerate(dates):
            code = _at(daily_payload.get("weather_code"), index, integer=True)
            daily.append(
                DailyWeather(
                    date=str(date),
                    temperature_max=_at(daily_payload.get("temperature_2m_max"), index),
                    temperature_min=_at(daily_payload.get("temperature_2m_min"), index),
                    precipitation_probability_max=_at(
                        daily_payload.get("precipitation_probability_max"), index
                    ),
                    weather_code=code,
                    summary=WMO_SUMMARIES.get(code),
                )
            )
        return WeatherForecast(
            location_name=location_name,
            daily=daily,
            current=current,
            units={**(payload.get("current_units") or {}), **(payload.get("daily_units") or {})},
            fetched_at=datetime.now(UTC),
        )


def _number(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _at(values: object, index: int, *, integer: bool = False):
    if not isinstance(values, list) or index >= len(values):
        return None
    return _integer(values[index]) if integer else _number(values[index])
