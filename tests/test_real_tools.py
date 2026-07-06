from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import httpx

from travel_agent.config import load_settings
from travel_agent.llm.schemas import ToolRequest
from travel_agent.services.time_service import LocalTimeService
from travel_agent.tools.base import ToolRequestContext
from travel_agent.tools.http_client import HttpClient, HttpClientError
from travel_agent.tools.tool_router import ToolRouter


def _request(name: str, **arguments) -> ToolRequest:
    return ToolRequest(tool_name=name, arguments=arguments, reason_zh="test")


def _live_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Austin",
                            "country": "United States",
                            "country_code": "US",
                            "admin1": "Texas",
                            "latitude": 30.2672,
                            "longitude": -97.7431,
                            "timezone": "America/Chicago",
                        }
                    ]
                },
            )
        if path.endswith("/v1/forecast"):
            return httpx.Response(
                200,
                json={
                    "current": {"temperature_2m": 31.0, "weather_code": 1, "wind_speed_10m": 8.0},
                    "current_units": {"temperature_2m": "°C"},
                    "daily": {
                        "time": ["2026-07-06", "2026-07-07"],
                        "weather_code": [1, 61],
                        "temperature_2m_max": [36.0, 34.0],
                        "temperature_2m_min": [25.0, 24.0],
                        "precipitation_probability_max": [10, 55],
                    },
                    "daily_units": {"temperature_2m_max": "°C"},
                },
            )
        if path.endswith("/latest"):
            return httpx.Response(200, json={"amount": 100, "base": "USD", "date": "2026-07-03", "rates": {"CNY": 720.0}})
        if path.endswith("/w/api.php"):
            return httpx.Response(200, json={"query": {"search": [{"title": "奥斯汀"}]}})
        if "/api/rest_v1/page/summary/" in path:
            return httpx.Response(200, json={"extract": "奥斯汀是美国得克萨斯州首府，以现场音乐和户外活动闻名。"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_weather_tool_returns_live_typed_metadata_without_real_network():
    router = ToolRouter(http_client=HttpClient(transport=_live_transport(), retries=0))
    result = router.execute(_request("weather", location="Austin", days=2), ToolRequestContext())
    assert result.status == "ok"
    assert result.source == "open_meteo"
    assert result.is_live is True
    assert result.is_mock is False
    assert result.fetched_at is not None
    assert result.data["daily"][0]["temperature_max"] == 36.0
    assert "Open-Meteo forecast" in result.message


def test_weather_failure_is_unavailable_and_never_fabricates_data():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    router = ToolRouter(http_client=HttpClient(transport=httpx.MockTransport(fail), retries=0))
    result = router.execute(_request("weather", location="Austin"), ToolRequestContext())
    assert result.status == "unavailable"
    assert result.data is None
    assert result.is_live is False
    assert "不会用猜测数据" in result.message


def test_currency_chinese_alias_conversion_uses_frankfurter():
    router = ToolRouter(http_client=HttpClient(transport=_live_transport(), retries=0))
    result = router.execute(
        _request("currency", amount=100, from_currency="美元", to_currency="人民币"),
        ToolRequestContext(),
    )
    assert result.status == "ok"
    assert result.source == "frankfurter"
    assert result.data["from_currency"] == "USD"
    assert result.data["to_currency"] == "CNY"
    assert result.data["converted_amount"] == 720.0


def test_currency_invalid_code_asks_for_clarification_without_network():
    router = ToolRouter(http_client=HttpClient(transport=_live_transport(), retries=0))
    result = router.execute(
        _request("currency", amount=10, from_currency="???", to_currency="CNY"),
        ToolRequestContext(),
    )
    assert result.status == "needs_clarification"
    assert result.error_code == "invalid_currency"


def test_local_time_tool_uses_geocoding_timezone_and_zoneinfo():
    router = ToolRouter(http_client=HttpClient(transport=_live_transport(), retries=0))
    result = router.execute(_request("time", location="Austin"), ToolRequestContext())
    assert result.status == "ok"
    assert result.source == "python_zoneinfo"
    assert result.data["timezone"] == "America/Chicago"
    assert result.data["geocoding_source"] == "open_meteo_geocoding"


def test_local_time_service_is_deterministic_when_clock_is_injected():
    result = LocalTimeService().resolve(
        location_name="Austin",
        timezone="America/Chicago",
        now_utc=datetime(2026, 7, 6, 12, 0, tzinfo=UTC),
    )
    assert result.local_time.startswith("2026-07-06T07:00")
    assert result.utc_offset == "-05:00"


def test_geocoding_ambiguous_city_requires_clarification():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois", "latitude": 1, "longitude": 2, "timezone": "America/Chicago"},
                    {"name": "Springfield", "country_code": "US", "admin1": "Missouri", "latitude": 3, "longitude": 4, "timezone": "America/Chicago"},
                ]
            },
        )

    router = ToolRouter(http_client=HttpClient(transport=httpx.MockTransport(handler), retries=0))
    result = router.execute(_request("weather", location="Springfield"), ToolRequestContext())
    assert result.status == "needs_clarification"
    assert result.error_code == "ambiguous_location"


def test_http_client_retries_at_most_twice_and_sanitizes_error():
    calls = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timeout", request=request)

    client = HttpClient(transport=httpx.MockTransport(fail), retries=2, backoff_seconds=0)
    try:
        client.get_json("https://example.test/data", headers={"Authorization": "Bearer secret"})
    except HttpClientError as exc:
        assert exc.code == "network_error"
        assert "secret" not in str(exc)
    else:
        raise AssertionError("expected HttpClientError")
    assert calls == 3


def test_http_client_ttl_cache_avoids_duplicate_calls():
    calls = 0

    def ok(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"value": 1})

    client = HttpClient(transport=httpx.MockTransport(ok), retries=0)
    assert client.get_json("https://example.test/data", cache_ttl=60) == {"value": 1}
    assert client.get_json("https://example.test/data", cache_ttl=60) == {"value": 1}
    assert calls == 1


def test_optional_destination_brief_is_attributed():
    settings = replace(
        load_settings(),
        enabled_tools=("weather", "airport_lookup", "time", "currency", "destination_brief"),
    )
    router = ToolRouter(settings=settings, http_client=HttpClient(transport=_live_transport(), retries=0))
    result = router.execute(_request("destination_brief", location="奥斯丁"), ToolRequestContext())
    assert result.status == "ok"
    assert result.source == "wikivoyage"
    assert "Wikivoyage" in result.message
