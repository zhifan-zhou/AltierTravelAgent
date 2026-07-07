from __future__ import annotations

from dataclasses import replace

import httpx

from travel_agent.config import load_settings
from travel_agent.llm.deepseek_client import DeepSeekRequirementAgent
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.tools.http_client import HttpClient
from travel_agent.tools.tool_router import ToolRouter


def v03_transport(*, rain: bool = False, fail: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("offline test transport", request=request)
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
                    "current": {"temperature_2m": 30, "weather_code": 61 if rain else 1},
                    "daily": {
                        "time": ["2026-07-06", "2026-07-07", "2026-07-08"],
                        "weather_code": [61 if rain else 1, 2, 1],
                        "temperature_2m_max": [36, 31, 30],
                        "temperature_2m_min": [24, 23, 22],
                        "precipitation_probability_max": [80 if rain else 10, 20, 10],
                    },
                    "current_units": {"temperature_2m": "°C"},
                    "daily_units": {"temperature_2m_max": "°C"},
                },
            )
        if path.endswith("/v1/latest"):
            return httpx.Response(
                200,
                json={"amount": 1, "base": "USD", "date": "2026-07-03", "rates": {"CNY": 6.78}},
            )
        if path.endswith("/w/api.php"):
            return httpx.Response(200, json={"query": {"search": [{"title": "奥斯汀"}]}})
        if "/api/rest_v1/page/summary/" in path:
            return httpx.Response(
                200,
                json={"extract": "奥斯汀是得克萨斯州首府，以现场音乐、公共空间和户外活动闻名。"},
            )
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def make_v03_router(*, rain: bool = False, fail: bool = False) -> ToolRouter:
    settings = replace(
        load_settings(),
        enabled_tools=("weather", "airport_lookup", "time", "currency", "destination_brief"),
    )
    return ToolRouter(
        settings=settings,
        http_client=HttpClient(
            transport=v03_transport(rain=rain, fail=fail),
            retries=0,
            backoff_seconds=0,
        ),
    )


def make_v03_session(*, rain: bool = False, fail: bool = False, debug: bool = False) -> LLMFirstChatSession:
    return LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(FakeRequirementLLM()),
        tool_router=make_v03_router(rain=rain, fail=fail),
        debug=debug,
    )
