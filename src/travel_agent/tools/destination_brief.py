"""Optional short destination brief from Wikimedia/Wikivoyage."""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote

from travel_agent.services.airport_service import AirportService
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, clarification, unavailable
from travel_agent.tools.http_client import HttpClient, HttpClientError
from travel_agent.tools.weather import _location_and_country


class DestinationBriefTool(BaseTool):
    name = "destination_brief"
    description = "Short destination introduction from Wikivoyage."
    input_schema = {"type": "object", "properties": {"location": {"type": "string"}}}

    def __init__(self, http: HttpClient, airport_service: AirportService | None = None):
        self.http = http
        self.airports = airport_service or AirportService()

    def execute(self, args: dict, context: ToolRequestContext) -> ToolResult:
        location, _ = _location_and_country(args, context, self.airports)
        if not location:
            return clarification(self.name, "你想了解哪个目的地？")
        try:
            search = self.http.get_json(
                "https://zh.wikivoyage.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": location,
                    "srlimit": 1,
                    "format": "json",
                    "origin": "*",
                },
                cache_ttl=24 * 60 * 60,
            )
            matches = ((search.get("query") or {}).get("search") or [])
            if not matches:
                return unavailable(
                    self.name,
                    f"Wikivoyage 暂时没有找到“{location}”的目的地简介。",
                    source="wikivoyage",
                    error_code="page_not_found",
                )
            title = str(matches[0].get("title") or "").strip()
            summary = self.http.get_json(
                f"https://zh.wikivoyage.org/api/rest_v1/page/summary/{quote(title, safe='')}",
                cache_ttl=24 * 60 * 60,
            )
            extract = str(summary.get("extract") or "").strip()
            if not extract:
                return unavailable(
                    self.name,
                    f"Wikivoyage 的“{title}”页面暂时没有可用摘要。",
                    source="wikivoyage",
                    error_code="empty_summary",
                )
            brief = extract[:700].rstrip()
            if len(extract) > len(brief):
                brief += "…"
            fetched_at = datetime.now(UTC)
            return ToolResult(
                tool_name=self.name,
                status="ok",
                data={"title": title, "extract": brief, "url": summary.get("content_urls")},
                message=f"{title}｜Wikivoyage 简介\n{brief}",
                source="wikivoyage",
                fetched_at=fetched_at,
                is_live=False,
            )
        except HttpClientError as exc:
            return unavailable(
                self.name,
                "Wikivoyage 当前不可用，暂时无法提供可靠目的地简介。",
                source="wikivoyage",
                error_code=exc.code,
                debug={"exception": exc.__class__.__name__},
            )
