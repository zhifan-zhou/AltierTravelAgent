"""Local airport lookup tool."""

from __future__ import annotations

from datetime import UTC, datetime

from travel_agent.services.airport_service import AirportService
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, clarification, unavailable


class AirportLookupTool(BaseTool):
    name = "airport_lookup"
    description = "Resolve airport/city names using bundled airport data."
    input_schema = {"type": "object", "properties": {"location": {"type": "string"}}}

    def __init__(self, airport_service: AirportService | None = None):
        self.airports = airport_service or AirportService()

    def execute(self, args: dict, context: ToolRequestContext) -> ToolResult:
        location = str(args.get("location") or "").strip()
        if not location and context.contract and context.contract.trip.destination_airport:
            location = context.contract.trip.destination_airport
        if not location:
            return clarification(self.name, "你想查哪个城市或机场？")
        codes = self.airports.resolve_location(location)
        if not codes:
            return unavailable(
                self.name,
                f"本地机场数据里暂时找不到“{location}”。",
                source="local_airport_data",
                error_code="airport_not_found",
            )
        rows = []
        lines = []
        for code in codes:
            row = self.airports.get(code) or {}
            rows.append(row)
            city = row.get("city_cn") or row.get("city") or code
            name = row.get("name") or code
            lines.append(f"{city}主要机场：{name} ({code})")
        return ToolResult(
            tool_name=self.name,
            status="ok",
            data={"location": location, "airports": rows},
            message="\n".join(lines),
            source="local_airport_data",
            fetched_at=datetime.now(UTC),
            is_live=False,
        )
