"""Optional manual smoke test for v0.3 public API and planning layers.

This script intentionally uses the network. It is not part of pytest/CI.
"""

from __future__ import annotations

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.schemas import ToolRequest
from travel_agent.planning.constraint_checker import ConstraintChecker
from travel_agent.planning.itinerary_builder import ItineraryBuilder
from travel_agent.planning.response_planner import ResponsePlanner
from travel_agent.rendering.response_streamer import ResponseStreamer
from travel_agent.tools.base import ToolRequestContext
from travel_agent.tools.tool_router import ToolRouter


def main() -> int:
    router = ToolRouter()
    requests = [
        ToolRequest(
            tool_name="weather",
            arguments={"location": "Austin", "days": 2},
            reason_zh="manual smoke",
        ),
        ToolRequest(
            tool_name="currency",
            arguments={"amount": 100, "from_currency": "USD", "to_currency": "CNY"},
            reason_zh="manual smoke",
        ),
        ToolRequest(
            tool_name="time",
            arguments={"location": "Austin"},
            reason_zh="manual smoke",
        ),
    ]
    failures = 0
    results = {}
    for request in requests:
        result = router.execute(request, ToolRequestContext())
        results[result.tool_name] = result
        print(
            f"[{result.status}] {result.tool_name}: source={result.source} "
            f"is_live={result.is_live} fetched_at={result.fetched_at or '(none)'} "
            f"error={result.error_code or '(none)'}"
        )
        print(result.message)
        print()
        if result.status != "ok":
            failures += 1
    contract = TravelRequirementContract()
    contract.trip.destination_text = "奥斯丁"
    contract.trip.destination_airport = "AUS"
    contract.time.duration_days = 3
    contract.budget.preference = "lower"
    plan = ItineraryBuilder().build(contract, weather_result=results.get("weather"))
    checks = ConstraintChecker().check(contract, weather_result=results.get("weather"))
    response = ResponsePlanner().itinerary(plan, checks)
    rendered = "".join(ResponseStreamer(chunk_size=24).stream_response(response))
    planning_ok = len(plan.days) == 3 and rendered == response.text and "Day 3" in rendered
    print(
        f"[{'ok' if planning_ok else 'error'}] itinerary: days={len(plan.days)} "
        f"stream_matches={rendered == response.text}"
    )
    if not planning_ok:
        failures += 1
    if failures:
        print(f"smoke result: {failures} adapter(s) unavailable; no fallback data was fabricated")
        return 1
    print("smoke result: all public API adapters returned live results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
