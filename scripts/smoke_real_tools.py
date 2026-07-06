"""Optional manual smoke test for v0.2 public API adapters.

This script intentionally uses the network. It is not part of pytest/CI.
"""

from __future__ import annotations

from travel_agent.llm.schemas import ToolRequest
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
    for request in requests:
        result = router.execute(request, ToolRequestContext())
        print(
            f"[{result.status}] {result.tool_name}: source={result.source} "
            f"is_live={result.is_live} fetched_at={result.fetched_at or '(none)'} "
            f"error={result.error_code or '(none)'}"
        )
        print(result.message)
        print()
        if result.status != "ok":
            failures += 1
    if failures:
        print(f"smoke result: {failures} adapter(s) unavailable; no fallback data was fabricated")
        return 1
    print("smoke result: all public API adapters returned live results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
