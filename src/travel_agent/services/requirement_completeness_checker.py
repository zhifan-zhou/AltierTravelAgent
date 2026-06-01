"""Checks whether a TravelRequirementContract is complete enough to search."""

from __future__ import annotations

from travel_agent.models.requirement_contract import TravelRequirementContract


class RequirementCompletenessChecker:
    """Decide if contract is ready, or what to ask next."""

    def check(self, contract: TravelRequirementContract) -> dict:
        result = {
            "ready": False,
            "missing": [],
            "clarifications": [],
            "assumptions": [],
        }

        # Required
        if not contract.has_required_route():
            result["missing"].append("route")
            result["clarifications"].append("请问从哪里出发，去哪里？")
            return result

        # Optional but useful
        if not contract.trip.departure_window_text and contract.trip.departure_flexible:
            result["assumptions"].append("未提供日期，使用默认灵活日期窗口（约2周后出发）。")

        # Ready enough for demo
        if contract.trip.cabin == "any":
            contract.trip.cabin = "economy"
            result["assumptions"].append("未指定舱位，默认经济舱。")

        if len(result["missing"]) == 0:
            result["ready"] = True

        return result
