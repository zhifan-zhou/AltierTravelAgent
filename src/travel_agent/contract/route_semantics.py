"""Validate route roles from user text after LLM schema extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.schemas import DecisionTraceItem, TravelRequirementContractUpdate
from travel_agent.services.airport_service import AirportService


@dataclass
class RouteSemanticValidationResult:
    update: TravelRequirementContractUpdate
    repaired: bool = False
    ambiguous: bool = False
    diagnostics: list[str] = field(default_factory=list)


class RouteSemanticValidator:
    """Checks LLM origin/destination roles against explicit route markers."""

    def __init__(self, airport_service: AirportService | None = None):
        self.airports = airport_service or AirportService()

    def validate_update(
        self,
        *,
        user_message: str,
        update: TravelRequirementContractUpdate,
        current_contract: TravelRequirementContract | None = None,
    ) -> RouteSemanticValidationResult:
        result = RouteSemanticValidationResult(update=update)
        self._repair_route_preservation(update, current_contract, result)
        expected = self._directional_pair(user_message)
        if not expected and update.update_type != "create_new" and "trip" not in update.field_updates:
            return result
        if expected:
            origin, destination = expected
            current_origin = _field(update.field_updates, ("trip", "origin_airport"))
            current_dest = _field(update.field_updates, ("trip", "destination_airport"))
            if current_origin != origin or current_dest != destination:
                if update.update_type in {"unknown", "advisory_question", "smalltalk", "help"}:
                    update.update_type = "create_new"
                _set_field(update.field_updates, ("trip", "origin_airport"), origin)
                _set_field(update.field_updates, ("trip", "destination_airport"), destination)
                _set_field(update.field_updates, ("trip", "origin_text"), self._city_label(origin))
                _set_field(update.field_updates, ("trip", "destination_text"), self._city_label(destination))
                update.user_facing_ack_zh = self._repair_ack(origin, destination, update.user_facing_ack_zh)
                if not update.decision_trace:
                    update.decision_trace.append(
                        DecisionTraceItem(
                            step="route_semantic_repair",
                            evidence=user_message,
                            decision="根据明确方向标记补齐出发地和目的地。",
                            affected_fields=["trip.origin_airport", "trip.destination_airport"],
                        )
                    )
                if update.next_action == "no_op" and update.update_type in {"create_new", "modify_existing", "clarification_answer"}:
                    update.next_action = "run_search"
                result.repaired = True
                result.diagnostics.append("LLM route role repaired from user text markers")
            return result

        if self._looks_like_route_request(user_message):
            mentions = self._location_mentions(user_message)
            distinct = []
            for mention in mentions:
                if mention not in distinct:
                    distinct.append(mention)
            if len(distinct) == 2:
                self._clear_trip_route_fields(update)
                update.next_action = "ask_clarification"
                update.should_search = False
                update.should_rerun_search = False
                update.should_rerank_only = False
                first = self._city_label(distinct[0])
                second = self._city_label(distinct[1])
                update.clarification_question_zh = f"你是从{first}去{second}，还是从{second}去{first}？"
                update.user_facing_ack_zh = update.clarification_question_zh
                result.ambiguous = True
                result.diagnostics.append("route direction ambiguous from two locations without directional marker")
        return result

    def _repair_route_preservation(
        self,
        update: TravelRequirementContractUpdate,
        current_contract: TravelRequirementContract | None,
        result: RouteSemanticValidationResult,
    ) -> None:
        if update.update_type != "create_new" or not current_contract or not current_contract.has_required_route():
            return
        if _field(update.field_updates, ("trip", "origin_airport")) or _field(update.field_updates, ("trip", "destination_airport")):
            return
        if not update.field_updates and not update.preferences_to_add and not update.constraints_to_add:
            return
        update.update_type = "clarification_answer"
        result.repaired = True
        result.diagnostics.append("create_new without route fields repaired to preserve current route")

    def _directional_pair(self, text: str) -> tuple[str, str] | None:
        normalized = text.strip()
        patterns = [
            r"from\s+(?P<origin>.+?)\s+(?:to|towards?)\s+(?P<dest>.+)",
            r"(?:departing\s+from|leaving\s+from)\s+(?P<origin>.+?)\s+(?:to|for)\s+(?P<dest>.+)",
            r"从(?P<origin>.+?)(?:到|去|飞|前往)(?P<dest>.+)",
            r"(?P<origin>.+?)(?:到|去|飞|前往|→|->|-)(?P<dest>.+)",
            r"(?P<origin>[A-Za-z][A-Za-z\s.-]+?)\s+to\s+(?P<dest>[A-Za-z][A-Za-z\s.-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            origin = self._resolve_fragment(match.group("origin"), prefer="last")
            dest = self._resolve_fragment(match.group("dest"), prefer="first")
            if origin and dest and origin != dest:
                return origin, dest
        return None

    def _resolve_fragment(self, fragment: str, *, prefer: str) -> str | None:
        mentions = self._location_mentions(fragment)
        if not mentions:
            return None
        return mentions[0] if prefer == "first" else mentions[-1]

    def _location_mentions(self, text: str) -> list[str]:
        found: list[tuple[int, int, str]] = []
        lowered = text.lower()
        aliases = sorted(self.airports.alias_map.items(), key=lambda item: len(item[0]), reverse=True)
        for alias, codes in aliases:
            if len(alias) < 2:
                continue
            if alias.isascii() and re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", lowered):
                code = self.airports.preferred_airport(codes)
                if code:
                    match = re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", lowered)
                    found.append((match.start(), len(alias), code))
            elif not alias.isascii() and alias in text:
                code = self.airports.preferred_airport(codes)
                if code:
                    found.append((text.index(alias), len(alias), code))
        found.sort(key=lambda item: (item[0], -item[1]))
        result: list[str] = []
        occupied: list[range] = []
        for start, length, code in found:
            span = range(start, start + length)
            if any(start in taken or start + length - 1 in taken for taken in occupied):
                continue
            occupied.append(span)
            if code not in result:
                result.append(code)
        return result

    def _looks_like_route_request(self, text: str) -> bool:
        if any(marker in text for marker in ["查一下", "飞", "航班", "机票", "路线", "行程", "去", "到"]):
            return True
        return len(self._location_mentions(text)) >= 2

    def _clear_trip_route_fields(self, update: TravelRequirementContractUpdate) -> None:
        trip = update.field_updates.get("trip")
        if isinstance(trip, dict):
            for key in [
                "origin_text",
                "origin_airport",
                "origin_city",
                "destination_text",
                "destination_airport",
                "destination_city",
            ]:
                trip.pop(key, None)

    def _city_label(self, code: str) -> str:
        row = self.airports.get(code) or {}
        return row.get("city_cn") or row.get("city") or code

    def _repair_ack(self, origin: str, dest: str, fallback: str) -> str:
        return f"明白：从{self._city_label(origin)} {origin} 到{self._city_label(dest)} {dest}。" if origin and dest else fallback


def _field(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, dict):
            return None
        if key in cursor:
            cursor = cursor[key]
            continue
        dotted = ".".join(path)
        return data.get(dotted)
    return cursor


def _set_field(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value
