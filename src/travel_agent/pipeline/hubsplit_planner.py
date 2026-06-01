"""Hub split candidate planner."""

from __future__ import annotations

from travel_agent.contract.compiler import SearchConstraints
from travel_agent.pipeline.types import HubCandidatePair
from travel_agent.services.airport_service import AirportService


class HubSplitPlanner:
    def __init__(self, airport_service: AirportService | None = None):
        self.airports = airport_service or AirportService()

    def plan(self, constraints: SearchConstraints) -> list[HubCandidatePair]:
        origin = constraints.origin_airport
        destination = constraints.destination_airport
        excluded = set(constraints.exclusions.excluded_airports)
        if origin in excluded or destination in excluded:
            return []

        origin_hubs = self._origin_hubs(origin, constraints)
        destination_hubs = self._destination_hubs(destination, constraints)
        pairs: list[HubCandidatePair] = []
        for origin_hub, origin_cost in origin_hubs:
            if origin_hub in excluded:
                continue
            for dest_hub, dest_cost in destination_hubs:
                if dest_hub in excluded:
                    continue
                if origin_hub == origin and dest_hub == destination:
                    continue
                pair_id = f"{origin}-{origin_hub}-{dest_hub}-{destination}"
                pairs.append(
                    HubCandidatePair(
                        pair_id=pair_id,
                        origin_airport=origin,
                        origin_hub=origin_hub,
                        destination_hub=dest_hub,
                        destination_airport=destination,
                        origin_access_cost_usd=origin_cost,
                        destination_access_cost_usd=dest_cost,
                        expected_savings_potential="high" if origin_hub != origin and dest_hub != destination else "medium",
                    )
                )
        return pairs[:40]

    def _origin_hubs(self, origin: str, constraints: SearchConstraints) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        origin_row = self.airports.get(origin)
        if origin_row and origin_row.get("is_international_hub"):
            values.append((origin, 0))
        for code in constraints.origin_hubs:
            values.append((code, 0))
        if constraints.nearby_hub_policy in {"allow", "prefer"}:
            for hub in self.airports.nearby_origin_hubs(origin):
                values.append((hub["hub_code"].upper(), float(hub.get("access_cost_usd", 0))))
        if not values:
            values.append((origin, 0))
        return _dedupe_pairs(values)

    def _destination_hubs(self, destination: str, constraints: SearchConstraints) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        destination_row = self.airports.get(destination)
        if destination_row and destination_row.get("is_international_hub"):
            values.append((destination, 0))
        for code in [*constraints.transfer_hubs, *constraints.destination_hubs]:
            values.append((code, 0))
        nearby = self.airports.nearby_destination_hubs(destination)
        if nearby:
            for hub in nearby:
                values.append((hub["hub_code"].upper(), float(hub.get("access_cost_usd", 0))))
        else:
            for code in self.airports.all_us_hub_defaults():
                if code != destination:
                    values.append((code, 180))
        if not values:
            values.append((destination, 0))
        return _dedupe_pairs(values)


def _dedupe_pairs(values: list[tuple[str, float]]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for code, cost in values:
        code = code.upper()
        if code not in seen:
            seen.add(code)
            result.append((code, cost))
    return result
