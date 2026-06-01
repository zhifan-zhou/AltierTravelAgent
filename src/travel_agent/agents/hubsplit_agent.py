"""HubSplit Agent — with exclusion pre-filter."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import ConstraintOutput, HubSplitOutput
from travel_agent.services.airport_service import AirportService
from travel_agent.services.constraint_compiler import ExclusionRules


class HubSplitAgent(BaseAgent[tuple[ConstraintOutput, ExclusionRules | None], HubSplitOutput]):
    name = "hubsplit"

    def __init__(self, airport_service: AirportService | None = None):
        super().__init__()
        self._airport_service = airport_service or AirportService()
        self._filter_diagnostics: dict = {}

    @property
    def last_filter_diagnostics(self) -> dict:
        return dict(self._filter_diagnostics)

    async def execute(self, data: tuple[ConstraintOutput, ExclusionRules | None]) -> HubSplitOutput:
        constraint_output, exclusions = data
        constraints = constraint_output.constraints
        self._filter_diagnostics = {}

        origin_text = constraint_output.original_request.origin_text
        dest_text = constraint_output.original_request.destination_text
        origin_code = self._airport_service.resolve_airport_code(origin_text)
        dest_code = self._airport_service.resolve_airport_code(dest_text)

        if not origin_code or not dest_code:
            return HubSplitOutput(
                plan=self._empty_plan(origin_code or "", dest_code or ""),
                search_tasks_generated=0,
            )

        if not constraints.soft.accept_nearby_hubs:
            return HubSplitOutput(
                plan=self._empty_plan(origin_code, dest_code),
                search_tasks_generated=0,
            )

        # Get raw hubs
        origin_hubs = self._airport_service.get_nearby_hubs(origin_code, "origin")
        dest_hubs = self._airport_service.get_nearby_hubs(dest_code, "destination")

        # Apply exclusion pre-filter
        removed_origin = []
        removed_dest = []
        if exclusions:
            origin_hubs = [h for h in origin_hubs
                           if not exclusions.is_airport_excluded(h.airport.code)
                           or removed_origin.append(h.airport.code)]
            dest_hubs = [h for h in dest_hubs
                         if not exclusions.is_airport_excluded(h.airport.code)
                         or removed_dest.append(h.airport.code)]

        # Check if origin/destination themselves are excluded
        if exclusions and (exclusions.is_airport_excluded(origin_code) or
                           exclusions.is_airport_excluded(dest_code)):
            self._filter_diagnostics = {
                "excluded_hubs_removed": removed_origin + removed_dest,
                "remaining_origin_hubs": [h.airport.code for h in origin_hubs],
                "remaining_destination_hubs": [h.airport.code for h in dest_hubs],
                "warning": "Origin or destination airport is excluded. Using remaining hubs only.",
            }

        has_any = len(origin_hubs) > 0 or len(dest_hubs) > 0
        if not has_any:
            return HubSplitOutput(
                plan=self._empty_plan(origin_code, dest_code),
                search_tasks_generated=0,
            )

        plan = self._airport_service.build_hub_split_plan(origin_code, dest_code,
                                                          max_origin_hubs=len(origin_hubs),
                                                          max_dest_hubs=len(dest_hubs))

        # Post-filter pairs with excluded airports
        if exclusions:
            before = len(plan.candidate_hub_pairs)
            plan.candidate_hub_pairs = [
                p for p in plan.candidate_hub_pairs
                if not exclusions.is_airport_excluded(p.origin_hub_code)
                and not exclusions.is_airport_excluded(p.destination_hub_code)
            ]
            after = len(plan.candidate_hub_pairs)
            self._filter_diagnostics = {
                "excluded_hubs_removed": removed_origin + removed_dest,
                "remaining_origin_hubs": [h.airport.code for h in origin_hubs],
                "remaining_destination_hubs": [h.airport.code for h in dest_hubs],
                "pairs_before_filter": before,
                "pairs_after_filter": after,
                "pairs_removed": before - after,
            }

        n_pairs = len(plan.candidate_hub_pairs)
        return HubSplitOutput(plan=plan, search_tasks_generated=n_pairs)

    def _empty_plan(self, origin_code: str, dest_code: str) -> "HubSplitPlan":
        from travel_agent.models.airport import HubSplitPlan
        return HubSplitPlan(origin_airport_code=origin_code, destination_airport_code=dest_code)
