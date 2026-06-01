"""Search Strategy Agent — with exclusion pre-filter."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.core.config import get_settings
from travel_agent.models.agent_outputs import HubSplitOutput, SearchStrategyOutput
from travel_agent.models.flight import FlightSearchRequest
from travel_agent.services.constraint_compiler import ExclusionRules


class SearchStrategyAgent(BaseAgent[tuple[HubSplitOutput, ExclusionRules | None], SearchStrategyOutput]):
    """Generate flight search tasks, dropping excluded airports."""

    name = "search_strategy"

    async def execute(self, data: tuple[HubSplitOutput, ExclusionRules | None]) -> SearchStrategyOutput:
        hub_split_output, exclusions = data
        plan = hub_split_output.plan
        settings = get_settings()
        window = settings.default_date_window_days
        tasks: list[FlightSearchRequest] = []
        hub_split_tasks: list[FlightSearchRequest] = []
        direct_task: FlightSearchRequest | None = None
        dropped: list[dict] = []
        before_count = 0

        # 1. Direct route search
        if plan.origin_airport_code and plan.destination_airport_code:
            if exclusions and (exclusions.is_airport_excluded(plan.origin_airport_code) or
                               exclusions.is_airport_excluded(plan.destination_airport_code)):
                pass  # Skip direct if excluded
            else:
                direct_task = FlightSearchRequest(
                    origin=plan.origin_airport_code, destination=plan.destination_airport_code,
                    flexible_dates=True, date_window_days=window,
                )
                tasks.append(direct_task)

        # 2. International hub-to-hub searches
        seen = set()
        for pair in plan.candidate_hub_pairs:
            key = f"{pair.origin_hub_code}->{pair.destination_hub_code}"
            if key in seen:
                continue
            seen.add(key)
            before_count += 1
            if exclusions and (exclusions.is_airport_excluded(pair.origin_hub_code) or
                               exclusions.is_airport_excluded(pair.destination_hub_code)):
                dropped.append({"route": key, "reason": exclusions.explain_exclusion(pair.origin_hub_code)})
                continue
            task = FlightSearchRequest(
                origin=pair.origin_hub_code, destination=pair.destination_hub_code,
                flexible_dates=True, date_window_days=window,
            )
            tasks.append(task)
            hub_split_tasks.append(task)

        # 3. Domestic connection searches
        seen_domestic = set()
        for dh in plan.destination_hubs:
            key = f"{dh.airport.code}->{plan.destination_airport_code}"
            if key in seen_domestic:
                continue
            seen_domestic.add(key)
            if exclusions and (exclusions.is_airport_excluded(dh.airport.code) or
                               exclusions.is_airport_excluded(plan.destination_airport_code)):
                dropped.append({"route": key, "reason": exclusions.explain_exclusion(dh.airport.code)})
                continue
            task = FlightSearchRequest(
                origin=dh.airport.code, destination=plan.destination_airport_code,
                flexible_dates=True, date_window_days=window,
            )
            tasks.append(task)

        self._last_filter_diagnostics = {
            "tasks_before_filter": before_count + len(seen_domestic),
            "tasks_after_filter": len(tasks),
            "dropped_tasks": dropped,
        }

        return SearchStrategyOutput(
            search_tasks=tasks, direct_task=direct_task, hub_split_tasks=hub_split_tasks,
        )
