"""Search Budget Agent: controls provider call explosion.

Prunes search tasks when using real (non-mock) providers to stay
within API rate limits and cost budgets.
"""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.core.config import get_settings
from travel_agent.models.agent_outputs import SearchStrategyOutput
from travel_agent.models.airport import HubPair


class SearchBudgetAgent(BaseAgent[SearchStrategyOutput, SearchStrategyOutput]):
    """Prune search tasks to stay within provider budgets."""

    name = "search_budget"

    async def execute(self, data: SearchStrategyOutput) -> SearchStrategyOutput:
        settings = get_settings()
        is_real = settings.travel_agent_provider not in ("mock", "")

        if not is_real:
            limit = settings.max_search_tasks_mock
        else:
            limit = settings.max_search_tasks_real

        tasks = data.search_tasks
        if len(tasks) <= limit:
            return data

        # Keep direct task + top hub_split tasks
        kept = []
        if data.direct_task:
            kept.append(data.direct_task)

        # Fill remaining budget with hub_split tasks
        remaining = limit - len(kept)
        kept.extend(data.hub_split_tasks[:remaining])

        return SearchStrategyOutput(
            search_tasks=kept,
            direct_task=data.direct_task,
            hub_split_tasks=kept[1:],  # everything after direct is hub_split
        )
