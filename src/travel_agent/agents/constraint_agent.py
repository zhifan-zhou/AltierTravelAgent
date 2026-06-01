"""Constraint Agent: converts intake output to hard/soft search constraints."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import ConstraintOutput, IntakeOutput
from travel_agent.models.user_request import (
    CabinClass,
    HardConstraints,
    SearchConstraints,
    SoftConstraints,
)


class ConstraintAgent(BaseAgent[IntakeOutput, ConstraintOutput]):
    """Convert parsed user intent into actionable search constraints."""

    name = "constraint"

    async def execute(self, data: IntakeOutput) -> ConstraintOutput:
        hard = HardConstraints(
            passengers=data.passengers,
            cabin=data.cabin,
            max_budget_usd=data.budget_usd,
            departure_date_start=data.departure_window.start_date,
            departure_date_end=data.departure_window.end_date,
        )

        # Determine soft constraint weights from preferences
        prefs = data.preferences
        soft = SoftConstraints(
            prefer_lowest_price="cheap" in prefs or not prefs,
            prefer_fewer_stops="fast" in prefs,
            prefer_comfort="comfort" in prefs or data.cabin == CabinClass.BUSINESS,
            prefer_low_risk="safe" in prefs or "family_friendly" in prefs,
            accept_nearby_hubs=data.accepts_nearby_hubs,
            accept_split_ticket=data.accepts_split_ticket,
        )

        # Family-friendly: lower risk tolerance, shorter access times
        if "family_friendly" in prefs:
            soft.max_access_time_hours = 4.0
            soft.max_layover_hours = 6.0
            soft.prefer_low_risk = True
            soft.prefer_fewer_stops = True

        constraints = SearchConstraints(hard=hard, soft=soft)
        return ConstraintOutput(
            constraints=constraints,
            original_request=data,
        )
