"""Test the HubSplit Agent — the core differentiator."""

import pytest
from travel_agent.models.user_request import (
    CabinClass,
    DateWindow,
    HardConstraints,
    SearchConstraints,
    SoftConstraints,
)
from travel_agent.models.agent_outputs import IntakeOutput, ConstraintOutput


class TestHubSplitAgent:
    def test_ningbo_to_pittsburgh_generates_hubs(self, hubsplit_agent):
        intake = IntakeOutput(
            origin_text="宁波",
            destination_text="匹兹堡",
            departure_window=DateWindow(flexible=True),
            cabin=CabinClass.ECONOMY,
            accepts_nearby_hubs=True,
            accepts_split_ticket=True,
            preferences=["cheap"],
        )
        constraints = SearchConstraints(
            hard=HardConstraints(cabin=CabinClass.ECONOMY),
            soft=SoftConstraints(
                accept_nearby_hubs=True,
                accept_split_ticket=True,
                prefer_lowest_price=True,
            ),
        )
        constraint_output = ConstraintOutput(
            constraints=constraints,
            original_request=intake,
        )

        import asyncio
        result = asyncio.run(hubsplit_agent.execute((constraint_output, None)))

        # Should generate origin hubs including PVG
        origin_codes = [h.airport.code for h in result.plan.origin_hubs]
        assert "PVG" in origin_codes, f"Expected PVG in origin hubs, got {origin_codes}"

        # Should generate destination hubs including JFK, IAD, ORD
        dest_codes = [h.airport.code for h in result.plan.destination_hubs]
        for expected in ["JFK", "IAD", "ORD"]:
            assert expected in dest_codes, f"Expected {expected} in dest hubs, got {dest_codes}"

        # Should generate candidate hub pairs
        assert len(result.plan.candidate_hub_pairs) > 0
        assert result.search_tasks_generated > 0

    def test_hub_split_without_nearby_acceptance(self, hubsplit_agent):
        intake = IntakeOutput(
            origin_text="宁波",
            destination_text="匹兹堡",
            accepts_nearby_hubs=False,
            accepts_split_ticket=False,
        )
        constraints = SearchConstraints(
            hard=HardConstraints(),
            soft=SoftConstraints(accept_nearby_hubs=False, accept_split_ticket=False),
        )
        constraint_output = ConstraintOutput(constraints=constraints, original_request=intake)

        import asyncio
        result = asyncio.run(hubsplit_agent.execute((constraint_output, None)))
        assert result.plan.candidate_hub_pairs == []

    def test_pvg_jfk_pair_has_high_savings_potential(self, hubsplit_agent):
        intake = IntakeOutput(
            origin_text="宁波",
            destination_text="匹兹堡",
            accepts_nearby_hubs=True,
            accepts_split_ticket=True,
        )
        constraints = SearchConstraints(
            hard=HardConstraints(),
            soft=SoftConstraints(accept_nearby_hubs=True, accept_split_ticket=True),
        )
        constraint_output = ConstraintOutput(constraints=constraints, original_request=intake)

        import asyncio
        result = asyncio.run(hubsplit_agent.execute((constraint_output, None)))

        pvg_jfk = [p for p in result.plan.candidate_hub_pairs
                    if p.origin_hub_code == "PVG" and p.destination_hub_code == "JFK"]
        assert len(pvg_jfk) == 1
        assert pvg_jfk[0].expected_savings_potential in ("high", "medium")
