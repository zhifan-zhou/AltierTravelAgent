"""Test the Ranking Agent."""

import uuid
from datetime import datetime

from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment
from travel_agent.models.user_request import HardConstraints, SearchConstraints, SoftConstraints
from travel_agent.models.agent_outputs import ConstraintOutput, RiskOutput, RouteComposerResult
from travel_agent.models.itinerary import RouteComposerOutput


class TestRankingAgent:
    @staticmethod
    def _make_request():
        from travel_agent.models.user_request import UserTravelRequest
        return UserTravelRequest(origin_text="宁波", destination_text="匹兹堡")

    def _make_itinerary(self, it_id: str, price: float, it_type: str = "hub_split") -> Itinerary:
        return Itinerary(
            id=it_id,
            type=it_type,
            total_price_usd=price,
            total_access_cost_usd=50,
            total_estimated_time_hours=22,
            number_of_segments=2,
            split_ticket_count=2 if it_type == "hub_split" else 0,
        )

    def test_split_route_cheaper_ranks_higher(self, ranking_agent):
        direct = self._make_itinerary("direct-1", 1850, "direct")
        split_a = self._make_itinerary("split-a", 1215, "hub_split")
        split_b = self._make_itinerary("split-b", 1350, "hub_split")

        route_output = RouteComposerResult(
            output=RouteComposerOutput(
                itineraries=[direct, split_a, split_b],
                baseline_itinerary_id="direct-1",
            )
        )

        risk_output = RiskOutput(assessments={
            "direct-1": RiskAssessment(risk_level="low", risk_score=0.1),
            "split-a": RiskAssessment(risk_level="medium", risk_score=0.55),
            "split-b": RiskAssessment(risk_level="medium", risk_score=0.55),
        })

        constraints = ConstraintOutput(
            constraints=SearchConstraints(
                hard=HardConstraints(),
                soft=SoftConstraints(prefer_lowest_price=True),
            ),
            original_request=self._make_request(),
        )

        import asyncio
        result = asyncio.run(ranking_agent.execute((route_output, risk_output, constraints)))

        assert len(result.rankings) == 3
        # Split route should rank #1 because it's cheaper
        top = result.rankings[0]
        assert top.itinerary.type == "hub_split"

    def test_outputs_categorized_recommendations(self, ranking_agent):
        direct = self._make_itinerary("direct-1", 1850, "direct")
        split = self._make_itinerary("split-a", 1215, "hub_split")

        route_output = RouteComposerResult(
            output=RouteComposerOutput(itineraries=[direct, split])
        )
        risk_output = RiskOutput(assessments={
            "direct-1": RiskAssessment(risk_level="low", risk_score=0.1),
            "split-a": RiskAssessment(risk_level="medium", risk_score=0.55),
        })
        constraints = ConstraintOutput(
            constraints=SearchConstraints(hard=HardConstraints(), soft=SoftConstraints()),
            original_request=self._make_request(),
        )

        import asyncio
        result = asyncio.run(ranking_agent.execute((route_output, risk_output, constraints)))

        assert result.best_overall is not None
        assert result.cheapest_reasonable is not None
        assert result.lowest_risk is not None

    def test_direct_route_is_lowest_risk(self, ranking_agent):
        direct = self._make_itinerary("direct-1", 1850, "direct")
        split = self._make_itinerary("split-a", 1215, "hub_split")

        route_output = RouteComposerResult(
            output=RouteComposerOutput(itineraries=[direct, split])
        )
        risk_output = RiskOutput(assessments={
            "direct-1": RiskAssessment(risk_level="low", risk_score=0.1),
            "split-a": RiskAssessment(risk_level="medium", risk_score=0.55),
        })
        constraints = ConstraintOutput(
            constraints=SearchConstraints(hard=HardConstraints(), soft=SoftConstraints()),
            original_request=self._make_request(),
        )

        import asyncio
        result = asyncio.run(ranking_agent.execute((route_output, risk_output, constraints)))

        assert result.lowest_risk.itinerary.id == "direct-1"
