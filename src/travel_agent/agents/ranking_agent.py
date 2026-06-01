"""Ranking Agent: deterministic scoring and ranking of itineraries."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import ConstraintOutput, RiskOutput, RouteComposerResult
from travel_agent.models.itinerary import Itinerary
from travel_agent.models.ranking import RankingOutput
from travel_agent.services.scoring_service import ScoringService
from travel_agent.services.airline_service import AirlineService


class RankingAgent(BaseAgent[tuple[RouteComposerResult, RiskOutput, ConstraintOutput], RankingOutput]):
    """Score and rank itineraries using deterministic scoring."""

    name = "ranking"

    def __init__(self, airline_service: AirlineService | None = None):
        super().__init__()
        self._airline_service = airline_service or AirlineService()

    async def execute(
        self, data: tuple[RouteComposerResult, RiskOutput, ConstraintOutput],
        profile: str = "balanced",
    ) -> RankingOutput:
        route_result, risk_output, constraint_output = data
        soft = constraint_output.constraints.soft

        scoring = ScoringService(soft_constraints=soft,
                                 airline_service=self._airline_service,
                                 profile=profile)

        baseline_price = 0.0
        for it in route_result.output.itineraries:
            if it.type == "direct":
                baseline_price = it.total_price_usd
                break
        if baseline_price == 0:
            from travel_agent.core.config import get_settings
            baseline_price = get_settings().default_baseline_price_usd

        rankings = []
        for it in route_result.output.itineraries:
            risk = risk_output.assessments.get(it.id)
            if risk is None:
                continue
            airline_info = self._airline_service.summarize_airlines(it)
            rec = scoring.score(it, risk, baseline_price,
                                airline_quality_score=airline_info["airline_quality_score"])
            rankings.append(rec)

        # Sort by final score descending
        rankings.sort(key=lambda r: r.final_score, reverse=True)

        # Assign ranks
        for i, rec in enumerate(rankings):
            rec.rank = i + 1

        # Categorize
        best_overall = rankings[0] if rankings else None
        if best_overall:
            best_overall.recommendation_type = "best_overall"

        cheapest_reasonable = None
        for rec in rankings:
            if rec.risk_assessment.risk_level in ("low", "medium"):
                if cheapest_reasonable is None or rec.itinerary.total_price_usd < cheapest_reasonable.itinerary.total_price_usd:
                    cheapest_reasonable = rec
        if cheapest_reasonable:
            cheapest_reasonable.recommendation_type = "cheapest_reasonable"

        lowest_risk = None
        for rec in rankings:
            if lowest_risk is None or rec.risk_score < lowest_risk.risk_score:
                lowest_risk = rec
        if lowest_risk:
            lowest_risk.recommendation_type = "lowest_risk"

        return RankingOutput(
            rankings=rankings,
            best_overall=best_overall,
            cheapest_reasonable=cheapest_reasonable,
            lowest_risk=lowest_risk,
        )
