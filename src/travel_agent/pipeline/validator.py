"""Recommendation validation and display guards."""

from __future__ import annotations

from travel_agent.contract.compiler import ExclusionRules
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.pipeline.types import Itinerary, Recommendation, SearchTask


class RecommendationValidator:
    def validate_itinerary(
        self,
        itinerary: Itinerary,
        *,
        contract: TravelRequirementContract,
        exclusions: ExclusionRules,
    ) -> bool:
        if not itinerary.route:
            return False
        if itinerary.route[0] != contract.trip.origin_airport:
            return False
        if itinerary.route[-1] != contract.trip.destination_airport:
            return False
        excluded = set(exclusions.excluded_airports)
        if any(code in excluded for code in itinerary.route):
            return False
        for segment in itinerary.segments:
            if exclusions.airline_is_excluded(segment.airline):
                return False
        return True

    def validate_recommendations(
        self,
        recommendations: list[Recommendation],
        *,
        contract: TravelRequirementContract,
        exclusions: ExclusionRules,
    ) -> list[Recommendation]:
        valid = [
            rec
            for rec in recommendations
            if self.validate_itinerary(rec.itinerary, contract=contract, exclusions=exclusions)
        ]
        for idx, rec in enumerate(valid, start=1):
            rec.rank = idx
        return valid

    def tasks_have_no_excluded_airports(self, tasks: list[SearchTask], exclusions: ExclusionRules) -> bool:
        excluded = set(exclusions.excluded_airports)
        return all(task.origin not in excluded and task.destination not in excluded for task in tasks)

    def relaxation_suggestion(self) -> str:
        return "在当前限制下没有找到合适方案。可以放宽限制，或考虑其他出发枢纽。"
