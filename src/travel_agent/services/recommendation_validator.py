"""Validates recommendations against exclusion rules before display."""

from __future__ import annotations

from travel_agent.services.constraint_compiler import ExclusionRules
from travel_agent.models.ranking import RankedRecommendation


class RecommendationValidator:
    """Final safety net: remove itineraries violating hard constraints."""

    def validate(self, recs: list[RankedRecommendation],
                 exclusions: ExclusionRules) -> list[RankedRecommendation]:
        valid = []
        for rec in recs:
            if self._is_valid(rec, exclusions):
                valid.append(rec)
        return valid

    def _is_valid(self, rec: RankedRecommendation, exclusions: ExclusionRules) -> bool:
        it = rec.itinerary

        for seg in it.segments:
            if seg.origin in exclusions.excluded_airports:
                return False
            if seg.destination in exclusions.excluded_airports:
                return False
            if seg.airline and seg.airline in exclusions.excluded_airlines:
                return False

        if it.origin_airport in exclusions.excluded_airports:
            return False
        if it.destination_airport in exclusions.excluded_airports:
            return False

        return True

    def get_relaxation_suggestion(self, exclusions: ExclusionRules) -> str:
        if exclusions.excluded_airports:
            names = ", ".join(exclusions.excluded_airports[:3])
            return f"在当前限制下没有找到合适方案。你可以放宽限制，例如允许{names}中转。"
        return "在当前限制下没有找到合适方案。你可以放宽中转/航司限制试试。"
