"""Scoring service for itinerary ranking — includes airline quality and preference profiles."""

from __future__ import annotations

from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment
from travel_agent.models.ranking import RankedRecommendation
from travel_agent.models.user_request import SoftConstraints
from travel_agent.models.preference import get_profile_weights, DEFAULT_PROFILE
from travel_agent.services.airline_service import AirlineService


class ScoringService:
    """Deterministic scoring with preference profiles."""

    def __init__(self, soft_constraints: SoftConstraints | None = None,
                 airline_service: AirlineService | None = None,
                 profile: str | None = None):
        self._constraints = soft_constraints
        self._airline_service = airline_service or AirlineService()
        self._profile = profile or DEFAULT_PROFILE
        self._weights = self._compute_weights()

    @property
    def profile(self) -> str:
        return self._profile

    def _compute_weights(self) -> dict[str, float]:
        weights = get_profile_weights(self._profile)
        if not self._constraints:
            return weights

        # Fine-tune based on user soft constraints
        if self._constraints.prefer_lowest_price:
            weights["savings"] += 0.05
            weights["airline"] = max(0.01, weights.get("airline", 0.07) - 0.02)
        if self._constraints.prefer_comfort:
            weights["comfort"] += 0.05
            weights["airline"] += 0.02
        if self._constraints.prefer_low_risk:
            weights["risk"] += 0.05
        if self._constraints.prefer_fewer_stops:
            weights["comfort"] += 0.03
            weights["time"] += 0.03

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def score(
        self, itinerary: Itinerary, risk: RiskAssessment, baseline_price: float,
        airline_quality_score: float = 0.5,
    ) -> RankedRecommendation:
        savings = baseline_price - itinerary.total_price_usd
        savings_pct = (savings / baseline_price * 100) if baseline_price > 0 else 0

        # Budget airline penalty in airline_priority mode
        adj_airline = airline_quality_score
        if self._profile == "airline_priority" and airline_quality_score < 0.4:
            adj_airline = airline_quality_score * 0.5  # Heavy penalty

        # Extra risk penalty in low_risk mode
        adj_risk = risk.risk_score
        if self._profile == "low_risk":
            if itinerary.split_ticket_count > 0:
                adj_risk = min(1.0, risk.risk_score + 0.15)

        # Ground transfer penalty in low_risk mode
        comfort = self._score_comfort(itinerary)
        if self._profile == "low_risk" and itinerary.total_access_cost_usd > 0:
            comfort = max(0.0, comfort - 0.15)

        final = (
            self._score_savings(savings_pct) * self._weights["savings"]
            + comfort * self._weights["comfort"]
            + self._score_time(itinerary) * self._weights["time"]
            + (1.0 - adj_risk) * self._weights["risk"]
            + adj_airline * self._weights["airline"]
            + 0.5 * self._weights["preference"]
        )

        return RankedRecommendation(
            itinerary=itinerary,
            risk_assessment=risk,
            savings_vs_baseline_usd=round(savings, 2),
            savings_percentage=round(savings_pct, 1),
            savings_score=round(self._score_savings(savings_pct), 4),
            comfort_score=round(comfort, 4),
            time_score=round(self._score_time(itinerary), 4),
            risk_score=round(adj_risk, 4),
            airline_quality_score=round(adj_airline, 4),
            preference_match_score=0.5,
            final_score=round(final, 4),
        )

    def _score_savings(self, pct: float) -> float:
        if pct >= 30: return 1.0
        if pct >= 15: return 0.8
        if pct >= 5: return 0.6
        if pct >= 0: return 0.4
        return 0.2

    def _score_comfort(self, it: Itinerary) -> float:
        s = 1.0
        if it.number_of_segments >= 3: s -= 0.3
        elif it.number_of_segments >= 2: s -= 0.1
        if it.total_access_cost_usd > 200: s -= 0.15
        elif it.total_access_cost_usd > 100: s -= 0.05
        if it.total_estimated_time_hours > 30: s -= 0.2
        elif it.total_estimated_time_hours > 24: s -= 0.1
        return max(0.0, s)

    def _score_time(self, it: Itinerary) -> float:
        t = it.total_estimated_time_hours
        if t <= 18: return 1.0
        if t <= 24: return 0.8
        if t <= 30: return 0.5
        if t <= 36: return 0.3
        return 0.1
