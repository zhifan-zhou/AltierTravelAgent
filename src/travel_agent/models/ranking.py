"""Ranking models for itinerary recommendations."""

from datetime import datetime

from pydantic import BaseModel, Field

from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment


class RankedRecommendation(BaseModel):
    """A single ranked itinerary recommendation."""
    itinerary: Itinerary
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    savings_vs_baseline_usd: float = 0.0
    savings_percentage: float = 0.0
    savings_score: float = 0.0
    comfort_score: float = 0.0
    time_score: float = 0.0
    risk_score: float = 0.0
    airline_quality_score: float = 0.5
    preference_match_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    recommendation_type: str = ""  # best_overall, cheapest_reasonable, lowest_risk


class RankingOutput(BaseModel):
    """Full ranking output with categorized recommendations."""
    rankings: list[RankedRecommendation] = Field(default_factory=list)
    best_overall: RankedRecommendation | None = None
    cheapest_reasonable: RankedRecommendation | None = None
    lowest_risk: RankedRecommendation | None = None
    generated_at: datetime = Field(default_factory=datetime.now)
