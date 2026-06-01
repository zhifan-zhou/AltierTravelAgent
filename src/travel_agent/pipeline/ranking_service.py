"""Rank recommendations deterministically."""

from __future__ import annotations

from statistics import mean

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.contract.special_requirements import SpecialRequirementInterpreter
from travel_agent.pipeline.risk_scorer import RiskScorer
from travel_agent.pipeline.types import Itinerary, Recommendation
from travel_agent.services.airline_service import AirlineService


PROFILE_WEIGHTS = {
    "balanced": {"price": 0.45, "risk": 0.25, "time": 0.15, "airline": 0.15},
    "cheapest": {"price": 0.70, "risk": 0.15, "time": 0.05, "airline": 0.10},
    "airline_priority": {"price": 0.25, "risk": 0.20, "time": 0.10, "airline": 0.45},
    "low_risk": {"price": 0.25, "risk": 0.50, "time": 0.15, "airline": 0.10},
    "fastest": {"price": 0.25, "risk": 0.15, "time": 0.45, "airline": 0.15},
}


class RankingService:
    def __init__(self, airline_service: AirlineService | None = None, risk_scorer: RiskScorer | None = None):
        self.airlines = airline_service or AirlineService()
        self.risk_scorer = risk_scorer or RiskScorer()
        self.specials = SpecialRequirementInterpreter()

    def rank(self, itineraries: list[Itinerary], contract: TravelRequirementContract) -> list[Recommendation]:
        if not itineraries:
            return []
        baseline = self._baseline_price(itineraries)
        min_price = min(it.total_price_usd for it in itineraries)
        max_price = max(it.total_price_usd for it in itineraries)
        min_time = min(it.total_estimated_time_hours for it in itineraries)
        max_time = max(it.total_estimated_time_hours for it in itineraries)
        weights = dict(PROFILE_WEIGHTS.get(contract.ranking.profile, PROFILE_WEIGHTS["balanced"]))
        weights = self._apply_special_requirement_weights(weights, contract)

        recommendations: list[Recommendation] = []
        for itinerary in itineraries:
            risk = self.risk_scorer.score(itinerary, contract)
            airline_quality = self._airline_quality(itinerary)
            price_score = _inverse_score(itinerary.total_price_usd, min_price, max_price)
            time_score = _inverse_score(itinerary.total_estimated_time_hours, min_time, max_time)
            risk_score = 1.0 - risk.risk_score
            score = (
                weights["price"] * price_score
                + weights["risk"] * risk_score
                + weights["time"] * time_score
                + weights["airline"] * airline_quality
            )
            recommendations.append(
                Recommendation(
                    rank=0,
                    recommendation_type="拆分" if itinerary.route_type == "hub_split" else "基准",
                    itinerary=itinerary,
                    score=round(score, 4),
                    savings_vs_baseline_usd=max(0.0, baseline - itinerary.total_price_usd),
                    risk=risk,
                    airline_quality_score=round(airline_quality, 2),
                    reason_zh=self._reason(contract, itinerary, risk, baseline),
                )
            )
        recommendations.sort(key=lambda rec: (-rec.score, rec.itinerary.total_price_usd))
        for idx, rec in enumerate(recommendations, start=1):
            rec.rank = idx
        return recommendations

    def _baseline_price(self, itineraries: list[Itinerary]) -> float:
        baseline_prices = [it.total_price_usd for it in itineraries if it.route_type == "baseline"]
        return min(baseline_prices) if baseline_prices else max(it.total_price_usd for it in itineraries)

    def _airline_quality(self, itinerary: Itinerary) -> float:
        scores = [self.airlines.quality_score(code) for code in itinerary.airlines]
        return mean(scores) if scores else 0.5

    def _reason(self, contract: TravelRequirementContract, itinerary: Itinerary, risk, baseline: float) -> str:
        if contract.special_requirements and risk.warnings:
            return "已结合特殊需求调整风险和排序，优先减少衔接不确定性。"
        if contract.ranking.profile == "cheapest":
            return "价格权重最高，优先展示总价更低的组合。"
        if contract.ranking.profile == "airline_priority":
            return "按主流航司和航司质量优先重新排序。"
        if contract.ranking.profile == "low_risk":
            return "优先低风险、少折腾，拆分方案会被扣分。"
        if baseline > itinerary.total_price_usd:
            return "相比基准方案有节省，同时满足当前硬约束。"
        return "满足当前硬约束的可行方案。"

    def _apply_special_requirement_weights(
        self,
        weights: dict[str, float],
        contract: TravelRequirementContract,
    ) -> dict[str, float]:
        effects = self.specials.interpret(contract.special_requirements)
        if effects.risk_weight_adjustment <= 0 and effects.airline_quality_weight_adjustment <= 0:
            return weights
        weights["risk"] = weights.get("risk", 0) + effects.risk_weight_adjustment
        weights["airline"] = weights.get("airline", 0) + effects.airline_quality_weight_adjustment
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {key: value / total for key, value in weights.items()}


def _inverse_score(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0
    return 1.0 - ((value - min_value) / (max_value - min_value))
