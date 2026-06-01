"""Risk scoring for composed itineraries."""

from __future__ import annotations

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.contract.special_requirements import SpecialRequirementInterpreter
from travel_agent.pipeline.types import Itinerary, RiskAssessment


class RiskScorer:
    def __init__(self, special_interpreter: SpecialRequirementInterpreter | None = None):
        self.specials = special_interpreter or SpecialRequirementInterpreter()

    def score(self, itinerary: Itinerary, contract: TravelRequirementContract) -> RiskAssessment:
        score = 0.18
        warnings: list[str] = []
        special_effects = self.specials.interpret(contract.special_requirements)
        if itinerary.route_type == "hub_split":
            score += 0.18
            warnings.append("拆分/多段方案需要更仔细安排衔接。")
        if len(itinerary.segments) >= 3:
            score += 0.12
            warnings.append("航段较多，误机和行李衔接风险更高。")
        if itinerary.has_estimated_data:
            score += 0.08
            warnings.append("部分价格为 mock fallback 估算。")
        if contract.ticketing.split_ticket_policy == "avoid" and itinerary.route_type == "hub_split":
            score += 0.18
            warnings.append("当前偏好避免拆票，本方案风险扣分。")
        if contract.passengers.family_or_parents and len(itinerary.segments) >= 3:
            score += 0.10
            warnings.append("家人同行时，多段中转更折腾。")
        if special_effects.risk_weight_adjustment:
            score += special_effects.risk_weight_adjustment
        if special_effects.avoid_self_transfer and itinerary.route_type == "hub_split":
            score += 0.12
            warnings.append("当前特殊需求不适合过多自助衔接或拆票。")
        warnings.extend(special_effects.warnings_to_display)
        level = "low" if score < 0.35 else "medium" if score < 0.65 else "high"
        return RiskAssessment(risk_score=round(min(score, 0.95), 2), risk_level=level, warnings=warnings)
