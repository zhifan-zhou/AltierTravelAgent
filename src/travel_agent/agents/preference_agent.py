"""Preference Agent: converts user query + clarification into ranking weights."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.llm.schemas import LLMPreferenceInference
from travel_agent.models.agent_outputs import IntakeOutput
from travel_agent.llm.prompts import get_llm_client, is_llm_enabled


class PreferenceAgent(BaseAgent[IntakeOutput, dict[str, float]]):
    """Infer ranking weights from user preferences.

    Outputs a dict of {price_weight, risk_weight, comfort_weight, time_weight}
    that can be used by the ScoringService/RankingAgent.
    """

    name = "preference"

    DEFAULT_WEIGHTS = {
        "price": 0.40,
        "risk": 0.25,
        "comfort": 0.15,
        "time": 0.15,
        "flexibility": 0.05,
    }

    async def execute(self, data: IntakeOutput) -> dict[str, float]:
        weights = dict(self.DEFAULT_WEIGHTS)

        # Deterministic adjustments
        prefs = data.preferences
        if "cheap" in prefs:
            weights["price"] += 0.15
            weights["risk"] -= 0.05
            weights["comfort"] -= 0.05
            weights["time"] -= 0.05
        if "comfort" in prefs:
            weights["comfort"] += 0.15
            weights["price"] -= 0.05
        if "safe" in prefs or "family_friendly" in prefs:
            weights["risk"] += 0.10
            weights["price"] -= 0.05
        if "fast" in prefs:
            weights["time"] += 0.10
            weights["price"] -= 0.05

        # Try LLM if enabled
        if is_llm_enabled("preference"):
            llm = get_llm_client()
            ctx = {
                "preferences": prefs,
                "origin": data.origin_text,
                "destination": data.destination_text,
                "deterministic_weights": weights,
            }
            llm_result = await llm.infer_preferences(data.raw_query, ctx)
            if llm_result and llm_result.confidence > 0.5:
                weights["price"] = llm_result.price_weight
                weights["risk"] = llm_result.risk_weight
                weights["comfort"] = llm_result.comfort_weight
                weights["time"] = llm_result.time_weight
                weights["flexibility"] = llm_result.flexibility_weight

        # Clamp to safe ranges
        for k in weights:
            weights[k] = max(0.0, min(1.0, weights[k]))

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}

        return weights
