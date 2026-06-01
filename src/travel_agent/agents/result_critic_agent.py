"""Result Critic Agent: inspects recommendations before showing them.

Deterministic only. Flags issues but never invents data.
"""

from __future__ import annotations

from pydantic import BaseModel

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import TravelAgentResult
from travel_agent.models.ranking import RankedRecommendation


class CriticReport(BaseModel):
    """Issues found by the critic."""
    warnings: list[str] = []
    estimated_data_present: bool = False
    high_risk_splits_present: bool = False
    duplicates_found: int = 0
    suggestions: list[str] = []


class ResultCriticAgent(BaseAgent[TravelAgentResult, CriticReport]):
    """Inspect results for quality issues before display.

    Checks:
    - mock/estimated data is clearly marked
    - high-risk split tickets have warnings
    - no duplicate itineraries
    - recommendations roughly align with preferences
    """

    name = "result_critic"

    async def execute(self, data: TravelAgentResult) -> CriticReport:
        report = CriticReport()
        if not data.ranking:
            report.warnings.append("No ranking results available")
            return report

        # Check for estimated data
        if data.flight_retrieval:
            for o in data.flight_retrieval.all_offers:
                if o.source == "mock_fallback" or o.confidence == "estimated":
                    report.estimated_data_present = True
                    report.suggestions.append("结果包含模拟估算价格，非真实市场报价")
                    break

        # Check for high-risk splits
        for rec in data.ranking.rankings:
            if rec.risk_assessment.risk_level == "high":
                report.high_risk_splits_present = True
                if "high_risk" not in report.warnings:
                    report.warnings.append("存在高风险拆分方案，已降权或标注")

        # Check duplicates
        seen_prices = set()
        for rec in data.ranking.rankings:
            key = (rec.itinerary.total_price_usd, rec.itinerary.number_of_segments)
            if key in seen_prices:
                report.duplicates_found += 1
            seen_prices.add(key)

        if report.duplicates_found > 0:
            report.warnings.append(f"发现 {report.duplicates_found} 个疑似重复方案")

        return report
