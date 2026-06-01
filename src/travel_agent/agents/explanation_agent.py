"""Explanation Agent: converts structured results into human-readable explanations."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import (
    ConstraintOutput,
    ExplanationOutput,
    HubSplitOutput,
    RankingOutput,
)
from travel_agent.utils.money import format_usd


class ExplanationAgent(BaseAgent[tuple[RankingOutput, HubSplitOutput, ConstraintOutput], ExplanationOutput]):
    """Generate human-readable (Chinese + English) explanations from structured results.

    All explanations are based on actual data — never hallucinated.
    """

    name = "explanation"

    async def execute(
        self, data: tuple[RankingOutput, HubSplitOutput, ConstraintOutput]
    ) -> ExplanationOutput:
        ranking, hub_split, constraints = data

        per_itinerary: dict[str, str] = {}
        not_recommended: list[str] = []

        for rec in ranking.rankings:
            it = rec.itinerary
            zh = self._explain_itinerary_zh(rec, hub_split)
            per_itinerary[it.id] = zh

            if rec.risk_assessment.risk_level == "high":
                not_recommended.append(
                    f"方案 {it.id}: {rec.risk_assessment.warnings[0] if rec.risk_assessment.warnings else '高风险'}"
                )

        # Summary
        parts_zh = []
        if ranking.best_overall:
            rec = ranking.best_overall
            it = rec.itinerary
            parts_zh.append(
                f"综合推荐：方案 {it.id}，预估总价 {format_usd(it.total_price_usd)}，"
                f"比 OTA 直接搜索约省 {format_usd(rec.savings_vs_baseline_usd)} ({rec.savings_percentage:.0f}%)。"
                f"风险等级：{rec.risk_assessment.risk_level}。"
            )
        if ranking.cheapest_reasonable and ranking.cheapest_reasonable != ranking.best_overall:
            rec = ranking.cheapest_reasonable
            parts_zh.append(
                f"最省钱合理方案：方案 {rec.itinerary.id}，{format_usd(rec.itinerary.total_price_usd)}，"
                f"风险 {rec.risk_assessment.risk_level}。"
            )
        if ranking.lowest_risk:
            rec = ranking.lowest_risk
            parts_zh.append(
                f"最低风险方案：方案 {rec.itinerary.id}，风险评分 {rec.risk_score:.2f}。"
            )

        summary_zh = "\n".join(parts_zh)
        summary_en = ""  # MVP: Chinese-only

        return ExplanationOutput(
            summary_zh=summary_zh,
            summary_en=summary_en,
            per_itinerary_explanation=per_itinerary,
            not_recommended_explanations=not_recommended,
        )

    def _explain_itinerary_zh(self, rec, hub_split) -> str:
        it = rec.itinerary
        risk = rec.risk_assessment
        lines = [f"## 方案 {it.id} (排名 #{rec.rank}, 综合评分 {rec.final_score:.2f})"]

        if it.type == "direct":
            lines.append(f"类型：传统 OTA 联程搜索")
        elif it.type == "hub_split":
            lines.append(f"类型：枢纽拆分方案")
            if it.main_international_leg:
                lines.append(f"国际主航段：{it.main_international_leg}")
            lines.append(f"接驳成本：{format_usd(it.total_access_cost_usd)}")
            lines.append(f"分段数：{it.number_of_segments} 段，分开出票 {it.split_ticket_count} 张")

        lines.append(f"总价：{format_usd(it.total_price_usd)}")
        lines.append(f"预估总时间：{it.total_estimated_time_hours:.1f} 小时")

        if rec.savings_vs_baseline_usd > 0:
            lines.append(f"比 OTA 基线省：{format_usd(rec.savings_vs_baseline_usd)} ({rec.savings_percentage:.0f}%)")
        elif rec.savings_vs_baseline_usd < 0:
            lines.append(f"比 OTA 基线贵：{format_usd(-rec.savings_vs_baseline_usd)}")

        lines.append(f"风险等级：{risk.risk_level} (评分 {risk.risk_score:.2f})")

        if risk.warnings:
            lines.append("风险提示：")
            for w in risk.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)
