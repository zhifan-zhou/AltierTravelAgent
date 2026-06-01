"""Generic interpretation of life/special travel requirements.

The LLM maps arbitrary user language into SpecialRequirement records. This
module converts those structured categories and impact areas into broad,
deterministic effects without adding phrase-specific chat parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.contract.models import SpecialRequirement


class SpecialRequirementEffects(BaseModel):
    risk_weight_adjustment: float = 0.0
    airline_quality_weight_adjustment: float = 0.0
    avoid_self_transfer: bool = False
    avoid_complex_transfers: bool = False
    prefer_full_service_airlines: bool = False
    require_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    warnings_to_display: list[str] = Field(default_factory=list)
    detail_view_notes: list[str] = Field(default_factory=list)


class SpecialRequirementInterpreter:
    """Maps schema-level categories/impact areas to generic downstream effects."""

    def interpret(self, requirements: list[SpecialRequirement]) -> SpecialRequirementEffects:
        effects = SpecialRequirementEffects()
        for item in requirements or []:
            if not item.active:
                continue
            impacts = {area.strip().lower() for area in item.impact_areas}
            category = item.category.strip().lower() or "unknown"
            weight = _weight_value(item.preference_weight)

            if impacts & {"risk", "connection_time", "self_transfer", "routing"}:
                effects.risk_weight_adjustment = max(effects.risk_weight_adjustment, 0.10 * weight)
            if impacts & {"self_transfer", "baggage", "connection_time"}:
                effects.avoid_self_transfer = True
                effects.avoid_complex_transfers = True
            if impacts & {"airline_policy", "cabin", "baggage"}:
                effects.airline_quality_weight_adjustment = max(
                    effects.airline_quality_weight_adjustment, 0.08 * weight
                )
                effects.prefer_full_service_airlines = True

            if category == "visa_constraint" or "documentation" in impacts:
                effects.risk_weight_adjustment = max(effects.risk_weight_adjustment, 0.18)
                effects.warnings_to_display.append("签证/过境政策需单独确认；本 demo 不判断签证可行性。")
                effects.detail_view_notes.append("签证/过境要求需要在出票前按护照、签证和转机国家单独核验。")
            if category == "pet_travel" or "airline_policy" in impacts:
                effects.warnings_to_display.append("特殊同行/航司政策需求已记录；下单前需要确认具体航司规则。")
                effects.detail_view_notes.append("需要向航司确认宠物、医疗、餐食或其他特殊政策名额与材料要求。")
            if category == "heavy_baggage":
                effects.warnings_to_display.append("行李较多时，拆票或自助转机可能增加重新托运风险。")
                effects.detail_view_notes.append("行李限制、托运额度和跨航司衔接需要在下单前复核。")
            if category in {"family_or_elderly", "child_travel", "accessibility"}:
                effects.avoid_self_transfer = True
                effects.avoid_complex_transfers = True
                effects.warnings_to_display.append("同行人照顾需求已记录，排序会更偏向低风险、少折腾。")
                effects.detail_view_notes.append("同行人照顾场景建议预留更长衔接时间，优先保护性更强的路线。")
            if category == "overnight_avoidance":
                effects.warnings_to_display.append("已记录避免红眼/过夜偏好；当前 mock 数据不含完整时刻表。")
                effects.detail_view_notes.append("真实搜索需用航班时刻表过滤红眼或过夜航段。")
            if category == "stopover_request":
                effects.warnings_to_display.append("停留需求已记录；当前 demo 会把它作为路线偏好，不生成入境/住宿安排。")
                effects.detail_view_notes.append("停留方案需要确认签证、行李直挂和分段出票规则。")

            if item.requires_clarification:
                effects.require_clarification = True
                if item.clarification_question_zh:
                    effects.clarification_questions.append(item.clarification_question_zh)

        effects.clarification_questions = _dedupe(effects.clarification_questions)
        effects.warnings_to_display = _dedupe(effects.warnings_to_display)
        effects.detail_view_notes = _dedupe(effects.detail_view_notes)
        return effects


def _weight_value(weight: str) -> float:
    return {"low": 0.5, "medium": 1.0, "high": 1.5}.get(weight, 1.0)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
