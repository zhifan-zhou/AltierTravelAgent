"""Convert internal planning outputs into response-only user objects."""

from __future__ import annotations

from travel_agent.planning.models import (
    ConstraintCheckResult,
    CostEstimate,
    ItineraryPlan,
    SourceRef,
    UserResponse,
)


class ResponsePlanner:
    def from_text(
        self,
        text: str,
        *,
        response_type: str = "general_answer",
        sources: list[SourceRef] | None = None,
        warnings: list[str] | None = None,
    ) -> UserResponse:
        return UserResponse(
            text=text.strip(),
            response_type=response_type,
            sources=sources or [],
            warnings=warnings or [],
        )

    def itinerary(self, plan: ItineraryPlan, constraints: ConstraintCheckResult) -> UserResponse:
        budget_note = "低预算" if any(day.budget_level == "low" for day in plan.days) else ""
        prefix = f"我先按 {plan.duration_days} 天给你做一个{plan.destination}{budget_note}草案。"
        lines = [prefix, "以下是规划建议，不是预订结果，也不包含实时门票或营业时间。", ""]
        for day in plan.days:
            lines.append(f"Day {day.day}：{day.title}")
            for label, values in [("上午", day.morning), ("下午", day.afternoon), ("晚上", day.evening)]:
                for value in values:
                    lines.append(f"- {label}：{value}")
            for value in day.weather_considerations:
                lines.append(f"- 天气考虑：{value}")
            for value in day.notes:
                lines.append(f"- 备注：{value}")
            lines.append("")
        if plan.assumptions:
            lines.append("假设：")
            lines.extend(f"- {item}" for item in plan.assumptions)
        warnings = list(plan.warnings)
        warnings.extend(item.message for item in constraints.findings if item.level in {"warning", "conflict"})
        _append_sources_and_warnings(lines, plan.sources, warnings)
        return UserResponse(
            text="\n".join(lines).strip(),
            response_type="itinerary",
            sources=plan.sources,
            warnings=_dedupe(warnings),
        )

    def cost_estimate(self, estimate: CostEstimate, constraints: ConstraintCheckResult) -> UserResponse:
        lines = ["这是一份粗略预算估算，不是实时报价，也不构成金融建议。", ""]
        for item in estimate.items:
            amount = _amount_range(item.amount_min, item.amount_max, item.currency)
            label = {
                "live": "live",
                "mock_demo": "demo/mock only",
                "estimate": "rough estimate",
                "user_provided": "user provided",
                "unknown": "unknown",
            }[item.source_type]
            lines.append(f"- {item.category}：{amount} [{label}] {item.note}".rstrip())
        if estimate.total_min is not None and estimate.total_max is not None:
            lines.extend(
                [
                    "",
                    f"粗略合计：{_amount_range(estimate.total_min, estimate.total_max, estimate.currency)}",
                ]
            )
        if estimate.assumptions:
            lines.append("假设：" + "；".join(estimate.assumptions))
        warnings = list(estimate.warnings)
        warnings.extend(item.message for item in constraints.findings if item.category == "budget")
        _append_sources_and_warnings(lines, estimate.sources, warnings)
        return UserResponse(
            text="\n".join(lines).strip(),
            response_type="cost_estimate",
            sources=estimate.sources,
            warnings=_dedupe(warnings),
        )

    def constraint_check(self, result: ConstraintCheckResult) -> UserResponse:
        if not result.findings:
            text = "当前没有识别到需要额外提醒的旅行约束。实时政策和价格仍需在出发前复核。"
        else:
            lines = ["当前约束检查结果："]
            for item in result.findings:
                label = {"info": "已记录", "warning": "提醒", "conflict": "可能冲突"}[item.level]
                lines.append(f"- {label}｜{item.message}")
            text = "\n".join(lines)
        return UserResponse(text=text, response_type="constraint_check")


def _append_sources_and_warnings(lines: list[str], sources: list[SourceRef], warnings: list[str]) -> None:
    if sources:
        lines.extend(["", "数据来源："])
        lines.extend(
            f"- {item.label}：{item.source}{'（live）' if item.is_live else ''}"
            for item in sources
        )
    if warnings:
        lines.extend(["", "注意："])
        lines.extend(f"- {item}" for item in _dedupe(warnings))


def _amount_range(low: float | None, high: float | None, currency: str) -> str:
    if low is None and high is None:
        return "暂缺"
    if low == high:
        return f"{low:,.2f} {currency}"
    return f"{(low or 0):,.2f}–{(high or 0):,.2f} {currency}"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
