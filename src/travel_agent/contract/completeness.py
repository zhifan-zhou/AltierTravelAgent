"""Requirement completeness checks for search readiness."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.contract.models import Assumption, TravelRequirementContract


class RequirementCompletenessResult(BaseModel):
    ready_to_search: bool
    missing_required_fields: list[str] = Field(default_factory=list)
    missing_recommended_fields: list[str] = Field(default_factory=list)
    should_ask_clarification: bool = False
    clarification_question_zh: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    reason: str = ""
    assumptions_added: list[Assumption] = Field(default_factory=list)

    @property
    def missing_fields(self) -> list[str]:
        return self.missing_required_fields

    @property
    def clarification_questions(self) -> list[str]:
        return [self.clarification_question_zh] if self.clarification_question_zh else []


CompletenessResult = RequirementCompletenessResult


class RequirementCompletenessChecker:
    """Separates mandatory search fields from recommended trip details."""

    def check(self, contract: TravelRequirementContract) -> RequirementCompletenessResult:
        missing_required: list[str] = []
        missing_recommended: list[str] = []
        question: str | None = None

        if not contract.trip.origin_airport:
            missing_required.append("trip.origin_airport")
            question = "我还缺一个关键信息：你要从哪里出发？"
        if not contract.trip.destination_airport:
            missing_required.append("trip.destination_airport")
            if question is None:
                question = "我还缺一个关键信息：你要去哪里？"

        added: list[Assumption] = []
        assumptions: list[str] = []
        if not contract.has_searchable_time():
            missing_required.append("time.departure_window")
            if question is None:
                question = "你大概什么时候出发？可以给一个日期或时间范围，比如 6月初、下周、8月20日前后。"
        elif contract.time.flexible_date_confirmed:
            text = "日期未定，按灵活时间窗口先看 demo 方案"
            assumptions.append(text)
            if not any(a.field == "time.departure_date" for a in contract.assumptions):
                assumption = Assumption(
                    field="time.departure_date",
                    value="flexible_default_window",
                    reason=text,
                    can_user_override=True,
                )
                contract.assumptions.append(assumption)
                added.append(assumption)

        if contract.passengers.passenger_count <= 0:
            contract.passengers.passenger_count = 1
        if not contract.cabin.cabin:
            contract.cabin.cabin = "economy"
        if contract.passengers.passenger_count == 1:
            missing_recommended.append("passengers.passenger_count")
        if contract.cabin.cabin == "economy":
            missing_recommended.append("cabin.cabin")
        if contract.ranking.profile == "balanced":
            missing_recommended.append("ranking.preference")

        ready = not missing_required
        contract.ready_to_search = ready
        contract.unresolved_questions = [question] if question else []
        return RequirementCompletenessResult(
            ready_to_search=ready,
            missing_required_fields=missing_required,
            missing_recommended_fields=missing_recommended,
            should_ask_clarification=not ready,
            clarification_question_zh=question,
            assumptions=assumptions,
            assumptions_added=added,
            reason="ready" if ready else "missing mandatory search fields",
        )
