"""Strict LLM schema for requirement updates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from travel_agent.contract.models import ConstraintItem, PreferenceItem, SpecialRequirement


UpdateType = Literal[
    "create_new",
    "modify_existing",
    "add_constraint",
    "remove_constraint",
    "add_preference",
    "remove_preference",
    "advisory_question",
    "add_special_requirement",
    "remove_special_requirement",
    "clarification_answer",
    "explain_option",
    "export",
    "quit",
    "help",
    "smalltalk",
    "unknown",
]

NextAction = Literal[
    "ask_clarification",
    "answer_advisory",
    "run_search",
    "rerank",
    "explain_result",
    "export",
    "help",
    "smalltalk",
    "quit",
    "tool_query",
    "itinerary",
    "cost_estimate",
    "constraint_check",
    "no_op",
]

ACTIONABLE_UPDATE_TYPES = {
    "create_new",
    "modify_existing",
    "add_constraint",
    "remove_constraint",
    "add_preference",
    "remove_preference",
    "add_special_requirement",
    "remove_special_requirement",
    "clarification_answer",
}


class DecisionTraceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    evidence: str
    decision: str
    affected_fields: list[str] = Field(default_factory=list)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason_zh: str = ""
    requires_current_contract: bool = False


class TravelRequirementContractUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    update_type: UpdateType = "unknown"
    field_updates: dict[str, Any] = Field(default_factory=dict)
    constraints_to_add: list[ConstraintItem] = Field(default_factory=list)
    constraints_to_remove: list[str] = Field(default_factory=list)
    preferences_to_add: list[PreferenceItem] = Field(default_factory=list)
    preferences_to_remove: list[str] = Field(default_factory=list)
    special_requirements_to_add: list[SpecialRequirement] = Field(default_factory=list)
    special_requirements_to_remove: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    should_search: bool = False
    should_rerun_search: bool = False
    should_rerank_only: bool = False
    selected_option_index: int | None = None
    next_action: NextAction = "no_op"
    user_intent_summary_zh: str = ""
    advisory_response_zh: str | None = None
    clarification_question_zh: str | None = None
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    user_facing_ack_zh: str = ""
    reasoning_summary: str = ""
    decision_trace: list[DecisionTraceItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def actionable_requires_trace(self) -> "TravelRequirementContractUpdate":
        has_schema_change = bool(
            self.field_updates
            or self.constraints_to_add
            or self.constraints_to_remove
            or self.preferences_to_add
            or self.preferences_to_remove
            or self.special_requirements_to_add
            or self.special_requirements_to_remove
        )
        if (self.update_type in ACTIONABLE_UPDATE_TYPES or has_schema_change) and not self.decision_trace:
            raise ValueError("actionable update requires non-empty decision_trace")
        if self.should_rerank_only and self.should_rerun_search:
            raise ValueError("should_rerank_only and should_rerun_search cannot both be true")
        if self.next_action == "rerank" and self.should_rerun_search:
            raise ValueError("rerank action cannot request full rerun")
        return self


class RequirementUpdateEnvelope(BaseModel):
    """Useful in tests and logs when pairing raw and parsed LLM output."""

    raw_content: str
    update: TravelRequirementContractUpdate
