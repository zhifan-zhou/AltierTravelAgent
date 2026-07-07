"""Structured planning and user-response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    label: str
    source: str
    is_live: bool = False
    note: str | None = None


ResponseType = Literal[
    "clarification",
    "tool_answer",
    "itinerary",
    "cost_estimate",
    "constraint_check",
    "flight_demo",
    "general_answer",
    "error",
]


class UserResponse(BaseModel):
    """Only object accepted by the ordinary user renderer."""

    text: str
    response_type: ResponseType = "general_answer"
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    is_streamable: bool = True


class ItineraryDay(BaseModel):
    day: int
    title: str
    morning: list[str] = Field(default_factory=list)
    afternoon: list[str] = Field(default_factory=list)
    evening: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    weather_considerations: list[str] = Field(default_factory=list)
    budget_level: Literal["low", "medium", "high", "unknown"] = "unknown"


class ItineraryPlan(BaseModel):
    destination: str
    duration_days: int
    days: list[ItineraryDay] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CostItem(BaseModel):
    category: str
    amount_min: float | None = None
    amount_max: float | None = None
    currency: str = "USD"
    confidence: Literal["low", "medium", "high"] = "low"
    source_type: Literal["live", "mock_demo", "estimate", "user_provided", "unknown"] = "unknown"
    note: str = ""


class CostEstimate(BaseModel):
    items: list[CostItem] = Field(default_factory=list)
    total_min: float | None = None
    total_max: float | None = None
    currency: str = "USD"
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)


class ConstraintFinding(BaseModel):
    category: str
    level: Literal["info", "warning", "conflict"] = "info"
    message: str
    evidence_type: Literal["live", "mock_demo", "estimate", "contract", "unknown"] = "contract"


class ConstraintCheckResult(BaseModel):
    findings: list[ConstraintFinding] = Field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return any(item.level == "conflict" for item in self.findings)
