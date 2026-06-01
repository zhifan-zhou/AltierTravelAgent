"""TravelRequirementContract v2 — robust unified schema for travel requirements.

Single source of truth for all downstream agents.
LLM builds/updates this contract from multi-turn conversation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TripRequirement(BaseModel):
    primary_origin_text: str | None = None
    primary_origin_airport: str | None = None
    primary_origin_city: str | None = None
    primary_destination_text: str | None = None
    primary_destination_airport: str | None = None
    primary_destination_city: str | None = None
    trip_type: str = "one_way"
    purpose: str = "unknown"
    route_locked: bool = False


class TimeRequirement(BaseModel):
    departure_text: str | None = None
    departure_start_date: str | None = None
    departure_end_date: str | None = None
    departure_flexible: bool = True
    return_text: str | None = None
    time_of_day_preference: str = "any"
    max_total_duration_hours: float | None = None
    max_layover_hours: float | None = None


class PassengerRequirement(BaseModel):
    passenger_count: int = 1
    senior_or_parent_traveling: bool = False
    student_travel: bool = False
    baggage_count: int | None = None
    baggage_heavy: bool = False


class CabinRequirement(BaseModel):
    cabin: str = "economy"
    cabin_flexible: bool = True


class GeographyRequirement(BaseModel):
    acceptable_origin_hubs: list[str] = Field(default_factory=list)
    acceptable_destination_hubs: list[str] = Field(default_factory=list)
    acceptable_transfer_hubs: list[str] = Field(default_factory=list)
    preferred_origin_hubs: list[str] = Field(default_factory=list)
    preferred_transfer_hubs: list[str] = Field(default_factory=list)
    avoid_airports: list[str] = Field(default_factory=list)
    avoid_cities: list[str] = Field(default_factory=list)
    unresolved_locations: list[str] = Field(default_factory=list)


class HubPreferences(BaseModel):
    nearby_hub_policy: str = "allow"
    allow_ground_access_to_origin_hub: bool = True
    max_origin_ground_access_hours: float | None = None
    allow_destination_domestic_connection: bool = True
    avoid_complex_transfers: bool = False


class TicketingPreferences(BaseModel):
    split_ticket_policy: str = "allow"
    allow_self_transfer: bool = True
    allow_overnight_connection: bool = False


class RankingPreferences(BaseModel):
    profile: str = "balanced"
    price_priority: str = "medium"
    time_priority: str = "medium"
    comfort_priority: str = "medium"
    airline_quality_priority: str = "medium"


class AirlinePreferences(BaseModel):
    preferred_airlines: list[str] = Field(default_factory=list)
    avoid_airlines: list[str] = Field(default_factory=list)
    prefer_major_airlines: bool = False
    avoid_low_cost_carriers: bool = False


class RiskPreferences(BaseModel):
    risk_tolerance: str = "medium"
    avoid_baggage_recheck: bool = False
    avoid_short_connection: bool = False
    avoid_separate_ticket_risk: bool = False
    family_friendly: bool = False


class BudgetPreferences(BaseModel):
    budget_usd: float | None = None
    budget_flexible: bool = True
    save_money_priority: str = "medium"


class ConstraintItem(BaseModel):
    type: str = "other"
    value: str = ""
    reason: str = ""
    source_user_message: str = ""
    active: bool = True


class ConstraintSet(BaseModel):
    hard_constraints: list[ConstraintItem] = Field(default_factory=list)
    soft_preferences: list[ConstraintItem] = Field(default_factory=list)


class Assumption(BaseModel):
    field: str = ""
    assumed_value: str = ""
    reason: str = ""
    can_user_override: bool = True


class ClarificationNeed(BaseModel):
    field: str = ""
    question_zh: str = ""
    priority: str = "medium"
    blocks_search: bool = False


class ConversationMetadata(BaseModel):
    original_user_goal: str | None = None
    last_user_message: str | None = None
    last_update_summary_zh: str | None = None
    update_count: int = 0
    contract_history: list[dict] = Field(default_factory=list)
    source: str = "chat"


class TravelRequirementContract(BaseModel):
    schema_version: str = "v2"
    trip: TripRequirement = Field(default_factory=TripRequirement)
    time: TimeRequirement = Field(default_factory=TimeRequirement)
    passengers: PassengerRequirement = Field(default_factory=PassengerRequirement)
    cabin: CabinRequirement = Field(default_factory=CabinRequirement)
    geography: GeographyRequirement = Field(default_factory=GeographyRequirement)
    hub_preferences: HubPreferences = Field(default_factory=HubPreferences)
    ticketing_preferences: TicketingPreferences = Field(default_factory=TicketingPreferences)
    ranking_preferences: RankingPreferences = Field(default_factory=RankingPreferences)
    airline_preferences: AirlinePreferences = Field(default_factory=AirlinePreferences)
    risk_preferences: RiskPreferences = Field(default_factory=RiskPreferences)
    budget_preferences: BudgetPreferences = Field(default_factory=BudgetPreferences)
    constraints: ConstraintSet = Field(default_factory=ConstraintSet)
    conversation_metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)
    assumptions: list[Assumption] = Field(default_factory=list)
    unresolved_questions: list[ClarificationNeed] = Field(default_factory=list)
    ready_to_search: bool = False
    contract_confidence: float = 0.0

    def has_required_route(self) -> bool:
        return bool(
            (self.trip.primary_origin_text or self.trip.primary_origin_airport)
            and (self.trip.primary_destination_text or self.trip.primary_destination_airport)
        )

    def add_hard_constraint(self, ctype: str, value: str, reason: str = "", source: str = ""):
        self.constraints.hard_constraints.append(ConstraintItem(
            type=ctype, value=value, reason=reason, source_user_message=source,
        ))

    def has_excluded_airport(self, code: str) -> bool:
        return code.upper() in {a.upper() for a in self.geography.avoid_airports}

    def normalize(self) -> dict:
        """Normalize all airport lists: deduplicate and remove conflicts.
        Returns diagnostics dict with counts of fixes applied.
        """
        diag = {"duplicates_removed": 0, "conflicts_removed": 0}

        # Helper: deduplicate while preserving order
        def dedup(lst: list) -> int:
            before = len(lst)
            seen = set()
            result = []
            for x in lst:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
            lst[:] = result
            return before - len(lst)

        # Deduplicate all airport lists
        geo = self.geography
        for attr in ("acceptable_origin_hubs", "acceptable_destination_hubs",
                      "acceptable_transfer_hubs", "preferred_origin_hubs",
                      "preferred_transfer_hubs", "preferred_destination_hubs",
                      "preferred_airports", "avoid_airports"):
            lst = getattr(geo, attr, None)
            if isinstance(lst, list):
                diag["duplicates_removed"] += dedup(lst)

        # Also dedup airline lists
        for attr in ("preferred_airlines", "avoid_airlines"):
            lst = getattr(self.airline_preferences, attr, None)
            if isinstance(lst, list):
                diag["duplicates_removed"] += dedup(lst)

        # Remove avoid_airports from all acceptable/preferred lists
        avoid_set = set(geo.avoid_airports)
        for attr in ("acceptable_origin_hubs", "acceptable_destination_hubs",
                      "acceptable_transfer_hubs", "preferred_origin_hubs",
                      "preferred_transfer_hubs", "preferred_destination_hubs",
                      "preferred_airports"):
            lst = getattr(geo, attr, None)
            if isinstance(lst, list):
                before = len(lst)
                lst[:] = [x for x in lst if x not in avoid_set]
                diag["conflicts_removed"] += before - len(lst)

        return diag

    def summary_zh(self) -> str:
        parts = []
        o = self.trip.primary_origin_text or self.trip.primary_origin_airport or "?"
        d = self.trip.primary_destination_text or self.trip.primary_destination_airport or "?"
        parts.append(f"{o}→{d}")
        if self.geography.avoid_airports:
            parts.append(f"避开:{','.join(self.geography.avoid_airports[:3])}")
        parts.append(f"排序:{self.ranking_preferences.profile}")
        return " | ".join(parts)

    def to_sft_target(self) -> dict:
        return self.model_dump(mode="json")

    def to_downstream_context(self) -> dict:
        return {
            "origin": self.trip.primary_origin_text or self.trip.primary_origin_airport,
            "destination": self.trip.primary_destination_text or self.trip.primary_destination_airport,
            "cabin": self.cabin.cabin,
            "profile": self.ranking_preferences.profile,
            "avoid_airports": self.geography.avoid_airports,
            "acceptable_origin_hubs": self.geography.acceptable_origin_hubs,
            "risk": self.risk_preferences.risk_tolerance,
            "family": self.risk_preferences.family_friendly,
        }


class DecisionTraceItem(BaseModel):
    step: str = ""
    evidence: str = ""
    decision: str = ""
    affected_fields: list[str] = Field(default_factory=list)


class TravelRequirementContractUpdate(BaseModel):
    update_type: str = "modify_existing"
    field_updates: dict = Field(default_factory=dict)
    constraints_to_add: list[ConstraintItem] = Field(default_factory=list)
    constraints_to_remove: list[str] = Field(default_factory=list)
    preferences_to_add: list[ConstraintItem] = Field(default_factory=list)
    preferences_to_remove: list[str] = Field(default_factory=list)
    clarification_questions: list[ClarificationNeed] = Field(default_factory=list)
    should_search: bool = False
    should_rerun_search: bool = False
    should_rerank_only: bool = False
    should_explain_existing_result: bool = False
    selected_option_index: int | None = None
    user_facing_ack_zh: str = ""
    confidence: float = 0.5
    reasoning_summary: str = ""
    decision_trace: list[DecisionTraceItem] = Field(default_factory=list)
