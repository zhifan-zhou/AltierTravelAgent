"""Single source-of-truth travel requirement contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from travel_agent.contract.normalization import (
    CITY_AIRPORTS,
    dedupe,
    expand_location,
    normalize_airport_list,
    normalize_code,
)


class Trip(BaseModel):
    origin_text: str | None = None
    origin_airport: str | None = None
    origin_city: str | None = None
    destination_text: str | None = None
    destination_airport: str | None = None
    destination_city: str | None = None
    trip_type: Literal["one_way", "round_trip", "unknown"] = "unknown"
    route_locked: bool = False


class TimeWindow(BaseModel):
    departure_text: str | None = None
    departure_window_text: str | None = None
    departure_start_date: str | None = None
    departure_end_date: str | None = None
    flexible: bool = True
    flexible_date_confirmed: bool = False


class Passengers(BaseModel):
    passenger_count: int = 1
    family_or_parents: bool = False
    student_travel: bool = False
    baggage_count: int | None = None
    baggage_heavy: bool = False


class Cabin(BaseModel):
    cabin: Literal["economy", "premium_economy", "business", "first", "any"] = "economy"
    flexible: bool = True


class Geography(BaseModel):
    acceptable_origin_hubs: list[str] = Field(default_factory=list)
    acceptable_transfer_hubs: list[str] = Field(default_factory=list)
    acceptable_destination_hubs: list[str] = Field(default_factory=list)
    preferred_airports: list[str] = Field(default_factory=list)
    avoid_airports: list[str] = Field(default_factory=list)
    avoid_cities: list[str] = Field(default_factory=list)
    unresolved_locations: list[str] = Field(default_factory=list)


class HubPolicy(BaseModel):
    nearby_hub_policy: Literal["allow", "prefer", "avoid"] = "allow"
    allow_ground_access: bool = True
    max_ground_access_hours: float | None = None
    avoid_complex_transfers: bool = False


class Ticketing(BaseModel):
    split_ticket_policy: Literal["allow", "prefer", "avoid"] = "allow"
    allow_self_transfer: bool = True
    allow_overnight: bool = False
    require_protected_connection: bool = False


class Ranking(BaseModel):
    profile: Literal["balanced", "cheapest", "airline_priority", "low_risk", "fastest"] = "balanced"
    price_priority: Literal["low", "medium", "high"] = "medium"
    risk_priority: Literal["low", "medium", "high"] = "medium"
    time_priority: Literal["low", "medium", "high"] = "medium"
    airline_quality_priority: Literal["low", "medium", "high"] = "medium"


class AirlinePreferences(BaseModel):
    prefer_major_airlines: bool = False
    avoid_low_cost_carriers: bool = False
    preferred_airlines: list[str] = Field(default_factory=list)
    avoid_airlines: list[str] = Field(default_factory=list)


class ConstraintItem(BaseModel):
    type: Literal[
        "avoid_airport",
        "avoid_city",
        "avoid_airline",
        "no_split_ticket",
        "max_budget",
        "other",
    ] = "other"
    value: Any = None
    normalized_values: list[str] = Field(default_factory=list)
    reason: str = ""
    source_user_message: str = ""
    active: bool = True


class PreferenceItem(BaseModel):
    type: Literal[
        "prefer_hub",
        "prefer_airline",
        "prefer_low_price",
        "prefer_low_risk",
        "prefer_short_time",
        "other",
    ] = "other"
    value: Any = None
    normalized_values: list[str] = Field(default_factory=list)
    weight_hint: Literal["low", "medium", "high"] = "medium"
    source_user_message: str = ""
    active: bool = True


class SpecialRequirement(BaseModel):
    category: str = "unknown"
    description_zh: str = ""
    structured_values: dict[str, Any] = Field(default_factory=dict)
    impact_areas: list[str] = Field(default_factory=list)
    hard_constraint: bool = False
    preference_weight: Literal["low", "medium", "high"] = "medium"
    requires_clarification: bool = False
    clarification_question_zh: str | None = None
    source_user_message: str = ""
    active: bool = True


class Constraints(BaseModel):
    hard_constraints: list[ConstraintItem] = Field(default_factory=list)
    soft_preferences: list[PreferenceItem] = Field(default_factory=list)


class Assumption(BaseModel):
    field: str
    value: Any = None
    reason: str = ""
    can_user_override: bool = True


class Metadata(BaseModel):
    original_user_goal: str | None = None
    last_user_message: str | None = None
    update_count: int = 0
    current_profile: str = "balanced"
    history_summary: str = ""


class TravelRequirementContract(BaseModel):
    schema_version: str = "v1"
    trip: Trip = Field(default_factory=Trip)
    time: TimeWindow = Field(default_factory=TimeWindow)
    passengers: Passengers = Field(default_factory=Passengers)
    cabin: Cabin = Field(default_factory=Cabin)
    geography: Geography = Field(default_factory=Geography)
    hub_policy: HubPolicy = Field(default_factory=HubPolicy)
    ticketing: Ticketing = Field(default_factory=Ticketing)
    ranking: Ranking = Field(default_factory=Ranking)
    airline_preferences: AirlinePreferences = Field(default_factory=AirlinePreferences)
    constraints: Constraints = Field(default_factory=Constraints)
    special_requirements: list[SpecialRequirement] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    ready_to_search: bool = False
    confidence: float = 0.0
    metadata: Metadata = Field(default_factory=Metadata)

    def has_required_route(self) -> bool:
        return bool(self.trip.origin_airport and self.trip.destination_airport)

    def has_searchable_time(self) -> bool:
        return bool(
            self.time.departure_start_date
            or self.time.departure_end_date
            or self.time.departure_text
            or self.time.departure_window_text
            or self.time.flexible_date_confirmed
        )

    def missing_mandatory_search_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.trip.origin_airport:
            missing.append("trip.origin_airport")
        if not self.trip.destination_airport:
            missing.append("trip.destination_airport")
        if not self.has_searchable_time():
            missing.append("time.departure_window")
        return missing

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, value: str) -> str:
        return value or "v1"

    def normalize(self) -> "TravelRequirementContract":
        self.constraints.hard_constraints = _coerce_model_list_silent(
            self.constraints.hard_constraints, ConstraintItem
        )
        self.constraints.soft_preferences = _coerce_model_list_silent(
            self.constraints.soft_preferences, PreferenceItem
        )
        self.special_requirements = _coerce_model_list_silent(
            self.special_requirements, SpecialRequirement
        )

        self.trip.origin_airport = normalize_code(self.trip.origin_airport or self.trip.origin_text)
        self.trip.destination_airport = normalize_code(
            self.trip.destination_airport or self.trip.destination_text
        )
        self.geography.acceptable_origin_hubs = normalize_airport_list(
            self.geography.acceptable_origin_hubs
        )
        self.geography.acceptable_transfer_hubs = normalize_airport_list(
            self.geography.acceptable_transfer_hubs
        )
        self.geography.acceptable_destination_hubs = normalize_airport_list(
            self.geography.acceptable_destination_hubs
        )
        self.geography.preferred_airports = normalize_airport_list(self.geography.preferred_airports)
        self.geography.avoid_airports = normalize_airport_list(self.geography.avoid_airports)
        self.geography.avoid_cities = dedupe(self.geography.avoid_cities)
        self.geography.unresolved_locations = dedupe(self.geography.unresolved_locations)

        city_exclusions: list[str] = []
        for city in self.geography.avoid_cities:
            city_exclusions.extend(expand_location(city))
        self.geography.avoid_airports = normalize_airport_list(
            [*self.geography.avoid_airports, *city_exclusions]
        )

        for constraint in self.constraints.hard_constraints:
            if constraint.type == "avoid_airport":
                constraint.normalized_values = normalize_airport_list(
                    constraint.normalized_values or [str(constraint.value)]
                )
            elif constraint.type == "avoid_city":
                constraint.normalized_values = normalize_airport_list(
                    constraint.normalized_values or expand_location(str(constraint.value))
                )
                self.geography.avoid_airports = normalize_airport_list(
                    [*self.geography.avoid_airports, *constraint.normalized_values]
                )
            elif constraint.type == "avoid_airline":
                constraint.normalized_values = [
                    str(v).upper() for v in (constraint.normalized_values or [constraint.value]) if v
                ]
        for preference in self.constraints.soft_preferences:
            if preference.type in {"prefer_hub", "prefer_airline"}:
                values = preference.normalized_values or [preference.value]
                preference.normalized_values = [
                    str(v).upper() for v in values if v is not None and str(v).strip()
                ]
        self.special_requirements = _dedupe_special_requirements(self.special_requirements)

        return self.remove_conflicts()

    def remove_conflicts(self) -> "TravelRequirementContract":
        excluded = set(self.geography.avoid_airports)
        self.geography.acceptable_origin_hubs = [
            code for code in self.geography.acceptable_origin_hubs if code not in excluded
        ]
        self.geography.acceptable_transfer_hubs = [
            code for code in self.geography.acceptable_transfer_hubs if code not in excluded
        ]
        self.geography.acceptable_destination_hubs = [
            code for code in self.geography.acceptable_destination_hubs if code not in excluded
        ]
        self.geography.preferred_airports = [
            code for code in self.geography.preferred_airports if code not in excluded
        ]
        self.airline_preferences.preferred_airlines = dedupe(
            [a.upper() for a in self.airline_preferences.preferred_airlines]
        )
        self.airline_preferences.avoid_airlines = dedupe(
            [a.upper() for a in self.airline_preferences.avoid_airlines]
        )
        avoided_airlines = set(self.airline_preferences.avoid_airlines)
        self.airline_preferences.preferred_airlines = [
            a for a in self.airline_preferences.preferred_airlines if a not in avoided_airlines
        ]
        self.metadata.current_profile = self.ranking.profile
        return self

    def summary_zh(self) -> str:
        origin = self.trip.origin_airport or self.trip.origin_text or "未定"
        dest = self.trip.destination_airport or self.trip.destination_text or "未定"
        hubs = "/".join(self.geography.acceptable_origin_hubs) or "自动"
        avoids = "/".join(self.geography.avoid_airports) or "无"
        time = self.time.departure_window_text or self.time.departure_text or ("灵活日期" if self.time.flexible_date_confirmed else "未定")
        return f"{origin} 到 {dest}；时间：{time}；出发枢纽：{hubs}；排序：{self.ranking.profile}；避开：{avoids}"

    def to_sft_target(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def airports_for_city_name(city: str) -> list[str]:
    return CITY_AIRPORTS.get(city, CITY_AIRPORTS.get(city.lower(), []))


def _coerce_model_list_silent(items: list[Any], model_cls: type[BaseModel]) -> list[Any]:
    result: list[Any] = []
    for item in items or []:
        if isinstance(item, model_cls):
            result.append(item)
            continue
        if isinstance(item, dict):
            try:
                result.append(model_cls.model_validate(item))
            except ValidationError:
                continue
    return result


def _dedupe_special_requirements(items: list[SpecialRequirement]) -> list[SpecialRequirement]:
    result: list[SpecialRequirement] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            item.category.strip().lower(),
            item.description_zh.strip(),
            item.source_user_message.strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        item.impact_areas = dedupe([str(area).strip() for area in item.impact_areas if str(area).strip()])
        result.append(item)
    return result
