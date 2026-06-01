"""Compile requirement contracts into deterministic pipeline constraints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.contract.normalization import expand_location, normalize_airport_list
from travel_agent.contract.special_requirements import SpecialRequirementInterpreter


class ExclusionRules(BaseModel):
    excluded_airports: list[str] = Field(default_factory=list)
    excluded_cities: list[str] = Field(default_factory=list)
    excluded_airlines: list[str] = Field(default_factory=list)
    reason_by_code: dict[str, str] = Field(default_factory=dict)

    def airport_is_excluded(self, code: str | None) -> bool:
        return bool(code) and code.upper() in set(self.excluded_airports)

    def airline_is_excluded(self, code: str | None) -> bool:
        return bool(code) and code.upper() in set(self.excluded_airlines)


class SearchConstraints(BaseModel):
    origin_airport: str
    destination_airport: str
    origin_hubs: list[str] = Field(default_factory=list)
    transfer_hubs: list[str] = Field(default_factory=list)
    destination_hubs: list[str] = Field(default_factory=list)
    cabin: str = "economy"
    nearby_hub_policy: str = "allow"
    split_ticket_policy: str = "allow"
    allow_self_transfer: bool = True
    allow_ground_access: bool = True
    profile: str = "balanced"
    ranking: dict[str, str] = Field(default_factory=dict)
    exclusions: ExclusionRules = Field(default_factory=ExclusionRules)


class ConstraintCompiler:
    def __init__(self, special_interpreter: SpecialRequirementInterpreter | None = None):
        self.specials = special_interpreter or SpecialRequirementInterpreter()

    def compile(self, contract: TravelRequirementContract) -> SearchConstraints:
        contract.normalize()
        special_effects = self.specials.interpret(contract.special_requirements)
        reason_by_code: dict[str, str] = {}
        excluded_airports = normalize_airport_list(contract.geography.avoid_airports)
        excluded_cities = list(contract.geography.avoid_cities)
        for city in excluded_cities:
            for code in expand_location(city):
                reason_by_code[code] = f"用户要求避开{city}"
                excluded_airports.append(code)

        excluded_airlines = [a.upper() for a in contract.airline_preferences.avoid_airlines]
        for item in contract.constraints.hard_constraints:
            if not item.active:
                continue
            if item.type in {"avoid_airport", "avoid_city"}:
                for code in normalize_airport_list(item.normalized_values or [str(item.value)]):
                    excluded_airports.append(code)
                    reason_by_code[code] = item.reason or "用户硬性限制"
            elif item.type == "avoid_airline":
                excluded_airlines.extend(str(v).upper() for v in (item.normalized_values or [item.value]) if v)

        excluded_airports = normalize_airport_list(excluded_airports)
        excluded_airlines = sorted(set(excluded_airlines))

        def allowed(values: list[str]) -> list[str]:
            blocked = set(excluded_airports)
            return [code for code in normalize_airport_list(values) if code not in blocked]

        return SearchConstraints(
            origin_airport=contract.trip.origin_airport or "",
            destination_airport=contract.trip.destination_airport or "",
            origin_hubs=allowed(contract.geography.acceptable_origin_hubs),
            transfer_hubs=allowed(contract.geography.acceptable_transfer_hubs),
            destination_hubs=allowed(contract.geography.acceptable_destination_hubs),
            cabin=contract.cabin.cabin,
            nearby_hub_policy=contract.hub_policy.nearby_hub_policy,
            split_ticket_policy=contract.ticketing.split_ticket_policy,
            allow_self_transfer=contract.ticketing.allow_self_transfer and not special_effects.avoid_self_transfer,
            allow_ground_access=contract.hub_policy.allow_ground_access,
            profile=contract.ranking.profile,
            ranking=contract.ranking.model_dump(mode="json"),
            exclusions=ExclusionRules(
                excluded_airports=excluded_airports,
                excluded_cities=excluded_cities,
                excluded_airlines=excluded_airlines,
                reason_by_code=reason_by_code,
            ),
        )
