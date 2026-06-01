"""Compiles TravelRequirementContract into pipeline constraints and exclusion rules."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.models.requirement_contract import TravelRequirementContract
from travel_agent.models.user_request import (
    HardConstraints, SoftConstraints, SearchConstraints, CabinClass,
)


CITY_TO_AIRPORTS: dict[str, list[str]] = {
    "shanghai": ["PVG", "SHA"], "上海": ["PVG", "SHA"],
    "new york": ["JFK", "EWR", "LGA"], "nyc": ["JFK", "EWR", "LGA"],
    "纽约": ["JFK", "EWR", "LGA"],
    "washington": ["IAD", "DCA", "BWI"], "dc": ["IAD", "DCA", "BWI"],
    "华盛顿": ["IAD", "DCA", "BWI"],
    "chicago": ["ORD", "MDW"], "芝加哥": ["ORD", "MDW"],
    "beijing": ["PEK", "PKX"], "北京": ["PEK", "PKX"],
    "los angeles": ["LAX"], "la": ["LAX"], "洛杉矶": ["LAX"],
    "san francisco": ["SFO"], "sf": ["SFO"], "旧金山": ["SFO"],
    "seattle": ["SEA"], "西雅图": ["SEA"],
    "boston": ["BOS"], "波士顿": ["BOS"],
    "toronto": ["YYZ"], "多伦多": ["YYZ"],
}


class ExclusionRules(BaseModel):
    excluded_airports: list[str] = Field(default_factory=list)
    excluded_cities: list[str] = Field(default_factory=list)
    excluded_airlines: list[str] = Field(default_factory=list)
    excluded_route_patterns: list[str] = Field(default_factory=list)
    reason_by_code: dict[str, str] = Field(default_factory=dict)

    def is_airport_excluded(self, code: str) -> bool:
        return code.upper() in {a.upper() for a in self.excluded_airports}

    def is_city_excluded(self, city: str) -> bool:
        return city.lower() in {c.lower() for c in self.excluded_cities}

    def is_airline_excluded(self, code: str) -> bool:
        return code.upper() in {a.upper() for a in self.excluded_airlines}

    def route_contains_exclusion(self, route_codes: list[str]) -> bool:
        return any(self.is_airport_excluded(c) for c in route_codes)

    def expand_cities(self) -> None:
        """Expand excluded cities into excluded airports."""
        for city in list(self.excluded_cities):
            airports = CITY_TO_AIRPORTS.get(city.lower())
            if airports:
                for a in airports:
                    if a not in self.excluded_airports:
                        self.excluded_airports.append(a)
                        self.reason_by_code[a] = f"城市 {city} 被用户排除"

    def explain_exclusion(self, code: str) -> str:
        return self.reason_by_code.get(code.upper(), f"{code} 被排除")


class CompiledConstraints(BaseModel):
    search: SearchConstraints
    exclusions: ExclusionRules
    profile: str = "balanced"


class ConstraintCompiler:
    """Compile a TravelRequirementContract into executable pipeline constraints."""

    CABIN_MAP = {
        "economy": CabinClass.ECONOMY, "premium_economy": CabinClass.PREMIUM_ECONOMY,
        "business": CabinClass.BUSINESS, "first": CabinClass.FIRST,
    }

    def compile(self, contract: TravelRequirementContract) -> CompiledConstraints:
        trip = contract.trip
        hub = contract.hub_preferences
        tick = contract.ticketing_preferences
        rank = contract.ranking_preferences
        air = contract.airline_preferences
        geo = contract.geography
        risk = contract.risk_preferences
        budget = contract.budget_preferences

        hard = HardConstraints(
            origin_airport_codes=[trip.primary_origin_airport] if trip.primary_origin_airport else [],
            destination_airport_codes=[trip.primary_destination_airport] if trip.primary_destination_airport else [],
            passengers=contract.passengers.passenger_count,
            cabin=self.CABIN_MAP.get(contract.cabin.cabin, CabinClass.ECONOMY),
        )

        soft = SoftConstraints(
            prefer_lowest_price=(rank.price_priority == "high" or budget.save_money_priority == "high"),
            prefer_fewer_stops=(risk.risk_tolerance == "low" or risk.avoid_short_connection),
            prefer_comfort=(rank.comfort_priority == "high"),
            prefer_low_risk=(risk.risk_tolerance == "low" or risk.family_friendly),
            accept_nearby_hubs=(hub.nearby_hub_policy != "avoid"),
            accept_split_ticket=(tick.split_ticket_policy != "avoid"),
        )

        exclusions = ExclusionRules(
            excluded_airports=list(geo.avoid_airports),
            excluded_cities=list(geo.avoid_cities),
            excluded_airlines=list(air.avoid_airlines),
            reason_by_code={a: "用户硬约束" for a in geo.avoid_airports},
        )

        # Expand cities to airports
        exclusions.expand_cities()

        if hub.nearby_hub_policy == "avoid":
            exclusions.excluded_airports.extend(geo.acceptable_origin_hubs)

        return CompiledConstraints(
            search=SearchConstraints(hard=hard, soft=soft),
            exclusions=exclusions,
            profile=rank.profile,
        )
