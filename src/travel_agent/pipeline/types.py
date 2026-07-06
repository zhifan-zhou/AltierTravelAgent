"""Shared deterministic pipeline models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.contract.compiler import ExclusionRules
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.schemas import TravelRequirementContractUpdate


class HubCandidatePair(BaseModel):
    pair_id: str
    origin_airport: str
    origin_hub: str
    destination_hub: str
    destination_airport: str
    origin_access_cost_usd: float = 0
    destination_access_cost_usd: float = 0
    expected_savings_potential: str = "medium"


class SearchTask(BaseModel):
    task_id: str
    pair_id: str | None = None
    leg_type: str
    origin: str
    destination: str
    cabin: str = "economy"


class FlightSegment(BaseModel):
    origin: str
    destination: str
    airline: str | None = None
    airline_name: str | None = None
    flight_number: str | None = None
    mode: str = "flight"


class FlightOffer(BaseModel):
    id: str
    task_id: str | None = None
    leg_type: str
    origin: str
    destination: str
    segments: list[FlightSegment]
    total_price_usd: float
    estimated_time_hours: float = 0.0
    source: str = "mock"
    confidence: str = "known"
    booking_available: bool = False
    data_source: str = "mock_demo"
    is_real_price: bool = False
    bookable: bool = False

    @property
    def has_estimated_data(self) -> bool:
        return self.source == "mock_fallback" or self.confidence == "estimated"


class Itinerary(BaseModel):
    id: str
    route_type: str
    route: list[str]
    offers: list[FlightOffer]
    segments: list[FlightSegment]
    total_price_usd: float
    total_estimated_time_hours: float
    source: str = "mock"
    confidence: str = "known"

    @property
    def airlines(self) -> list[str]:
        result: list[str] = []
        for segment in self.segments:
            if segment.airline and segment.airline not in {"GROUND", "MOCK"} and segment.airline not in result:
                result.append(segment.airline)
        return result

    @property
    def has_estimated_data(self) -> bool:
        return self.confidence == "estimated" or any(offer.has_estimated_data for offer in self.offers)

    @property
    def data_quality_label(self) -> str:
        return "mock_fallback/estimated" if self.has_estimated_data else "mock"


class RiskAssessment(BaseModel):
    risk_score: float
    risk_level: str
    warnings: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    rank: int
    recommendation_type: str
    itinerary: Itinerary
    score: float
    savings_vs_baseline_usd: float
    risk: RiskAssessment
    airline_quality_score: float
    reason_zh: str


class PipelineResult(BaseModel):
    contract: TravelRequirementContract
    exclusions: ExclusionRules
    hub_pairs: list[HubCandidatePair] = Field(default_factory=list)
    search_tasks: list[SearchTask] = Field(default_factory=list)
    provider_calls: list[SearchTask] = Field(default_factory=list)
    offers: list[FlightOffer] = Field(default_factory=list)
    itineraries: list[Itinerary] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider_call_count: int = 0


class ChatTurnResult(BaseModel):
    ok: bool
    message: str
    update: TravelRequirementContractUpdate | None = None
    contract: TravelRequirementContract | None = None
    pipeline_result: PipelineResult | None = None
    full_search_ran: bool = False
    rerank_only: bool = False
    provider_call_count: int = 0
    export_dir: str | None = None
    debug_summary: str = ""
    tool_results: list[dict] = Field(default_factory=list)
