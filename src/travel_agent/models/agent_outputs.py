"""Agent output models — structured outputs for each agent in the pipeline."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from travel_agent.models.airport import HubSplitPlan
from travel_agent.models.constraints import SearchConstraints
from travel_agent.models.flight import FlightOffer, FlightSearchRequest
from travel_agent.models.itinerary import Itinerary, RouteComposerOutput
from travel_agent.models.ranking import RankingOutput
from travel_agent.models.risk import RiskAssessment
from travel_agent.models.user_request import UserTravelRequest


class IntakeOutput(UserTravelRequest):
    """Intake Agent output — structured user request."""
    pass


class ConstraintOutput(BaseModel):
    """Constraint Agent output."""
    constraints: SearchConstraints
    original_request: UserTravelRequest


class HubSplitOutput(BaseModel):
    """HubSplit Agent output."""
    plan: HubSplitPlan
    search_tasks_generated: int = 0


class SearchStrategyOutput(BaseModel):
    """Search Strategy Agent output."""
    search_tasks: list[FlightSearchRequest] = Field(default_factory=list)
    direct_task: FlightSearchRequest | None = None
    hub_split_tasks: list[FlightSearchRequest] = Field(default_factory=list)


class FlightRetrievalOutput(BaseModel):
    """Flight Retrieval Agent output."""
    direct_offers: list[FlightOffer] = Field(default_factory=list)
    hub_split_offers: list[FlightOffer] = Field(default_factory=list)
    domestic_offers: list[FlightOffer] = Field(default_factory=list)
    all_offers: list[FlightOffer] = Field(default_factory=list)


class RouteComposerResult(BaseModel):
    """Wrapper for Route Composer output."""
    output: RouteComposerOutput


class RiskOutput(BaseModel):
    """Risk & Compliance Agent output."""
    assessments: dict[str, RiskAssessment] = Field(default_factory=dict)


class ExplanationOutput(BaseModel):
    """Explanation Agent output."""
    summary_zh: str = ""
    summary_en: str = ""
    per_itinerary_explanation: dict[str, str] = Field(default_factory=dict)
    not_recommended_explanations: list[str] = Field(default_factory=list)


class TravelAgentResult(BaseModel):
    """Final output of the entire Travel Agent pipeline."""
    query: str
    intake: Optional[IntakeOutput] = None
    constraints: Optional[ConstraintOutput] = None
    hub_split: Optional[HubSplitOutput] = None
    search_strategy: Optional[SearchStrategyOutput] = None
    flight_retrieval: Optional[FlightRetrievalOutput] = None
    route_composer: Optional[RouteComposerResult] = None
    risk: Optional[RiskOutput] = None
    ranking: Optional[RankingOutput] = None
    explanation: Optional[ExplanationOutput] = None
    error: Optional[str] = None
    debug_artifacts: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
