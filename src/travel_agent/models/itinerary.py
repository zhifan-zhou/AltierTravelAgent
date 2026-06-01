"""Itinerary models — composed routes from flight offers."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from travel_agent.models.flight import FlightOffer, FlightSegment


class Itinerary(BaseModel):
    """A complete travel itinerary composed of one or more flight offers."""
    id: str = Field(description="Unique itinerary ID")
    type: str = Field(default="direct", description="direct, hub_split, multi_airline")
    segments: list[FlightSegment] = Field(default_factory=list)
    offers: list[FlightOffer] = Field(default_factory=list)

    total_price_usd: float = 0.0
    total_access_cost_usd: float = 0.0
    total_estimated_time_hours: float = 0.0
    number_of_segments: int = 0
    split_ticket_count: int = 0

    origin_airport: str = ""
    destination_airport: str = ""
    main_international_leg: Optional[str] = None

    risk_level: str = "unknown"
    warnings: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.now)


class RouteComposerOutput(BaseModel):
    """Output of the Route Composer agent."""
    itineraries: list[Itinerary] = Field(default_factory=list)
    baseline_itinerary_id: Optional[str] = None
