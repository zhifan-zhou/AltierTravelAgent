"""Flight-related models: offers, segments, search requests."""

from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from travel_agent.models.user_request import CabinClass


class FlightSegment(BaseModel):
    """A single flight leg within an offer."""
    origin: str = Field(description="IATA origin code")
    destination: str = Field(description="IATA destination code")
    departure_time: datetime
    arrival_time: datetime
    airline: str = ""
    airline_name: str = ""
    flight_number: str = ""
    cabin: CabinClass = CabinClass.ECONOMY
    aircraft: str = ""


class FlightOffer(BaseModel):
    """A priced flight offer with data provenance tracking."""
    id: str = Field(description="Unique offer identifier from provider")
    segments: list[FlightSegment] = Field(default_factory=list)
    total_price_usd: float = 0.0
    currency: str = "USD"

    # Provider identity
    provider_name: str = Field(default="mock", description="Provider name: mock, duffel, serpapi, etc.")
    source: str = Field(default="mock_fallback", description="Data provenance tag")
    is_real: bool = Field(default=False, description="Whether this offer came from a real API")
    confidence: str = Field(default="estimated", description="verified | demo | estimated")

    # Timestamps
    last_verified_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now() + timedelta(hours=1),
        description="When this offer's price may no longer be valid"
    )

    # Booking
    booking_available: bool = False
    baggage_included: bool = True
    refundable: bool = False

    # Raw provider payload (optional, for debugging)
    raw_provider_payload: Optional[dict] = Field(default=None, description="Original provider response")

    @property
    def data_quality(self) -> str:
        """Overall data quality label for this offer."""
        if self.confidence == "verified" and self.is_real:
            return "verified"
        if self.is_real:
            return "real"
        if self.source == "mock_exact":
            return "demo_exact"
        return "demo_estimated"


class FlightSearchRequest(BaseModel):
    """Request to search for flights."""
    origin: str = Field(description="IATA origin code")
    destination: str = Field(description="IATA destination code")
    departure_date: Optional[datetime] = None
    return_date: Optional[datetime] = None
    cabin: CabinClass = CabinClass.ECONOMY
    passengers: int = 1
    flexible_dates: bool = False
    date_window_days: int = 3
