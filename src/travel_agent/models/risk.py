"""Risk assessment models for itineraries."""

from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    """Risk evaluation for a single itinerary."""
    risk_level: str = Field(default="low", description="low, medium, high")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    split_ticket_risk: bool = False
    short_connection_risk: bool = False
    baggage_recheck_risk: bool = False
    overnight_stay_risk: bool = False
    airport_transfer_risk: bool = False
    visa_entry_risk: bool = False
    hidden_city_risk: bool = False
    price_expiration_risk: bool = False
    details: dict = Field(default_factory=dict)
