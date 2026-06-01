"""Core data models for the Travel Agent system."""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CabinClass(str, Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class DateWindow(BaseModel):
    """Flexible date range for travel search."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    flexible: bool = True


class UserTravelRequest(BaseModel):
    """Structured representation of a user's travel intent."""
    origin_text: str = Field(description="User's stated origin city/airport")
    destination_text: str = Field(description="User's stated destination city/airport")
    departure_window: DateWindow = Field(default_factory=DateWindow)
    return_window: Optional[DateWindow] = None
    cabin: CabinClass = CabinClass.ECONOMY
    passengers: int = Field(default=1, ge=1, le=9)
    budget_usd: Optional[float] = None
    accepts_nearby_hubs: bool = False
    accepts_split_ticket: bool = False
    preferences: list[str] = Field(default_factory=list)
    raw_query: str = ""


class HardConstraints(BaseModel):
    """Non-negotiable search constraints."""
    origin_airport_codes: list[str] = Field(default_factory=list)
    destination_airport_codes: list[str] = Field(default_factory=list)
    departure_date_start: Optional[date] = None
    departure_date_end: Optional[date] = None
    passengers: int = 1
    cabin: CabinClass = CabinClass.ECONOMY
    max_budget_usd: Optional[float] = None


class SoftConstraints(BaseModel):
    """Preference-based constraints that affect scoring."""
    prefer_lowest_price: bool = True
    prefer_fewer_stops: bool = False
    prefer_comfort: bool = False
    prefer_low_risk: bool = False
    accept_nearby_hubs: bool = False
    accept_split_ticket: bool = False
    max_access_time_hours: float = 6.0
    max_layover_hours: float = 8.0


class SearchConstraints(BaseModel):
    """Combined hard and soft constraints for search."""
    hard: HardConstraints = Field(default_factory=HardConstraints)
    soft: SoftConstraints = Field(default_factory=SoftConstraints)
