"""Airport and hub-related models."""

from pydantic import BaseModel, Field


class Airport(BaseModel):
    """Reference data for an airport."""
    code: str = Field(description="IATA airport code, e.g. PVG")
    name: str
    city: str
    city_cn: str = ""
    country: str = "US"
    lat: float = 0.0
    lon: float = 0.0
    is_international_hub: bool = False
    is_domestic_only: bool = False
    preferred: bool = False  # Preferred airport for shared city aliases (e.g. PVG > SHA)


class NearbyHub(BaseModel):
    """A nearby hub airport that can serve as an alternate origin/destination."""
    airport: Airport
    access_mode: str = Field(description="e.g. train, car, domestic_flight, bus")
    access_time_hours: float = Field(description="Estimated travel time to hub")
    access_cost_usd: float = Field(description="Estimated cost to reach hub")
    hub_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How good a hub this is for the purpose")


class HubPair(BaseModel):
    """A candidate origin-hub to destination-hub pairing."""
    origin_hub_code: str
    destination_hub_code: str
    estimated_access_cost_usd: float = 0.0
    estimated_access_time_hours: float = 0.0
    split_ticket_required: bool = True
    expected_savings_potential: str = "unknown"  # "high", "medium", "low", "unknown"
    split_mode: str = "both_side"  # "origin_side", "destination_side", "both_side"
    reason: str = ""  # Human-readable explanation for this pair


class HubSplitPlan(BaseModel):
    """Output of the HubSplit agent — all candidate hub pairs for a query."""
    origin_airport_code: str
    destination_airport_code: str
    origin_hubs: list[NearbyHub] = Field(default_factory=list)
    destination_hubs: list[NearbyHub] = Field(default_factory=list)
    candidate_hub_pairs: list[HubPair] = Field(default_factory=list)
