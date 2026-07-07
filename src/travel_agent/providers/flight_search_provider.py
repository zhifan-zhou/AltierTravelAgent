"""Safe provider metadata for future flight-search integrations.

v0.4 still uses mock/demo flight data only. This module documents the boundary
future providers must preserve: search metadata can be surfaced to the UI, but
booking, payment, ticketing, passenger document collection, and price locks are
outside the product prototype.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.pipeline.types import PipelineResult


class FlightProviderDisclosure(BaseModel):
    provider_name: str
    data_source: str = "mock_demo"
    is_real_price: bool = False
    bookable: bool = False
    safety_label: str = "Demo/mock only. Not real price, not bookable."


class FlightSearchProvider(Protocol):
    disclosure: FlightProviderDisclosure

    def search(self, contract: TravelRequirementContract) -> PipelineResult:
        """Return flight-search results without booking or payment side effects."""
        ...


MOCK_FLIGHT_DISCLOSURE = FlightProviderDisclosure(provider_name="mock")
