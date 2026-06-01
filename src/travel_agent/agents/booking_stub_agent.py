"""Booking Stub Agent: MVP placeholder for booking flow."""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.agents.base import BaseAgent
from travel_agent.models.itinerary import Itinerary


class BookingStubOutput(BaseModel):
    """Booking stub result for MVP."""
    itinerary_id: str
    booking_status: str = "not_implemented_mvp"
    next_steps: str = "Connect Duffel/Amadeus booking API for real booking."
    required_user_data: list[str] = Field(
        default_factory=lambda: ["passport", "passenger_info", "payment_token"]
    )
    price_verified: bool = False
    message: str = "MVP 阶段不实现真实出票。以上价格为模拟数据，请连接真实 API 后完成出票。"


class BookingStubAgent(BaseAgent[Itinerary, BookingStubOutput]):
    """MVP booking stub — does NOT perform real booking."""

    name = "booking_stub"

    async def execute(self, data: Itinerary) -> BookingStubOutput:
        return BookingStubOutput(
            itinerary_id=data.id,
            booking_status="not_implemented_mvp",
            next_steps="Connect Duffel/Amadeus booking API for real booking.",
            required_user_data=["passport", "passenger_info", "payment_token"],
            price_verified=False,
            message="MVP 阶段不实现真实出票。以上价格为模拟数据，请连接真实 API 后完成出票。",
        )
