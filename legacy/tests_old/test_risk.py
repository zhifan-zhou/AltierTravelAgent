"""Test the Risk Assessment Service & Agent."""

from datetime import datetime, timedelta

import pytest

from travel_agent.models.itinerary import Itinerary
from travel_agent.models.flight import FlightOffer, FlightSegment
from travel_agent.models.user_request import CabinClass
from travel_agent.agents.risk_compliance_agent import RiskComplianceAgent
from travel_agent.models.agent_outputs import RouteComposerResult
from travel_agent.models.itinerary import RouteComposerOutput


class TestRiskAssessment:
    def test_split_ticket_gets_medium_risk(self, risk_service):
        segs = [
            FlightSegment(
                origin="PVG", destination="JFK",
                departure_time=datetime(2026, 6, 15, 17, 0),
                arrival_time=datetime(2026, 6, 15, 20, 0),
                airline="MU", flight_number="MU587", cabin=CabinClass.ECONOMY,
            ),
            FlightSegment(
                origin="JFK", destination="PIT",
                departure_time=datetime(2026, 6, 16, 8, 0),
                arrival_time=datetime(2026, 6, 16, 9, 30),
                airline="DL", flight_number="DL5123", cabin=CabinClass.ECONOMY,
            ),
        ]
        offer = FlightOffer(
            id="mock-001", segments=segs, total_price_usd=1000,
            provider_name="mock", source="test",
        )
        it = Itinerary(
            id="test-split",
            type="hub_split",
            segments=segs,
            offers=[offer],
            total_price_usd=1000,
            split_ticket_count=2,
        )

        assessment = risk_service.assess(it)

        # Split ticket + mock should be medium risk (0.55 ≤ 0.60)
        assert assessment.risk_level == "medium", f"Expected medium, got {assessment.risk_level} (score={assessment.risk_score})"
        assert assessment.split_ticket_risk is True
        assert assessment.baggage_recheck_risk is True
        assert len(assessment.warnings) >= 2

    def test_direct_route_is_low_risk(self, risk_service):
        seg = FlightSegment(
            origin="NGB", destination="PIT",
            departure_time=datetime(2026, 6, 15, 8, 0),
            arrival_time=datetime(2026, 6, 15, 22, 0),
            airline="CA", flight_number="CA999", cabin=CabinClass.ECONOMY,
        )
        offer = FlightOffer(
            id="mock-direct", segments=[seg], total_price_usd=1850,
            provider_name="mock", source="test",
        )
        it = Itinerary(
            id="test-direct",
            type="direct",
            segments=[seg],
            offers=[offer],
            total_price_usd=1850,
            split_ticket_count=0,
        )

        assessment = risk_service.assess(it)

        assert assessment.risk_level == "low"
        assert assessment.split_ticket_risk is False

    def test_risk_compliance_agent(self, risk_service):
        agent = RiskComplianceAgent(risk_service=risk_service)

        segs = [
            FlightSegment(
                origin="PVG", destination="JFK",
                departure_time=datetime(2026, 6, 15, 17, 0),
                arrival_time=datetime(2026, 6, 15, 20, 0),
                airline="MU", flight_number="MU587", cabin=CabinClass.ECONOMY,
            ),
            FlightSegment(
                origin="JFK", destination="PIT",
                departure_time=datetime(2026, 6, 16, 8, 0),
                arrival_time=datetime(2026, 6, 16, 9, 30),
                airline="DL", flight_number="DL5123", cabin=CabinClass.ECONOMY,
            ),
        ]
        offer = FlightOffer(id="mock-001", segments=segs, total_price_usd=1000, provider_name="mock", source="test")
        it = Itinerary(
            id="test", type="hub_split", segments=segs, offers=[offer],
            total_price_usd=1000, split_ticket_count=2,
        )

        route_output = RouteComposerResult(
            output=RouteComposerOutput(itineraries=[it])
        )

        import asyncio
        result = asyncio.run(agent.execute(route_output))

        assert "test" in result.assessments
        assert result.assessments["test"].risk_level in ("medium", "high")
