"""Phase 7 tests: itinerary display, airline service, ranking, CLI UX."""

import asyncio
from datetime import datetime

import pytest

from travel_agent.models.flight import FlightSegment, FlightOffer, CabinClass
from travel_agent.models.itinerary import Itinerary
from travel_agent.services.itinerary_display_service import ItineraryDisplayService
from travel_agent.services.airline_service import AirlineService
from travel_agent.services.scoring_service import ScoringService
from travel_agent.providers.mock_flight_provider import MockFlightProvider


# ── ItineraryDisplayService ──────────────────────────────────────────

class TestItineraryDisplay:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_format_route_codes_never_returns_only_id(self, display):
        it = Itinerary(
            id="test-123", type="direct",
            segments=[FlightSegment(origin="WNZ", destination="PIT",
                                     departure_time=datetime(2026,6,15,8,0),
                                     arrival_time=datetime(2026,6,15,20,0))],
            origin_airport="WNZ", destination_airport="PIT",
        )
        result = display.format_route_codes(it)
        assert "WNZ" in result
        assert "PIT" in result
        assert "test-123" not in result

    def test_format_route_human_zh_with_multi_leg(self, display):
        """WNZ→PVG→JFK→PIT should display as Chinese city names."""
        it = Itinerary(
            id="split-1", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
                FlightSegment(origin="JFK", destination="PIT",
                              departure_time=datetime(2026,6,16,8,0),
                              arrival_time=datetime(2026,6,16,9,30),
                              airline="DL", cabin=CabinClass.ECONOMY),
            ],
            origin_airport="WNZ", destination_airport="PIT",
            total_access_cost_usd=50,
        )
        result = display.format_route_human_zh(it)
        assert "上海" in result or "PVG" in result
        assert "匹兹堡" in result or "PIT" in result
        # Should not be "test"
        assert "test" not in result

    def test_format_leg_details_has_multiple_lines(self, display):
        it = Itinerary(
            id="split-1", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", flight_number="MU587",
                              cabin=CabinClass.ECONOMY),
                FlightSegment(origin="JFK", destination="PIT",
                              departure_time=datetime(2026,6,16,8,0),
                              arrival_time=datetime(2026,6,16,9,30),
                              airline="DL", flight_number="DL5123",
                              cabin=CabinClass.ECONOMY),
            ],
            origin_airport="WNZ", destination_airport="PIT",
            total_access_cost_usd=50,
            offers=[FlightOffer(id="o1", segments=[], total_price_usd=820, source="mock_fallback")],
        )
        lines = display.format_leg_details_zh(it)
        assert len(lines) >= 2, f"Expected at least 2 leg detail lines, got {len(lines)}"

    def test_airline_summary(self, display):
        it = Itinerary(
            id="test", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
                FlightSegment(origin="JFK", destination="PIT",
                              departure_time=datetime(2026,6,16,8,0),
                              arrival_time=datetime(2026,6,16,9,30),
                              airline="DL", cabin=CabinClass.ECONOMY),
            ],
        )
        summary = display.format_airline_summary(it)
        assert "MU" in summary or "Eastern" in summary or "东" in summary

    def test_recommendation_label_translates(self, display):
        assert display.recommendation_label("cheapest_reasonable") == "最省钱"
        assert display.recommendation_label("lowest_risk") == "最低风险"
        assert display.recommendation_label("best_overall") == "综合最优"


# ── AirlineService ───────────────────────────────────────────────────

class TestAirlineService:
    @pytest.fixture
    def airline_service(self):
        return AirlineService()

    def test_known_airline_lookup(self, airline_service):
        mu = airline_service.get_airline("MU")
        assert mu is not None
        assert "Eastern" in mu["name"]

    def test_unknown_airline_returns_none(self, airline_service):
        assert airline_service.get_airline("ZZ") is None

    def test_premium_scores_higher_than_budget(self, airline_service):
        premium = airline_service.score_airline_for_route("NH", True)  # ANA
        budget = airline_service.score_airline_for_route("NK", True)   # Spirit
        assert premium > budget + 0.3

    def test_display_name_zh(self, airline_service):
        name = airline_service.get_display_name("CA", "zh")
        assert "国航" in name or "Air China" in name

    def test_airline_summary_includes_warnings_for_budget(self, airline_service):
        it = Itinerary(
            id="test", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="NK", cabin=CabinClass.ECONOMY),
            ],
        )
        result = airline_service.summarize_airlines(it)
        assert result["airline_quality_score"] < 0.6
        assert len(result["airline_warnings"]) >= 1

    def test_premium_airline_gets_high_score(self, airline_service):
        it = Itinerary(
            id="test", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="NH", cabin=CabinClass.ECONOMY),
            ],
        )
        result = airline_service.summarize_airlines(it)
        assert result["airline_quality_score"] > 0.8


# ── Scoring with airline quality ─────────────────────────────────────

class TestScoringAirline:
    def test_airline_affects_final_score(self):
        scoring = ScoringService()
        it = Itinerary(
            id="test", type="hub_split", total_price_usd=1000,
            segments=[], offers=[],
        )
        from travel_agent.models.risk import RiskAssessment
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        rec_high = scoring.score(it, risk, 2000, airline_quality_score=0.9)
        rec_low = scoring.score(it, risk, 2000, airline_quality_score=0.3)
        assert rec_high.final_score > rec_low.final_score


# ── Mock provider fallback airlines ──────────────────────────────────

class TestMockFallbackAirlines:
    def test_fallback_offer_has_airline_code(self):
        provider = MockFlightProvider()
        from travel_agent.models.flight import FlightSearchRequest
        async def go():
            return await provider.search_flights(FlightSearchRequest(origin="WNZ", destination="SEA"))
        offers = asyncio.run(go())
        assert len(offers) >= 1
        for o in offers:
            if o.source == "mock_fallback":
                assert o.segments[0].airline, f"Fallback offer {o.id} should have airline code"
                assert o.segments[0].airline_name

    def test_exact_offer_has_airline_code(self):
        provider = MockFlightProvider()
        async def go():
            from travel_agent.models.flight import FlightSearchRequest
            return await provider.search_flights(FlightSearchRequest(origin="PVG", destination="JFK"))
        offers = asyncio.run(go())
        for o in offers:
            if o.source == "mock_exact":
                for seg in o.segments:
                    assert seg.airline, f"Exact offer should have airline code"


# ── CLI entry point ──────────────────────────────────────────────────

class TestCLIEntry:
    def test_display_service_imports(self):
        assert ItineraryDisplayService is not None
