"""Phase 14 tests: exclusion pre-filter propagation through pipeline."""

import asyncio
from datetime import datetime

import pytest

from travel_agent.services.constraint_compiler import ExclusionRules, ConstraintCompiler
from travel_agent.services.recommendation_validator import RecommendationValidator
from travel_agent.models.requirement_contract import (
    TravelRequirementContract, TripRequirement, HubPreferences,
)
from travel_agent.agents.hubsplit_agent import HubSplitAgent
from travel_agent.agents.search_strategy_agent import SearchStrategyAgent
from travel_agent.agents.route_composer_agent import RouteComposerAgent
from travel_agent.agents.flight_retrieval_agent import FlightRetrievalAgent
from travel_agent.models.agent_outputs import (
    ConstraintOutput, IntakeOutput, HubSplitOutput, SearchStrategyOutput,
    FlightRetrievalOutput,
)
from travel_agent.models.user_request import (
    HardConstraints, SearchConstraints, SoftConstraints, CabinClass,
)
from travel_agent.models.airport import HubSplitPlan
from travel_agent.providers.mock_flight_provider import MockFlightProvider


# ── ExclusionRules ───────────────────────────────────────────────────

class TestExclusionRules:
    def test_shanghai_expands_to_pvg_sha(self):
        ex = ExclusionRules(excluded_cities=["上海"])
        ex.expand_cities()
        assert "PVG" in ex.excluded_airports
        assert "SHA" in ex.excluded_airports

    def test_new_york_expands_to_jfk_ewr_lga(self):
        ex = ExclusionRules(excluded_cities=["New York"])
        ex.expand_cities()
        assert "JFK" in ex.excluded_airports

    def test_is_airport_excluded(self):
        ex = ExclusionRules(excluded_airports=["PVG", "SHA"])
        assert ex.is_airport_excluded("PVG")
        assert not ex.is_airport_excluded("HGH")

    def test_route_contains_exclusion(self):
        ex = ExclusionRules(excluded_airports=["PVG"])
        assert ex.route_contains_exclusion(["WNZ", "PVG", "JFK", "PIT"])
        assert not ex.route_contains_exclusion(["WNZ", "HGH", "JFK", "PIT"])


# ── HubSplit with exclusions ─────────────────────────────────────────

def _make_intake(origin="温州", dest="匹兹堡"):
    return IntakeOutput(origin_text=origin, destination_text=dest,
                        accepts_nearby_hubs=True, accepts_split_ticket=True,
                        cabin=CabinClass.ECONOMY)

def _make_constraint():
    intake = _make_intake()
    return ConstraintOutput(
        constraints=SearchConstraints(
            hard=HardConstraints(cabin=CabinClass.ECONOMY),
            soft=SoftConstraints(accept_nearby_hubs=True, accept_split_ticket=True),
        ),
        original_request=intake,
    )


class TestHubSplitExclusion:
    @pytest.fixture
    def agent(self):
        return HubSplitAgent()

    def test_excludes_pvg_when_shanghai_excluded(self, agent):
        ex = ExclusionRules(excluded_airports=["PVG", "SHA"])
        result = asyncio.run(agent.execute((_make_constraint(), ex)))
        for p in result.plan.candidate_hub_pairs:
            assert p.origin_hub_code != "PVG", f"PVG should be excluded but found: {p.origin_hub_code}->{p.destination_hub_code}"
            assert p.origin_hub_code != "SHA"

    def test_no_exclusions_allows_pvg(self, agent):
        result = asyncio.run(agent.execute((_make_constraint(), None)))
        codes = {p.origin_hub_code for p in result.plan.candidate_hub_pairs}
        assert "PVG" in codes, f"PVG should appear without exclusions. Got: {codes}"


# ── SearchStrategy with exclusions ───────────────────────────────────

class TestSearchStrategyExclusion:
    @pytest.fixture
    def agent(self):
        return SearchStrategyAgent()

    def test_drops_pvg_tasks(self, agent):
        from travel_agent.models.airport import HubPair
        plan = HubSplitPlan(
            origin_airport_code="WNZ", destination_airport_code="PIT",
            destination_hubs=[],
            candidate_hub_pairs=[
                HubPair(origin_hub_code="PVG", destination_hub_code="JFK", split_mode="both_side"),
                HubPair(origin_hub_code="HGH", destination_hub_code="JFK", split_mode="both_side"),
            ],
        )
        ex = ExclusionRules(excluded_airports=["PVG", "SHA"])
        result = asyncio.run(agent.execute((HubSplitOutput(plan=plan), ex)))
        origins = {t.origin for t in result.search_tasks}
        assert "PVG" not in origins
        assert "HGH" in origins


# ── RouteComposer with exclusions ────────────────────────────────────

class TestRouteComposerExclusion:
    @pytest.fixture
    def agent(self):
        return RouteComposerAgent()

    def test_filters_pvg_itinerary(self, agent):
        from travel_agent.models.flight import FlightSegment, FlightOffer
        it_plan = HubSplitPlan(origin_airport_code="WNZ", destination_airport_code="PIT")

        offer = FlightOffer(
            id="test", segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
            ], total_price_usd=820, provider_name="mock", source="test",
        )

        hub = HubSplitOutput(plan=it_plan)
        flight = FlightRetrievalOutput(
            hub_split_offers=[offer], all_offers=[offer],
        )
        ex = ExclusionRules(excluded_airports=["PVG"])
        result = asyncio.run(agent.execute((flight, hub, ex)))
        # Should have filtered out the PVG itinerary
        for it in result.output.itineraries:
            codes = [s.origin for s in it.segments]
            assert "PVG" not in codes, f"PVG should be filtered: {codes}"


# ── RecommendationValidator final safety net ─────────────────────────

class TestValidatorSafetyNet:
    def test_validator_removes_pvg(self):
        from travel_agent.models.flight import FlightSegment
        from travel_agent.models.itinerary import Itinerary
        from travel_agent.models.ranking import RankedRecommendation

        it = Itinerary(
            id="t", type="hub_split",
            segments=[FlightSegment(origin="PVG", destination="JFK",
                        departure_time=datetime(2026,6,15,17,0),
                        arrival_time=datetime(2026,6,15,20,0), airline="MU")],
            origin_airport="WNZ", destination_airport="PIT",
        )
        rec = RankedRecommendation(itinerary=it)
        ex = ExclusionRules(excluded_airports=["PVG"])
        validator = RecommendationValidator()
        valid = validator.validate([rec], ex)
        assert len(valid) == 0


# ── FlightRetrieval skips excluded ───────────────────────────────────

class TestFlightRetrievalExclusion:
    def test_skips_excluded_tasks(self):
        from travel_agent.models.flight import FlightSearchRequest
        agent = FlightRetrievalAgent(router=None)
        search = SearchStrategyOutput(search_tasks=[
            FlightSearchRequest(origin="PVG", destination="JFK"),
            FlightSearchRequest(origin="HGH", destination="JFK"),
        ])
        ex = ExclusionRules(excluded_airports=["PVG", "SHA"])
        result = asyncio.run(agent.execute((search, ex)))
        origins = {s.origin for o in result.all_offers for s in o.segments}
        assert "PVG" not in origins
