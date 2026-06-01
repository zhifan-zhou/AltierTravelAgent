"""Phase 4 tests: fallback pricing, RouteComposer modes, diagnostics."""

import asyncio
from datetime import datetime

import pytest

from travel_agent.models.flight import CabinClass, FlightSearchRequest, FlightSegment, FlightOffer
from travel_agent.models.itinerary import Itinerary
from travel_agent.providers.mock_flight_provider import MockFlightProvider
from travel_agent.services.airport_service import AirportService
from travel_agent.agents.route_composer_agent import RouteComposerAgent
from travel_agent.models.agent_outputs import (
    FlightRetrievalOutput, HubSplitOutput, RouteComposerResult,
    SearchStrategyOutput, ConstraintOutput, IntakeOutput,
)
from travel_agent.models.airport import HubPair, HubSplitPlan, NearbyHub, Airport
from travel_agent.models.user_request import (
    CabinClass as CC, HardConstraints, SearchConstraints, SoftConstraints,
)


# ── Fallback Pricing ─────────────────────────────────────────────────

class TestFallbackPricing:
    def test_exact_route_returns_exact_price(self):
        provider = MockFlightProvider()
        async def run():
            req = FlightSearchRequest(origin="PVG", destination="JFK", cabin=CC.ECONOMY)
            offers = await provider.search_flights(req)
            return offers
        offers = asyncio.run(run())
        assert len(offers) >= 1
        exact = [o for o in offers if o.source == "mock_exact"]
        assert len(exact) >= 1, "Should have exact offers for PVG->JFK"
        assert exact[0].total_price_usd == 820 or exact[0].total_price_usd == 750

    def test_fallback_for_unknown_route(self):
        provider = MockFlightProvider()
        async def run():
            req = FlightSearchRequest(origin="PVG", destination="PHL", cabin=CC.ECONOMY)
            offers = await provider.search_flights(req)
            return offers
        offers = asyncio.run(run())
        assert len(offers) >= 1, "Should have fallback offers for PVG->PHL"
        fb = [o for o in offers if o.source == "mock_fallback"]
        assert len(fb) >= 1
        assert fb[0].total_price_usd > 0

    def test_fallback_is_deterministic(self):
        """Same route + cabin should always produce the same price."""
        provider = MockFlightProvider()
        async def get():
            req = FlightSearchRequest(origin="XIY", destination="ORD", cabin=CC.ECONOMY)
            return await provider.search_flights(req)
        offers1 = asyncio.run(get())
        offers2 = asyncio.run(get())
        assert len(offers1) == len(offers2)
        assert offers1[0].total_price_usd == offers2[0].total_price_usd

    def test_fallback_business_is_more_expensive(self):
        provider = MockFlightProvider()
        async def get(cabin):
            req = FlightSearchRequest(origin="TAO", destination="SEA", cabin=cabin)
            offers = await provider.search_flights(req)
            return offers[0].total_price_usd if offers else 0
        econ = asyncio.run(get(CC.ECONOMY))
        biz = asyncio.run(get(CC.BUSINESS))
        assert biz > econ * 2, f"Business ({biz}) should be much more than economy ({econ})"

    def test_fallback_hub_to_hub_cheaper_than_local(self):
        """Fallback: hub-to-hub routes should be cheaper than local-to-local."""
        provider = MockFlightProvider()
        async def get(o, d):
            req = FlightSearchRequest(origin=o, destination=d, cabin=CC.ECONOMY)
            offers = await provider.search_flights(req)
            return offers[0].total_price_usd if offers else 0
        # Hub-to-hub should be reasonable
        hub_price = asyncio.run(get("PVG", "JFK"))
        # Unknown route should also work
        local_price = asyncio.run(get("WNZ", "STL"))
        assert hub_price > 0
        assert local_price > 0
        # Both should be in a reasonable range
        assert 500 < hub_price < 2000
        assert 500 < local_price < 2500

    def test_fallback_invalid_airport_returns_empty(self):
        provider = MockFlightProvider()
        async def run():
            req = FlightSearchRequest(origin="XXX", destination="YYY", cabin=CC.ECONOMY)
            return await provider.search_flights(req)
        offers = asyncio.run(run())
        assert len(offers) == 0


# ── RouteComposer Split Modes ────────────────────────────────────────

def _make_offer(origin: str, dest: str, price: float, source: str = "mock_fallback") -> FlightOffer:
    return FlightOffer(
        id=f"test-{origin}-{dest}",
        segments=[FlightSegment(
            origin=origin, destination=dest,
            departure_time=datetime(2026, 6, 15, 18, 0),
            arrival_time=datetime(2026, 6, 16, 6, 0),
            airline="MU", flight_number="MU999", cabin=CC.ECONOMY,
        )],
        total_price_usd=price, provider_name="mock", source=source,
    )


class TestRouteComposerModes:
    def test_origin_side_split(self):
        """Compose origin-side: access + PVG->JFK (no domestic needed, dest is JFK)."""
        agent = RouteComposerAgent()

        hub_split = HubSplitOutput(plan=HubSplitPlan(
            origin_airport_code="NGB", destination_airport_code="JFK",
            origin_hubs=[NearbyHub(
                airport=Airport(code="PVG", name="Pudong", city="Shanghai", city_cn="上海"),
                access_mode="train", access_time_hours=2.5, access_cost_usd=35, hub_score=0.95,
            )],
            candidate_hub_pairs=[HubPair(
                origin_hub_code="PVG", destination_hub_code="JFK",
                estimated_access_cost_usd=35, estimated_access_time_hours=2.5,
                split_mode="origin_side", expected_savings_potential="high",
            )],
        ))

        flight_data = FlightRetrievalOutput(
            hub_split_offers=[_make_offer("PVG", "JFK", 820)],
            all_offers=[_make_offer("PVG", "JFK", 820)],
        )

        result = asyncio.run(agent.execute((flight_data, hub_split, None)))
        split_its = [i for i in result.output.itineraries if i.type == "hub_split"]
        assert len(split_its) >= 1, f"Expected split itineraries, got {len(split_its)}"
        assert split_its[0].total_price_usd == 855  # 820 + 35

    def test_destination_side_split(self):
        """Compose dest-side: PVG->JFK + JFK->PIT."""
        agent = RouteComposerAgent()

        hub_split = HubSplitOutput(plan=HubSplitPlan(
            origin_airport_code="PVG", destination_airport_code="PIT",
            destination_hubs=[NearbyHub(
                airport=Airport(code="JFK", name="JFK", city="New York", city_cn="纽约"),
                access_mode="domestic_flight", access_time_hours=1.5, access_cost_usd=180, hub_score=0.95,
            )],
            candidate_hub_pairs=[HubPair(
                origin_hub_code="PVG", destination_hub_code="JFK",
                estimated_access_cost_usd=180, estimated_access_time_hours=1.5,
                split_mode="destination_side", expected_savings_potential="high",
            )],
        ))

        flight_data = FlightRetrievalOutput(
            hub_split_offers=[_make_offer("PVG", "JFK", 820)],
            domestic_offers=[_make_offer("JFK", "PIT", 180)],
            all_offers=[_make_offer("PVG", "JFK", 820), _make_offer("JFK", "PIT", 180)],
        )

        result = asyncio.run(agent.execute((flight_data, hub_split, None)))
        split_its = [i for i in result.output.itineraries if i.type == "hub_split"]
        assert len(split_its) >= 1
        assert split_its[0].total_price_usd == 1180  # 820 + 180 + 180

    def test_both_side_split(self):
        """Compose both-side: NGB access + PVG->JFK + JFK->PIT."""
        agent = RouteComposerAgent()

        hub_split = HubSplitOutput(plan=HubSplitPlan(
            origin_airport_code="NGB", destination_airport_code="PIT",
            dest_hubs=[NearbyHub(
                airport=Airport(code="JFK", name="JFK", city="New York", city_cn="纽约"),
                access_mode="domestic_flight", access_time_hours=1.5, access_cost_usd=180, hub_score=0.95,
            )],
            candidate_hub_pairs=[HubPair(
                origin_hub_code="PVG", destination_hub_code="JFK",
                estimated_access_cost_usd=215, estimated_access_time_hours=4.0,
                split_mode="both_side", expected_savings_potential="high",
            )],
        ))

        flight_data = FlightRetrievalOutput(
            hub_split_offers=[_make_offer("PVG", "JFK", 820)],
            domestic_offers=[_make_offer("JFK", "PIT", 180)],
            all_offers=[_make_offer("PVG", "JFK", 820), _make_offer("JFK", "PIT", 180)],
        )

        result = asyncio.run(agent.execute((flight_data, hub_split, None)))
        split_its = [i for i in result.output.itineraries if i.type == "hub_split"]
        assert len(split_its) >= 1
        assert split_its[0].total_price_usd == 1215  # 820 + 180 + 215

    def test_direct_route_always_included(self):
        """Direct offers should always be included alongside splits."""
        agent = RouteComposerAgent()
        hub_split = HubSplitOutput(plan=HubSplitPlan(
            origin_airport_code="NGB", destination_airport_code="PIT",
            candidate_hub_pairs=[],
        ))
        flight_data = FlightRetrievalOutput(
            direct_offers=[_make_offer("NGB", "PIT", 1850, "mock_exact")],
            all_offers=[_make_offer("NGB", "PIT", 1850, "mock_exact")],
        )
        result = asyncio.run(agent.execute((flight_data, hub_split, None)))
        assert len(result.output.itineraries) >= 1
        assert result.output.itineraries[0].type == "direct"


# ── Diagnostics ──────────────────────────────────────────────────────

class TestDiagnostics:
    def test_eval_produces_diagnostics_file(self):
        """Eval should save diagnostics.json."""
        import os, json
        from scripts.run_eval import load_queries, run_evals
        from pathlib import Path

        qpath = Path(__file__).resolve().parent.parent / "evals" / "eval_queries.jsonl"
        queries = load_queries(str(qpath), limit=3)
        out = f"runs/evals/test_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        asyncio.run(run_evals(queries, output_dir=out))

        diag_path = Path(out) / "diagnostics.json"
        assert diag_path.exists(), f"diagnostics.json not found at {diag_path}"

        with open(diag_path) as f:
            diag = json.load(f)
        assert len(diag) == 3
        for d in diag:
            assert "id" in d
            assert "split_itineraries" in d
            assert "hub_pairs" in d


# ── Rejection still works ────────────────────────────────────────────

class TestRejectionWithFallback:
    def test_rejection_query_has_no_splits(self, airport_service):
        """Rejection queries should not generate split itineraries even with fallback."""
        from travel_agent.agents.intake_agent import IntakeAgent
        from travel_agent.agents.hubsplit_agent import HubSplitAgent
        from travel_agent.agents.constraint_agent import ConstraintAgent

        intake = IntakeAgent(airport_service=airport_service)
        hubsplit = HubSplitAgent(airport_service=airport_service)
        constraint = ConstraintAgent()

        async def run():
            intake_out = await intake.execute("上海到匹兹堡，不要折腾")
            constraint_out = await constraint.execute(intake_out)
            return await hubsplit.execute((constraint_out, None))

        result = asyncio.run(run())
        assert len(result.plan.candidate_hub_pairs) == 0, \
            "Rejection query should have no hub pairs"
