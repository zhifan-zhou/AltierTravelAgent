"""Phase 5 tests: provider architecture, provenance, intake fix, hybrid mode."""

import asyncio

import pytest

from travel_agent.providers.provider_router import ProviderRouter
from travel_agent.providers.base import ProviderConfigurationError
from travel_agent.providers.skeleton_providers import (
    SerpApiGoogleFlightsProvider, SearchApiGoogleFlightsProvider,
    SkyscannerProvider, KiwiProvider, AmadeusLegacyProvider,
)
from travel_agent.providers.mock_flight_provider import MockFlightProvider
from travel_agent.models.flight import FlightOffer, FlightSegment, CabinClass, FlightSearchRequest
from travel_agent.agents.intake_agent import IntakeAgent
from travel_agent.services.airport_service import AirportService


# ── Mock mode works without keys ─────────────────────────────────────

class TestMockMode:
    def test_router_defaults_to_mock(self):
        router = ProviderRouter()
        assert router.mode == "mock"

    def test_mock_search_works(self):
        router = ProviderRouter()
        async def run():
            return await router.search_flights(FlightSearchRequest(origin="PVG", destination="JFK"))
        offers = asyncio.run(run())
        assert len(offers) > 0

    def test_mock_offers_have_provenance(self):
        router = ProviderRouter()
        async def run():
            return await router.search_flights(FlightSearchRequest(origin="PVG", destination="JFK"))
        offers = asyncio.run(run())
        for o in offers:
            assert o.provider_name == "mock"
            assert o.is_real is False
            assert o.source in ("mock_exact", "mock_fallback")
            assert o.confidence in ("demo", "estimated")
            assert o.data_quality in ("demo_exact", "demo_estimated")


# ── Duffel mode missing key ──────────────────────────────────────────

class TestDuffelMissingKey:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "duffel")

    def test_duffel_missing_token_gives_config_error(self):
        from travel_agent.providers.duffel_provider import DuffelProvider
        provider = DuffelProvider()
        with pytest.raises(ProviderConfigurationError, match="DUFFEL_API_TOKEN"):
            provider.validate_config()


# ── Hybrid mode ──────────────────────────────────────────────────────

class TestHybridMode:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("TRAVEL_AGENT_PROVIDER", "mock")  # Default

    def test_hybrid_falls_back_to_mock_when_no_real_keys(self):
        """Hybrid with no real keys should use mock fallback."""
        router = ProviderRouter()
        async def run():
            return await router.search_flights(FlightSearchRequest(origin="PVG", destination="JFK"))
        offers = asyncio.run(run())
        assert len(offers) > 0
        # All should be mock since no real keys
        for o in offers:
            assert o.is_real is False


# ── Provider priority parsing ────────────────────────────────────────

class TestProviderPriority:
    def test_default_priority_has_correct_order(self):
        from travel_agent.core.config import get_settings
        settings = get_settings()
        priority = settings.real_provider_priority
        providers = [p.strip() for p in priority.split(",")]
        assert "duffel" in providers
        assert providers[0] == "duffel", "Duffel should be first in priority"

    def test_custom_priority(self, monkeypatch):
        monkeypatch.setenv("REAL_PROVIDER_PRIORITY", "skyscanner,kiwi,duffel")
        from travel_agent.core.config import Settings
        s = Settings()
        providers = [p.strip() for p in s.real_provider_priority.split(",")]
        assert providers[0] == "skyscanner"


# ── Provider skeletons ───────────────────────────────────────────────

class TestSkeletons:
    def test_serpapi_requires_key(self):
        p = SerpApiGoogleFlightsProvider()
        with pytest.raises(ProviderConfigurationError, match="SERPAPI_API_KEY"):
            p.validate_config()

    def test_searchapi_requires_key(self):
        p = SearchApiGoogleFlightsProvider()
        with pytest.raises(ProviderConfigurationError, match="SEARCHAPI_API_KEY"):
            p.validate_config()

    def test_skyscanner_requires_key(self):
        p = SkyscannerProvider()
        with pytest.raises(ProviderConfigurationError, match="SKYSCANNER_API_KEY"):
            p.validate_config()

    def test_kiwi_requires_key(self):
        p = KiwiProvider()
        with pytest.raises(ProviderConfigurationError, match="KIWI_API_KEY"):
            p.validate_config()

    def test_amadeus_legacy_requires_keys(self):
        p = AmadeusLegacyProvider()
        with pytest.raises(ProviderConfigurationError, match="AMADEUS_API_KEY"):
            p.validate_config()

    def test_all_skeletons_support_search_only(self):
        for cls in [SerpApiGoogleFlightsProvider, SearchApiGoogleFlightsProvider,
                     SkyscannerProvider, KiwiProvider, AmadeusLegacyProvider]:
            caps = cls().capabilities
            assert caps.supports_search is True
            assert caps.supports_booking is False
            assert caps.is_real_provider is True

    def test_skeletons_return_empty_offers(self):
        p = AmadeusLegacyProvider()
        async def run():
            return await p.search_flights(FlightSearchRequest(origin="PVG", destination="JFK"))
        offers = asyncio.run(run())
        assert offers == []


# ── FlightOffer provenance fields ────────────────────────────────────

class TestFlightOfferProvenance:
    def test_default_fields(self):
        offer = FlightOffer(id="test", segments=[], total_price_usd=100)
        assert offer.provider_name == "mock"
        assert offer.source == "mock_fallback"
        assert offer.is_real is False
        assert offer.confidence == "estimated"
        assert offer.data_quality == "demo_estimated"

    def test_real_provider_fields(self):
        offer = FlightOffer(
            id="test", total_price_usd=100,
            provider_name="duffel", source="duffel_api",
            is_real=True, confidence="verified",
        )
        assert offer.data_quality == "verified"
        assert offer.is_real is True

    def test_raw_payload_field(self):
        payload = {"raw": "data", "nested": {"key": "value"}}
        offer = FlightOffer(id="test", total_price_usd=100, raw_provider_payload=payload)
        assert offer.raw_provider_payload == payload


# ── Intake fix: 温州到匹兹堡、可以从上海走 ───────────────────────────

class TestIntakeFix:
    def test_wenzhou_primary_not_shanghai(self, airport_service):
        agent = IntakeAgent(airport_service=airport_service)
        async def run():
            return await agent.execute("温州到匹兹堡，可以从上海走")
        result = asyncio.run(run())
        # Origin should be 温州, NOT 上海
        assert result.origin_text == "温州", \
            f"Expected 温州 as primary origin, got {result.origin_text}"
        assert result.destination_text == "匹兹堡"
        # Should still accept nearby hubs
        assert result.accepts_nearby_hubs is True

    def test_ningbo_primary_not_shanghai(self, airport_service):
        agent = IntakeAgent(airport_service=airport_service)
        async def run():
            return await agent.execute("宁波到纽约，可以从上海走")
        result = asyncio.run(run())
        assert result.origin_text == "宁波", \
            f"Expected 宁波 as primary origin, got {result.origin_text}"

    def test_original_style_query_still_works(self, airport_service):
        """Queries without '从X走' should still work normally."""
        agent = IntakeAgent(airport_service=airport_service)
        async def run():
            return await agent.execute("我要从宁波飞匹兹堡，便宜点，可以从上海走，也可以纽约或者华盛顿转")
        result = asyncio.run(run())
        # "从宁波飞" should extract 宁波 as origin
        assert result.origin_text == "宁波"


# ── Provider cache ───────────────────────────────────────────────────

class TestProviderCache:
    def test_cache_hit(self):
        from travel_agent.providers.provider_cache import ProviderCache
        cache = ProviderCache(ttl_seconds=60)
        cache.set("mock", "PVG", "JFK", "2026-06-15", "economy", 1, ["cached"])
        result = cache.get("mock", "PVG", "JFK", "2026-06-15", "economy", 1)
        assert result == ["cached"]

    def test_cache_miss(self):
        from travel_agent.providers.provider_cache import ProviderCache
        cache = ProviderCache(ttl_seconds=60)
        assert cache.get("mock", "XXX", "YYY", "any", "economy", 1) is None

    def test_cache_size(self):
        from travel_agent.providers.provider_cache import ProviderCache
        cache = ProviderCache()
        cache.set("a", "b", "c", "d", "e", 1, "v")
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0
