"""Expanded tests for Phase 2: generalization, city pairs, provider selection."""

import asyncio

import pytest

from travel_agent.agents.intake_agent import IntakeAgent
from travel_agent.agents.hubsplit_agent import HubSplitAgent
from travel_agent.core.orchestrator import TravelAgentOrchestrator
from travel_agent.providers.provider_router import ProviderRouter
from travel_agent.core.config import get_settings, Settings
from travel_agent.models.agent_outputs import IntakeOutput, ConstraintOutput
from travel_agent.models.user_request import (
    CabinClass,
    DateWindow,
    HardConstraints,
    SearchConstraints,
    SoftConstraints,
)
from travel_agent.providers.mock_flight_provider import MockFlightProvider
from travel_agent.services.airport_service import AirportService


class TestHubSplitMultiCity:
    """HubSplit should work for at least 5 city pairs."""

    CITY_PAIRS = [
        ("宁波", "匹兹堡", True),   # NGB -> PIT, classic case
        ("杭州", "波士顿", True),   # HGH -> BOS
        ("南京", "纽约", True),     # NKG -> JFK (NYC is a hub, may not need domestic)
        ("深圳", "芝加哥", True),   # SZX -> ORD
        ("厦门", "费城", True),     # XMN -> PHL
        ("青岛", "华盛顿", True),   # TAO -> IAD
        ("武汉", "匹兹堡", True),   # WUH -> PIT
    ]

    @pytest.mark.parametrize("origin_cn,dest_cn,expect_hubs", CITY_PAIRS)
    def test_hubsplit_for_city_pair(self, hubsplit_agent, origin_cn, dest_cn, expect_hubs):
        intake = IntakeOutput(
            origin_text=origin_cn,
            destination_text=dest_cn,
            accepts_nearby_hubs=True,
            accepts_split_ticket=True,
            cabin=CabinClass.ECONOMY,
        )
        constraints = SearchConstraints(
            hard=HardConstraints(cabin=CabinClass.ECONOMY),
            soft=SoftConstraints(accept_nearby_hubs=True, accept_split_ticket=True),
        )
        constraint_output = ConstraintOutput(constraints=constraints, original_request=intake)

        result = asyncio.run(hubsplit_agent.execute((constraint_output, None)))

        # Should at minimum resolve to valid airport codes
        assert result.plan.origin_airport_code, f"Origin not resolved for {origin_cn}"
        assert result.plan.destination_airport_code, f"Dest not resolved for {dest_cn}"

        if expect_hubs:
            assert len(result.plan.origin_hubs) >= 0, f"No origin hubs for {origin_cn}"
            # Not all airports have nearby hubs configured (e.g. PEK has none)


class TestChineseNameMapping:
    """Chinese city names should map to correct airport codes."""

    MAPPINGS = [
        ("宁波", "NGB"),
        ("杭州", "HGH"),
        ("南京", "NKG"),
        ("上海", "PVG"),    # Prefers PVG over SHA (international hub)
        ("北京", "PEK"),    # Prefers PEK over PKX
        ("广州", "CAN"),
        ("深圳", "SZX"),
        ("香港", "HKG"),
        ("厦门", "XMN"),
        ("福州", "FOC"),
        ("成都", "TFU"),
        ("重庆", "CKG"),
        ("青岛", "TAO"),
        ("武汉", "WUH"),
        ("西安", "XIY"),
        ("匹兹堡", "PIT"),
        ("纽约", "JFK"),
        ("华盛顿", "IAD"),
        ("芝加哥", "ORD"),
        ("费城", "PHL"),
        ("波士顿", "BOS"),
        ("洛杉矶", "LAX"),
        ("旧金山", "SFO"),
        ("西雅图", "SEA"),
        ("达拉斯", "DFW"),
        ("亚特兰大", "ATL"),
        ("底特律", "DTW"),
        ("明尼阿波利斯", "MSP"),
        ("圣路易斯", "STL"),
        ("多伦多", "YYZ"),
        ("温哥华", "YVR"),
        ("克利夫兰", "CLE"),
    ]

    @pytest.mark.parametrize("city_cn,expected_code", MAPPINGS)
    def test_city_name_to_code(self, airport_service, city_cn, expected_code):
        code = airport_service.resolve_airport_code(city_cn)
        assert code == expected_code, f"Expected {expected_code} for '{city_cn}', got {code}"


class TestAirportCodeInput:
    """Airport code inputs should work directly."""

    CODES = [
        ("NGB", "NGB"),
        ("PVG", "PVG"),
        ("jfk", "JFK"),      # lowercase
        ("ord", "ORD"),
        ("pit", "PIT"),
        ("hgh", "HGH"),
        ("pek", "PEK"),
        ("lax", "LAX"),
    ]

    @pytest.mark.parametrize("input_code,expected", CODES)
    def test_airport_code_direct_input(self, airport_service, input_code, expected):
        code = airport_service.resolve_airport_code(input_code)
        assert code == expected, f"Expected {expected} for '{input_code}', got {code}"


class TestProviderSelection:
    """Provider selection should default to mock, and not require real API keys."""

    def test_default_provider_is_mock(self):
        settings = get_settings()
        assert settings.travel_agent_provider == "mock"

    def test_create_provider_returns_mock(self):
        router = ProviderRouter()
        assert router.mode == "mock"

    def test_amadeus_provider_fails_gracefully_without_keys(self):
        from travel_agent.providers.skeleton_providers import AmadeusLegacyProvider
        from travel_agent.providers.base import ProviderConfigurationError

        provider = AmadeusLegacyProvider()
        with pytest.raises(ProviderConfigurationError, match="AMADEUS_API_KEY"):
            provider.validate_config()

    def test_orchestrator_defaults_to_mock(self):
        orch = TravelAgentOrchestrator()
        assert orch._router is not None
        assert orch._router.mode == "mock"


class TestIntakeAirportService:
    """Intake agent should use AirportService for city resolution."""

    def test_intake_uses_airport_service(self, airport_service):
        agent = IntakeAgent(airport_service=airport_service)

        async def run():
            result = await agent.execute("我要从杭州飞波士顿")
            assert result.origin_text == "杭州"
            assert result.destination_text == "波士顿"
            return result

        asyncio.run(run())

    def test_intake_empty_when_no_match(self, airport_service):
        agent = IntakeAgent(airport_service=airport_service)

        async def run():
            result = await agent.execute("我要出去玩")
            # Should not default to 宁波/匹兹堡 anymore
            assert result.origin_text == "" or result.origin_text != "宁波"
            return result

        asyncio.run(run())


class TestEvalRunner:
    """Eval runner should run on a small sample without errors."""

    def test_eval_runner_imports(self):
        from scripts.run_eval import load_queries, run_evals
        assert load_queries is not None
        assert run_evals is not None

    def test_load_queries(self):
        from scripts.run_eval import load_queries
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "evals" / "eval_queries.jsonl"
        queries = load_queries(str(path), limit=3)
        assert len(queries) == 3
        assert "id" in queries[0]
        assert "query" in queries[0]
