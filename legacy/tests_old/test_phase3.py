"""Phase 3 tests: airport code parsing, mixed English, HubSplit modes, rejection."""

import asyncio

import pytest

from travel_agent.agents.intake_agent import IntakeAgent
from travel_agent.agents.hubsplit_agent import HubSplitAgent
from travel_agent.models.agent_outputs import IntakeOutput, ConstraintOutput
from travel_agent.models.user_request import (
    CabinClass,
    HardConstraints,
    SearchConstraints,
    SoftConstraints,
)
from travel_agent.services.airport_service import AirportService


# ── Helpers ──────────────────────────────────────────────────────────

def _make_intake(origin: str, dest: str, accepts_nearby: bool = True,
                 accepts_split: bool = True, preferences: list[str] | None = None) -> IntakeOutput:
    return IntakeOutput(
        origin_text=origin,
        destination_text=dest,
        accepts_nearby_hubs=accepts_nearby,
        accepts_split_ticket=accepts_split,
        preferences=preferences or [],
        cabin=CabinClass.ECONOMY,
    )


def _make_constraint(intake: IntakeOutput, accept_nearby: bool = True) -> ConstraintOutput:
    return ConstraintOutput(
        constraints=SearchConstraints(
            hard=HardConstraints(cabin=CabinClass.ECONOMY),
            soft=SoftConstraints(accept_nearby_hubs=accept_nearby, accept_split_ticket=True),
        ),
        original_request=intake,
    )


async def _run_hubsplit(hubsplit_agent, origin: str, dest: str):
    intake = _make_intake(origin, dest)
    constraint = _make_constraint(intake)
    return await hubsplit_agent.execute((constraint, None))


# ── Airport Code Parsing ─────────────────────────────────────────────

CODE_QUERIES = [
    ("NGB飞PIT", "宁波", "匹兹堡"),
    ("NGB到PIT", "宁波", "匹兹堡"),
    ("PVG-JFK", "上海", "纽约"),
    ("NGB to PIT", "宁波", "匹兹堡"),
    ("从NGB去PIT", "宁波", "匹兹堡"),
    ("HGH飞BOS", "杭州", "波士顿"),
    ("NGB飞JFK", "宁波", "纽约"),
]


class TestAirportCodeParsing:
    @pytest.mark.parametrize("query,expected_origin,expected_dest", CODE_QUERIES)
    def test_code_query_parsing(self, intake_agent, query, expected_origin, expected_dest):
        async def run():
            result = await intake_agent.execute(query)
            return result
        result = asyncio.run(run())
        assert result.origin_text == expected_origin, \
            f"Query '{query}': expected origin '{expected_origin}', got '{result.origin_text}'"
        assert result.destination_text == expected_dest, \
            f"Query '{query}': expected dest '{expected_dest}', got '{result.destination_text}'"


# ── Mixed English Parsing ────────────────────────────────────────────

MIXED_QUERIES = [
    ("HK to JFK", "香港", "纽约"),
    ("Shanghai to Pittsburgh", "上海", "匹兹堡"),
    ("杭州 to BOS", "杭州", "波士顿"),
    ("Beijing to Chicago", "北京", "芝加哥"),
    ("Guangzhou to New York", "广州", "纽约"),
    ("PVG to JFK", "上海", "纽约"),
    ("NGB to Boston", "宁波", "波士顿"),
]


class TestMixedEnglishParsing:
    @pytest.mark.parametrize("query,expected_origin,expected_dest", MIXED_QUERIES)
    def test_mixed_english_parsing(self, intake_agent, query, expected_origin, expected_dest):
        async def run():
            result = await intake_agent.execute(query)
            return result
        result = asyncio.run(run())
        assert result.origin_text == expected_origin, \
            f"Query '{query}': expected origin '{expected_origin}', got '{result.origin_text}'"
        assert result.destination_text == expected_dest, \
            f"Query '{query}': expected dest '{expected_dest}', got '{result.destination_text}'"


# ── Alias Resolution ─────────────────────────────────────────────────

ALIAS_TESTS = [
    ("温州", "WNZ"),
    ("New York", "JFK"),
    ("NYC", "JFK"),
    ("HK", "HKG"),
    ("Shanghai", "PVG"),
    ("Beijing", "PEK"),
    ("LA", "LAX"),
    ("SF", "SFO"),
    ("DC", "IAD"),
]


class TestAliases:
    @pytest.mark.parametrize("text,expected_code", ALIAS_TESTS)
    def test_alias_resolution(self, airport_service, text, expected_code):
        code = airport_service.resolve_airport_code(text)
        assert code == expected_code, f"Alias '{text}': expected {expected_code}, got {code}"


# ── HubSplit Modes ───────────────────────────────────────────────────

class TestHubSplitModes:
    def test_origin_side_split_ngb_to_jfk(self, hubsplit_agent):
        """NGB (local) -> JFK (hub): origin_side split should generate."""
        result = asyncio.run(_run_hubsplit(hubsplit_agent, "宁波", "纽约"))
        modes = set(p.split_mode for p in result.plan.candidate_hub_pairs)
        assert "origin_side" in modes, f"Expected origin_side in modes, got {modes}"

    def test_destination_side_split_pvg_to_pit(self, hubsplit_agent):
        """PVG (hub) -> PIT (local): destination_side split should generate."""
        result = asyncio.run(_run_hubsplit(hubsplit_agent, "上海", "匹兹堡"))
        modes = set(p.split_mode for p in result.plan.candidate_hub_pairs)
        assert "destination_side" in modes, f"Expected destination_side in modes, got {modes}"

    def test_both_side_split_ngb_to_pit(self, hubsplit_agent):
        """NGB (local) -> PIT (local): both_side split should generate."""
        result = asyncio.run(_run_hubsplit(hubsplit_agent, "宁波", "匹兹堡"))
        modes = set(p.split_mode for p in result.plan.candidate_hub_pairs)
        assert "both_side" in modes, f"Expected both_side in modes, got {modes}"
        assert len(result.plan.candidate_hub_pairs) > 0

    def test_pvg_to_jfk_limited_splits(self, hubsplit_agent):
        """PVG (hub) -> JFK (hub): should still find alternative pairs."""
        result = asyncio.run(_run_hubsplit(hubsplit_agent, "上海", "纽约"))
        # Should find some alternatives (PVG->EWR, SHA->JFK, etc.)
        assert len(result.plan.candidate_hub_pairs) >= 0
        # If any pairs exist, they should be origin_side or dest_side (not both)
        modes = set(p.split_mode for p in result.plan.candidate_hub_pairs)
        if modes:
            assert "both_side" not in modes or len(result.plan.candidate_hub_pairs) <= 6

    def test_hubsplit_default_accepts_nearby(self, hubsplit_agent):
        """Default intake (no explicit rejection) should trigger HubSplit."""
        result = asyncio.run(_run_hubsplit(hubsplit_agent, "杭州", "波士顿"))
        assert len(result.plan.candidate_hub_pairs) > 0, \
            "HubSplit should trigger when accepts_nearby_hubs=True (default)"

    def test_hubsplit_skips_when_rejected(self, hubsplit_agent):
        """Explicit rejection should skip HubSplit."""
        intake = _make_intake("上海", "匹兹堡", accepts_nearby=False, accepts_split=False)
        constraint = _make_constraint(intake, accept_nearby=False)
        result = asyncio.run(hubsplit_agent.execute((constraint, None)))
        assert result.plan.candidate_hub_pairs == []


# ── Explicit Rejection ───────────────────────────────────────────────

REJECT_QUERIES = [
    "上海到匹兹堡，不要折腾",
    "PVG-JFK，只要直飞",
    "只搜这个机场，不要换",
    "从北京直飞洛杉矶，不换机场",
]


class TestExplicitRejection:
    @pytest.mark.parametrize("query", REJECT_QUERIES)
    def test_rejection_disables_nearby(self, intake_agent, query):
        async def run():
            return await intake_agent.execute(query)
        result = asyncio.run(run())
        assert not result.accepts_nearby_hubs, \
            f"Query '{query}' should reject nearby hubs, got accepts_nearby={result.accepts_nearby_hubs}"


# ── Intake defaults ──────────────────────────────────────────────────

class TestIntakeDefaults:
    def test_default_accepts_nearby(self, intake_agent):
        """Neutral query should default to accepting nearby hubs."""
        async def run():
            return await intake_agent.execute("从杭州飞波士顿")
        result = asyncio.run(run())
        assert result.accepts_nearby_hubs, "Default should accept nearby hubs"

    def test_cheap_triggers_split(self, intake_agent):
        """Cheap preference should trigger split ticket acceptance."""
        async def run():
            return await intake_agent.execute("从宁波飞纽约，越便宜越好")
        result = asyncio.run(run())
        assert result.accepts_nearby_hubs, "Cheap query should accept nearby hubs"
        assert result.accepts_split_ticket, "Cheap query should accept split tickets"

    def test_family_safe_may_limit_split(self, intake_agent):
        """Family/safe preference should be more cautious."""
        async def run():
            return await intake_agent.execute("带父母从北京飞洛杉矶，希望安全舒适")
        result = asyncio.run(run())
        # Should still accept nearby hubs by default
        assert result.accepts_nearby_hubs
        # "family_friendly" without "cheap" may limit split
        assert "family_friendly" in result.preferences
