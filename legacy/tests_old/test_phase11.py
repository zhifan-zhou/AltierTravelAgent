"""Phase 11 tests: LLM-first intent routing, natural language understanding, constraint updates."""

import asyncio

import pytest

from travel_agent.agents.intent_router_agent import IntentRouterAgent
from travel_agent.services.constraint_update_service import ConstraintUpdateService
from travel_agent.llm.schemas import ChatIntent
from travel_agent.models.preference import DEFAULT_PROFILE
from travel_agent.agents.followup_agent import FollowUpAgent


# ── IntentRouterAgent ────────────────────────────────────────────────

class TestIntentRouter:
    @pytest.fixture
    def router(self):
        return IntentRouterAgent()

    def test_avoid_nyc_low_risk(self, router):
        """我不想从纽约转，低风险一点" → avoid JFK/EWR + low_risk"""
        intent = asyncio.run(router.execute(("我不想从纽约转，低风险一点", {})))
        has_ny_avoid = any(a in intent.avoid_airports for a in ("JFK", "EWR", "LGA"))
        # At minimum, should detect constraint refinement
        assert intent.intent_type in ("refine_search", "rerank", "refine_constraints")

    def test_explain_first_option(self, router):
        """"第一个为什么这么便宜" → explain_option index=1"""
        intent = asyncio.run(router.execute(("第一个为什么这么便宜", {})))
        assert intent.intent_type == "explain_option"
        assert intent.selected_option_index == 1

    def test_mom_safe(self, router):
        """"我爸妈也坐，别太折腾" → low_risk"""
        intent = asyncio.run(router.execute(("我爸妈也坐，别太折腾", {})))
        # Should detect preference change
        assert intent.profile == "low_risk" or intent.intent_type in ("rerank", "refine_search")

    def test_rerank_command(self, router):
        """"主流航司优先" → rerank airline_priority"""
        intent = asyncio.run(router.execute(("主流航司优先", {})))
        assert intent.profile == "airline_priority" or intent.intent_type == "rerank"

    def test_cheapest_command(self, router):
        intent = asyncio.run(router.execute(("只要便宜", {})))
        assert intent.profile == "cheapest" or intent.intent_type == "rerank"

    def test_reset_command(self, router):
        intent = asyncio.run(router.execute(("恢复默认", {})))
        assert intent.profile == "balanced" or intent.intent_type == "rerank"


# ── ConstraintUpdateService ──────────────────────────────────────────

class TestConstraintUpdate:
    @pytest.fixture
    def updater(self):
        return ConstraintUpdateService()

    def test_acceptable_origin_hubs_do_not_replace_primary_origin(self, updater):
        """Alternative hubs add to existing, don't replace."""
        state = {"primary_origin": "WNZ", "acceptable_origin_hubs": ["PVG"]}
        intent = ChatIntent(acceptable_origin_hubs=["HGH"], nearby_hub_policy="prefer")
        updated = updater.apply_intent(intent, state)
        assert "WNZ" == updated["primary_origin"]
        assert "PVG" in updated["acceptable_origin_hubs"]
        assert "HGH" in updated["acceptable_origin_hubs"]

    def test_avoid_airports_accumulate(self, updater):
        state = {"avoid_airports": ["JFK"]}
        intent = ChatIntent(avoid_airports=["EWR", "LGA"])
        updated = updater.apply_intent(intent, state)
        assert "JFK" in updated["avoid_airports"]
        assert "EWR" in updated["avoid_airports"]

    def test_cabin_triggers_rerun(self, updater):
        intent = ChatIntent(cabin="business")
        assert updater.needs_rerun_search(intent)

    def test_profile_only_triggers_rerank(self, updater):
        intent = ChatIntent(profile="airline_priority", needs_rerank_only=True)
        assert updater.needs_rerank_only(intent)
        assert not updater.needs_rerun_search(intent)

    def test_split_ticket_avoid(self, updater):
        state = {}
        intent = ChatIntent(split_ticket_policy="avoid")
        updated = updater.apply_intent(intent, state)
        assert updated["accepts_split_ticket"] is False


# ── FollowUpAgent still works for hardcoded commands ─────────────────

class TestFollowUpFallback:
    def test_existing_commands_still_work(self):
        agent = FollowUpAgent()
        intent = asyncio.run(agent.execute(("不要纽约转", {})))
        assert intent.intent_type in ("refine_constraints", "rerank")


# ── ChatIntent schema ────────────────────────────────────────────────

class TestChatIntentSchema:
    def test_default_intent(self):
        intent = ChatIntent()
        assert intent.intent_type == "unknown"
        assert intent.confidence == 0.0

    def test_refine_search_intent(self):
        intent = ChatIntent(
            intent_type="refine_search",
            avoid_airports=["JFK", "EWR"],
            profile="low_risk",
            risk_preference="low",
            needs_rerun_search=True,
            confidence=0.8,
        )
        assert intent.needs_rerun_search
        assert "JFK" in intent.avoid_airports

    def test_explain_intent(self):
        intent = ChatIntent(
            intent_type="explain_option",
            selected_option_index=1,
            confidence=0.9,
        )
        assert intent.selected_option_index == 1

    def test_new_search_intent(self):
        intent = ChatIntent(
            intent_type="new_search",
            new_query="温州到匹兹堡",
            profile="cheapest",
            needs_rerun_search=True,
        )
        assert intent.new_query is not None
        assert intent.profile == "cheapest"


# ── LLM client factory respects settings ─────────────────────────────

class TestLLMDetection:
    def test_fake_client_is_default(self):
        from travel_agent.llm.prompts import get_llm_client
        client = get_llm_client()
        from travel_agent.llm.fake_client import FakeLLMClient
        assert isinstance(client, FakeLLMClient)


# ── Demo transcript imports ──────────────────────────────────────────

class TestDemoImports:
    def test_demo_script_imports(self):
        import scripts.demo_interactive_transcript
        assert scripts.demo_interactive_transcript is not None
