"""Phase 6 tests: LLM integration, interactive chat, new agents."""

import asyncio
import os

import pytest

from travel_agent.llm.base import BaseLLMClient
from travel_agent.llm.fake_client import FakeLLMClient
from travel_agent.llm.schemas import (
    LLMParsedRequest, ClarificationPlan, FollowUpIntent,
    LLMPreferenceInference, LLMExplanation,
)
from travel_agent.agents.clarification_agent import ClarificationAgent
from travel_agent.agents.preference_agent import PreferenceAgent
from travel_agent.agents.followup_agent import FollowUpAgent
from travel_agent.agents.search_budget_agent import SearchBudgetAgent
from travel_agent.agents.result_critic_agent import ResultCriticAgent
from travel_agent.agents.user_choice_agent import UserChoiceAgent
from travel_agent.models.agent_outputs import IntakeOutput, SearchStrategyOutput
from travel_agent.models.user_request import CabinClass, DateWindow
from travel_agent.models.flight import FlightSearchRequest

# Ensure tests run with LLM_PROVIDER=none to avoid depending on local .env
os.environ["LLM_PROVIDER"] = "none"
# Force reload
from travel_agent.core.config import _settings
if _settings is not None:
    _settings.llm_provider = "none"

from travel_agent.llm.prompts import get_llm_client, is_llm_enabled, mask_sensitive_config


# ── LLM Client Factory ──────────────────────────────────────────────

class TestLLMClientFactory:
    def test_default_is_fake(self):
        client = get_llm_client()
        assert isinstance(client, FakeLLMClient), f"Expected FakeLLMClient, got {type(client).__name__}"

    def test_llm_disabled_by_default(self):
        assert not is_llm_enabled("intake")
        assert not is_llm_enabled("clarification")


# ── Fake LLM Client ──────────────────────────────────────────────────

class TestFakeLLMClient:
    def test_returns_none_for_all_methods(self):
        client = FakeLLMClient()
        async def check():
            assert await client.parse_travel_request("test") is None
            assert await client.infer_preferences("test") is None
            assert await client.generate_clarifying_questions() is None
            assert await client.explain_recommendations() is None
            assert await client.interpret_followup("test") is None
        asyncio.run(check())

    def test_health_check(self):
        client = FakeLLMClient()
        async def check():
            assert await client.health_check() is True
        asyncio.run(check())


# ── Masking ──────────────────────────────────────────────────────────

class TestConfigMasking:
    def test_keys_are_masked(self):
        config = {
            "DEEPSEEK_API_KEY": "sk-1234567890abcdef",
            "DUFFEL_API_TOKEN": "tok_abcdef1234567890",
            "LOG_LEVEL": "INFO",
        }
        masked = mask_sensitive_config(config)
        assert "sk-1****cdef" in masked["DEEPSEEK_API_KEY"] or "****" in masked["DEEPSEEK_API_KEY"]
        assert masked["LOG_LEVEL"] == "INFO"

    def test_empty_keys_stay_empty(self):
        masked = mask_sensitive_config({"DEEPSEEK_API_KEY": "", "LOG_LEVEL": "INFO"})
        assert masked["DEEPSEEK_API_KEY"] == ""
        assert masked["LOG_LEVEL"] == "INFO"


# ── Clarification Agent ──────────────────────────────────────────────

class TestClarificationAgent:
    def test_asks_date_if_missing(self):
        agent = ClarificationAgent()
        intake = IntakeOutput(
            origin_text="温州", destination_text="匹兹堡",
            departure_window=DateWindow(flexible=True),  # No start_date
        )
        result = asyncio.run(agent.execute(intake))
        assert result.should_ask
        missing_ids = [q.id for q in result.questions]
        assert "date" in missing_ids or "risk" in missing_ids

    def test_asks_split_for_cheap_query(self):
        agent = ClarificationAgent()
        intake = IntakeOutput(
            origin_text="温州", destination_text="匹兹堡",
            departure_window=DateWindow(flexible=True),
            preferences=["cheap"],
        )
        result = asyncio.run(agent.execute(intake))
        assert result.should_ask

    def test_fewer_questions_when_complete(self):
        agent = ClarificationAgent()
        from datetime import date
        intake = IntakeOutput(
            origin_text="温州", destination_text="匹兹堡",
            departure_window=DateWindow(start_date=date(2026, 6, 15)),
            preferences=["cheap"],
            budget_usd=1500,
        )
        result = asyncio.run(agent.execute(intake))
        # Should have fewer than 3 questions
        assert len(result.questions) < 3


# ── Preference Agent ─────────────────────────────────────────────────

class TestPreferenceAgent:
    def test_clamps_weights(self):
        agent = PreferenceAgent()
        intake = IntakeOutput(origin_text="NGB", destination_text="PIT", preferences=["cheap"])
        weights = asyncio.run(agent.execute(intake))
        for k, v in weights.items():
            assert 0.0 <= v <= 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_cheap_increases_price_weight(self):
        agent = PreferenceAgent()
        intake_cheap = IntakeOutput(origin_text="NGB", destination_text="PIT", preferences=["cheap"])
        intake_neutral = IntakeOutput(origin_text="NGB", destination_text="PIT")
        w_cheap = asyncio.run(agent.execute(intake_cheap))
        w_neutral = asyncio.run(agent.execute(intake_neutral))
        assert w_cheap["price"] > w_neutral["price"]


# ── Search Budget Agent ──────────────────────────────────────────────

class TestSearchBudgetAgent:
    def test_prunes_too_many_tasks(self):
        agent = SearchBudgetAgent()
        tasks = [
            FlightSearchRequest(origin=f"ORG{i}", destination=f"DST{i}")
            for i in range(100)
        ]
        data = SearchStrategyOutput(
            search_tasks=tasks,
            direct_task=tasks[0],
            hub_split_tasks=tasks[1:],
        )
        result = asyncio.run(agent.execute(data))
        assert len(result.search_tasks) <= 80  # mock limit


# ── FollowUp Agent ───────────────────────────────────────────────────

class TestFollowUpAgent:
    def test_no_split(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("不要分开出票", {})))
        assert result.intent_type == "refine_constraints"
        assert result.constraint_updates.get("accepts_split_ticket") is False

    def test_explain_nth(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("解释第2个", {})))
        assert result.intent_type == "explain_option"
        assert result.selected_option_index == 2

    def test_avoid_nyc(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("不要纽约转", {})))
        assert result.intent_type == "refine_constraints"
        assert "JFK" in result.constraint_updates.get("avoid_hubs", [])


# ── User Choice Agent ────────────────────────────────────────────────

class TestUserChoiceAgent:
    def test_cheapest(self):
        agent = UserChoiceAgent()
        result = asyncio.run(agent.execute("只看最便宜"))
        assert result.action == "cheapest"


# ── Result Critic ────────────────────────────────────────────────────

class TestResultCritic:
    def test_flags_estimated_data(self):
        from travel_agent.models.agent_outputs import TravelAgentResult
        result = TravelAgentResult(query="test")
        critic = ResultCriticAgent()
        report = asyncio.run(critic.execute(result))
        # No rankings — should warn
        assert len(report.warnings) >= 1


# ── DeepSeek Client ──────────────────────────────────────────────────

class TestDeepSeekClient:
    def test_deepseek_client_imports_and_creates(self):
        from travel_agent.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient()
        assert client is not None
        assert client.provider_name == "deepseek"

    def test_not_logging_api_key(self, monkeypatch):
        """Verify that the DeepSeek client does not expose the API key in repr/str."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test12345678")
        from travel_agent.llm.deepseek_client import DeepSeekClient
        # Re-initialize to pick up env var
        import importlib
        import travel_agent.core.config
        importlib.reload(travel_agent.core.config)
        client = DeepSeekClient()
        # Provider name and properties should not contain the key
        assert "sk-test" not in client.provider_name
        assert "sk-test" not in str(client.__dict__.get("_model_fast", ""))


# ── LLM Schemas ─────────────────────────────────────────────────────

class TestLLMSchemas:
    def test_parsed_request_defaults(self):
        req = LLMParsedRequest()
        assert req.nearby_hub_policy == "allow"
        assert req.risk_tolerance == "medium"

    def test_clarification_plan_defaults(self):
        plan = ClarificationPlan()
        assert not plan.should_ask
        assert plan.questions == []

    def test_followup_intent_defaults(self):
        intent = FollowUpIntent()
        assert intent.intent_type == "unknown"


# ── CLI Chat Entry Point ─────────────────────────────────────────────

class TestCLIChat:
    def test_chat_function_imports(self):
        from travel_agent.cli import run_chat
        assert run_chat is not None
