from __future__ import annotations

import pytest
import httpx

from travel_agent.llm.deepseek_client import DeepSeekRequirementAgent
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.tools.http_client import HttpClient
from travel_agent.tools.tool_router import ToolRouter


def session() -> LLMFirstChatSession:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline test transport", request=request)

    return LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(FakeRequirementLLM()),
        tool_router=ToolRouter(
            http_client=HttpClient(
                transport=httpx.MockTransport(offline),
                retries=0,
                backoff_seconds=0,
            )
        ),
    )


@pytest.mark.asyncio
async def test_eval_multiturn_chengdu_austin_date_and_budget_state():
    chat = session()
    first = await chat.handle_user_message("我想从成都飞奥斯丁")
    assert first.contract.trip.origin_airport == "TFU"
    assert first.contract.trip.destination_airport == "AUS"
    assert first.update.next_action == "ask_clarification"
    assert first.contract.pending.expected_answer_type == "departure_date_or_window"

    second = await chat.handle_user_message("六月初，越便宜越好")
    assert second.contract.time.departure_window_text == "六月初"
    assert second.contract.ranking.profile == "cheapest"
    assert second.pipeline_result is not None
    assert all(not offer.booking_available for offer in second.pipeline_result.offers)
    assert all(not offer.bookable and not offer.is_real_price for offer in second.pipeline_result.offers)
    assert second.contract.pending.pending_question is None


@pytest.mark.asyncio
async def test_eval_ambiguous_route_never_guesses_direction():
    result = await session().handle_user_message("成都奥斯丁")
    assert result.update.next_action == "ask_clarification"
    assert result.contract.trip.origin_airport is None
    assert result.contract.trip.destination_airport is None


@pytest.mark.asyncio
async def test_eval_tool_request_shapes_without_calling_network():
    chat = session()
    await chat.handle_user_message("我想从成都飞奥斯丁")
    weather = await chat.handle_user_message("目的地天气怎么样？")
    assert weather.update.next_action == "tool_query"
    assert weather.update.tool_requests[0].tool_name == "weather"

    airport = await chat.handle_user_message("奥斯丁机场是哪个？")
    assert airport.update.tool_requests[0].tool_name == "airport_lookup"
    assert airport.tool_results[0]["source"] == "local_airport_data"

    time = await chat.handle_user_message("奥斯丁现在几点？")
    assert time.update.tool_requests[0].tool_name == "time"

    currency = await chat.handle_user_message("100美元是多少人民币？")
    assert currency.update.tool_requests[0].tool_name == "currency"
    assert currency.update.tool_requests[0].arguments == {
        "amount": 100.0,
        "from_currency": "USD",
        "to_currency": "CNY",
    }


@pytest.mark.asyncio
async def test_eval_pet_constraint_add_then_cancel_marks_history_inactive():
    chat = session()
    added = await chat.handle_user_message("我想带狗一起去")
    assert any(pet.active for pet in added.contract.companions.pets)
    assert any(item.type == "pet_companion" and item.active for item in added.contract.constraints.hard_constraints)

    cancelled = await chat.handle_user_message("算了，不带狗了")
    assert all(not pet.active for pet in cancelled.contract.companions.pets)
    assert any(item.type == "pet_companion" and not item.active for item in cancelled.contract.constraints.hard_constraints)
    assert any(item.category == "pet_travel" and not item.active for item in cancelled.contract.special_requirements)


@pytest.mark.asyncio
async def test_eval_budget_and_general_flight_preferences():
    chat = session()
    budget = await chat.handle_user_message("我预算低一点")
    assert budget.contract.budget.preference == "lower"
    assert budget.contract.budget.priority == "high"

    red_eye = await chat.handle_user_message("我不想坐红眼航班")
    assert red_eye.contract.preferences.avoid_red_eye is True

    nonstop = await chat.handle_user_message("最好不要转机")
    assert nonstop.contract.preferences.nonstop_preferred is True
    assert nonstop.contract.preferences.max_stops == 0


@pytest.mark.asyncio
async def test_eval_debug_mode_shows_structured_turn_and_tool_metadata_only_when_enabled():
    debug_chat = session()
    debug_chat.debug = True
    await debug_chat.handle_user_message("我想从成都飞奥斯丁")
    debug_result = await debug_chat.handle_user_message("目的地天气怎么样？")
    assert "[debug] intent:" in debug_result.message
    assert "[debug] contract_diff:" in debug_result.message
    assert "[debug] next_action: tool_query" in debug_result.message
    assert "[debug] tool_request:" in debug_result.message
    assert "- status: unavailable" in debug_result.message

    normal_result = await session().handle_user_message("奥斯丁机场是哪个？")
    assert "[debug]" not in normal_result.message
