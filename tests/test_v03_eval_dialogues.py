from __future__ import annotations

import pytest

from travel_agent.rendering.response_streamer import ResponseStreamer
from tests.helpers_v03 import make_v03_session


@pytest.mark.asyncio
async def test_direct_three_day_austin_low_budget_itinerary():
    result = await make_v03_session().handle_user_message("帮我安排奥斯丁三天行程，预算低一点")
    assert result.user_response.response_type == "itinerary"
    assert result.contract.time.duration_days == 3
    assert result.contract.budget.preference == "lower"
    assert result.message.count("Day ") == 3
    assert "实时门票" in result.message


@pytest.mark.asyncio
async def test_multiturn_route_date_then_itinerary_preserves_state():
    chat = make_v03_session()
    await chat.handle_user_message("我想从成都飞奥斯丁")
    await chat.handle_user_message("六月初，越便宜越好")
    result = await chat.handle_user_message("帮我安排三天行程")
    assert result.contract.trip.destination_airport == "AUS"
    assert result.contract.time.departure_window_text == "六月初"
    assert result.contract.ranking.profile == "cheapest"
    assert result.user_response.response_type == "itinerary"


@pytest.mark.asyncio
async def test_rain_aware_itinerary_and_unavailable_weather_honesty():
    rainy = await make_v03_session(rain=True).handle_user_message("帮我安排奥斯丁三天行程")
    assert "降水概率较高" in rainy.message
    unavailable = await make_v03_session(fail=True).handle_user_message("帮我安排奥斯丁三天行程")
    assert "实时天气暂时不可用" in unavailable.message
    assert "预报摘要" not in unavailable.message


@pytest.mark.asyncio
async def test_pet_constraint_flows_into_itinerary_for_more_than_one_pet_phrase():
    for phrase in ["我想带狗一起去", "这次带猫同行"]:
        chat = make_v03_session()
        await chat.handle_user_message("我想从成都飞奥斯丁")
        await chat.handle_user_message(phrase)
        result = await chat.handle_user_message("帮我安排三天行程")
        assert "宠物政策" in result.message
        assert any(pet.active for pet in result.contract.companions.pets)


@pytest.mark.asyncio
async def test_cost_estimate_labels_estimate_and_mock_boundaries():
    chat = make_v03_session()
    result = await chat.handle_user_message("估算一下三天奥斯丁大概要多少钱")
    assert result.user_response.response_type == "cost_estimate"
    assert "rough estimate" in result.message
    assert "不是实时报价" in result.message
    assert "未将机票计入总计" in result.message


@pytest.mark.asyncio
async def test_preferences_and_streamed_final_response_are_structured_and_clean():
    chat = make_v03_session()
    await chat.handle_user_message("我想从成都飞奥斯丁")
    red_eye = await chat.handle_user_message("我不想坐红眼航班")
    assert red_eye.contract.preferences.avoid_red_eye
    nonstop = await chat.handle_user_message("最好不要转机")
    assert nonstop.contract.preferences.nonstop_preferred
    itinerary = await chat.handle_user_message("帮我安排三天行程")
    streamed = "".join(ResponseStreamer(chunk_size=10).stream_response(itinerary.user_response))
    assert streamed == itinerary.message
    assert "[debug]" not in streamed
