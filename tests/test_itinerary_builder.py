from __future__ import annotations

from datetime import UTC, datetime

from travel_agent.contract.models import PetCompanion, TravelRequirementContract
from travel_agent.planning.itinerary_builder import ItineraryBuilder
from travel_agent.tools.base import ToolResult


def _contract() -> TravelRequirementContract:
    contract = TravelRequirementContract()
    contract.trip.destination_text = "奥斯丁"
    contract.trip.destination_airport = "AUS"
    contract.time.duration_days = 3
    contract.budget.preference = "lower"
    return contract


def _rain_weather() -> ToolResult:
    return ToolResult(
        tool_name="weather",
        status="ok",
        data={
            "daily": [
                {
                    "date": "2026-07-06",
                    "summary": "中雨",
                    "temperature_max": 36,
                    "temperature_min": 24,
                    "precipitation_probability_max": 80,
                }
            ]
        },
        message="rain",
        source="open_meteo",
        fetched_at=datetime.now(UTC),
        is_live=True,
    )


def test_itinerary_builder_creates_three_day_budget_weather_aware_plan():
    plan = ItineraryBuilder().build(_contract(), weather_result=_rain_weather())
    assert plan.destination == "奥斯丁"
    assert plan.duration_days == 3
    assert len(plan.days) == 3
    assert all(day.budget_level == "low" for day in plan.days)
    assert "降水概率较高" in " ".join(plan.days[0].weather_considerations)
    assert "高温" in " ".join(plan.days[0].weather_considerations)
    assert plan.sources[0].source == "open_meteo"


def test_itinerary_builder_defaults_to_three_days_and_does_not_invent_weather():
    contract = _contract()
    contract.time.duration_days = None
    unavailable = ToolResult(
        tool_name="weather",
        status="unavailable",
        message="unavailable",
        source="open_meteo",
        error_code="network_error",
    )
    plan = ItineraryBuilder().build(contract, weather_result=unavailable)
    assert plan.duration_days == 3
    assert "先按 3 天草案" in " ".join(plan.assumptions)
    assert all(not day.weather_considerations for day in plan.days)
    assert "天气暂时不可用" in " ".join(plan.warnings)


def test_itinerary_builder_adds_generic_pet_policy_reminder():
    contract = _contract()
    contract.companions.pets = [PetCompanion(kind="cat", active=True)]
    plan = ItineraryBuilder().build(contract)
    text = plan.model_dump_json()
    assert "宠物同行" in text
    assert "官方宠物政策" in text
