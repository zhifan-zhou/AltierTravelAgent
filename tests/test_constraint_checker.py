from __future__ import annotations

from datetime import datetime

from travel_agent.contract.compiler import ConstraintCompiler
from travel_agent.contract.models import PetCompanion, SpecialRequirement, TravelRequirementContract
from travel_agent.pipeline.types import (
    FlightSegment,
    Itinerary,
    PipelineResult,
    Recommendation,
    RiskAssessment,
)
from travel_agent.planning.constraint_checker import ConstraintChecker
from travel_agent.planning.models import CostEstimate
from travel_agent.tools.base import ToolResult


def _contract() -> TravelRequirementContract:
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "TFU"
    contract.trip.destination_airport = "AUS"
    return contract


def _pipeline(contract: TravelRequirementContract, *, red_eye: bool = True) -> PipelineResult:
    segments = [
        FlightSegment(
            origin="TFU",
            destination="DFW",
            departure_time=datetime(2026, 6, 1, 23, 30) if red_eye else datetime(2026, 6, 1, 10, 0),
        ),
        FlightSegment(origin="DFW", destination="AUS", departure_time=datetime(2026, 6, 2, 8, 0)),
    ]
    itinerary = Itinerary(
        id="demo",
        route_type="hub_split",
        route=["TFU", "DFW", "AUS"],
        offers=[],
        segments=segments,
        total_price_usd=900,
        total_estimated_time_hours=20,
    )
    rec = Recommendation(
        rank=1,
        recommendation_type="demo",
        itinerary=itinerary,
        score=1,
        savings_vs_baseline_usd=0,
        risk=RiskAssessment(risk_score=0.2, risk_level="low"),
        airline_quality_score=0.8,
        reason_zh="demo",
    )
    return PipelineResult(
        contract=contract,
        exclusions=ConstraintCompiler().compile(contract).exclusions,
        recommendations=[rec],
    )


def test_constraint_checker_flags_pet_red_eye_nonstop_and_transfer_risks():
    contract = _contract()
    contract.companions.pets = [PetCompanion(kind="cat")]
    contract.preferences.avoid_red_eye = True
    contract.preferences.nonstop_preferred = True
    findings = ConstraintChecker().check(contract, pipeline_result=_pipeline(contract)).findings
    categories = {item.category for item in findings}
    assert {"pet", "red_eye", "nonstop", "airport_transfer", "documents"} <= categories
    assert any(item.category == "red_eye" and item.level == "conflict" for item in findings)
    assert any(item.category == "nonstop" and item.level == "conflict" for item in findings)


def test_constraint_checker_flags_budget_and_live_weather_risk():
    contract = _contract()
    contract.budget.amount = 300
    contract.budget.currency = "USD"
    estimate = CostEstimate(total_min=500, total_max=700, currency="USD")
    weather = ToolResult(
        tool_name="weather",
        status="ok",
        data={"daily": [{"precipitation_probability_max": 80, "temperature_max": 37}]},
        message="ok",
        source="open_meteo",
        is_live=True,
    )
    findings = ConstraintChecker().check(contract, cost_estimate=estimate, weather_result=weather).findings
    assert any(item.category == "budget" and item.level == "conflict" for item in findings)
    assert sum(item.category == "weather" for item in findings) == 2


def test_constraint_checker_uses_official_source_reminders_for_documents_and_accessibility():
    contract = _contract()
    contract.special_requirements = [SpecialRequirement(category="accessibility", active=True)]
    text = " ".join(item.message for item in ConstraintChecker().check(contract).findings)
    assert "官方签证" in text
    assert "无障碍" in text
    assert "保证" not in text
