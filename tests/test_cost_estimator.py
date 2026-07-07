from __future__ import annotations

from datetime import UTC, datetime

from travel_agent.contract.compiler import ConstraintCompiler
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.pipeline.types import FlightOffer, PipelineResult
from travel_agent.planning.cost_estimator import CostEstimator
from travel_agent.tools.base import ToolResult


def _contract() -> TravelRequirementContract:
    contract = TravelRequirementContract()
    contract.trip.destination_airport = "AUS"
    contract.time.duration_days = 3
    contract.budget.preference = "lower"
    return contract


def _pipeline(contract: TravelRequirementContract) -> PipelineResult:
    offer = FlightOffer(
        id="demo",
        leg_type="main",
        origin="TFU",
        destination="AUS",
        segments=[],
        total_price_usd=900,
        source="mock_demo",
    )
    return PipelineResult(
        contract=contract,
        exclusions=ConstraintCompiler().compile(contract).exclusions,
        offers=[offer],
    )


def test_cost_estimator_labels_mock_flight_and_rough_local_costs():
    contract = _contract()
    estimate = CostEstimator().estimate(contract, pipeline_result=_pipeline(contract))
    flight = estimate.items[0]
    assert flight.source_type == "mock_demo"
    assert "demo/mock only" in flight.note
    assert all(item.source_type == "estimate" for item in estimate.items[1:])
    assert estimate.total_min < estimate.total_max


def test_cost_estimator_supports_live_currency_conversion():
    contract = _contract()
    contract.budget.currency = "CNY"
    conversion = ToolResult(
        tool_name="currency",
        status="ok",
        data={"rate": 6.8},
        message="ok",
        source="frankfurter",
        fetched_at=datetime.now(UTC),
        is_live=True,
    )
    estimate = CostEstimator().estimate(contract, conversion_result=conversion)
    assert estimate.currency == "CNY"
    assert all(item.currency == "CNY" for item in estimate.items)
    assert estimate.sources[0].source == "frankfurter"


def test_cost_estimator_does_not_invent_flight_price_when_missing():
    estimate = CostEstimator().estimate(_contract())
    flight = estimate.items[0]
    assert flight.source_type == "unknown"
    assert flight.amount_min is None
    assert "未将机票计入总计" in flight.note
