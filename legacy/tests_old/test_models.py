"""Test Pydantic model validation."""

from datetime import date

from travel_agent.models.user_request import (
    CabinClass,
    DateWindow,
    UserTravelRequest,
    HardConstraints,
    SoftConstraints,
    SearchConstraints,
)
from travel_agent.models.airport import Airport, HubSplitPlan, HubPair, NearbyHub
from travel_agent.models.risk import RiskAssessment
from travel_agent.models.ranking import RankedRecommendation, RankingOutput


class TestModels:
    def test_date_window_default(self):
        dw = DateWindow()
        assert dw.flexible is True
        assert dw.start_date is None

    def test_user_travel_request_defaults(self):
        req = UserTravelRequest(origin_text="宁波", destination_text="匹兹堡")
        assert req.cabin == CabinClass.ECONOMY
        assert req.passengers == 1
        assert req.accepts_nearby_hubs is False

    def test_airport_creation(self):
        a = Airport(code="PVG", name="Shanghai Pudong", city="Shanghai", city_cn="上海")
        assert a.code == "PVG"
        assert a.is_international_hub is False

    def test_hub_split_plan_defaults(self):
        plan = HubSplitPlan(origin_airport_code="NGB", destination_airport_code="PIT")
        assert plan.origin_hubs == []
        assert plan.candidate_hub_pairs == []

    def test_risk_assessment_defaults(self):
        ra = RiskAssessment()
        assert ra.risk_level == "low"
        assert ra.risk_score == 0.0

    def test_ranking_output(self):
        output = RankingOutput()
        assert output.rankings == []
        assert output.best_overall is None
