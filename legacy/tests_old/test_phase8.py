"""Phase 8 tests: tier labels, scoring profiles, reranking, ground transfer, disclaimer, detail view."""

import asyncio
from datetime import datetime

import pytest

from travel_agent.models.flight import FlightSegment, FlightOffer, CabinClass
from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment
from travel_agent.services.airline_service import AirlineService
from travel_agent.services.scoring_service import ScoringService
from travel_agent.models.preference import SCORING_PROFILES, get_profile_label
from travel_agent.services.itinerary_display_service import ItineraryDisplayService
from travel_agent.agents.followup_agent import FollowUpAgent


# ── Airline tier labels ──────────────────────────────────────────────

class TestAirlineTierLabels:
    @pytest.fixture
    def svc(self):
        return AirlineService()

    def test_premium_score_gets_top_tier_label(self, svc):
        label = svc.tier_label_for_score(0.95)
        assert label == "顶级航司"

    def test_major_score_gets_major_label(self, svc):
        label = svc.tier_label_for_score(0.82)
        assert label == "主流航司"

    def test_standard_score_gets_standard_label(self, svc):
        label = svc.tier_label_for_score(0.65)
        assert label == "标准航司"

    def test_budget_score_gets_budget_label(self, svc):
        label = svc.tier_label_for_score(0.40)
        assert label == "廉价航司"

    def test_tier_label_map_zh(self, svc):
        assert svc.tier_label_zh("premium") == "顶级航司"
        assert svc.tier_label_zh("major") == "主流航司"
        assert svc.tier_label_zh("standard") == "标准航司"
        assert svc.tier_label_zh("budget") == "廉价航司"


# ── Scoring profiles ─────────────────────────────────────────────────

def _make_itinerary(price: float, risk_score: float = 0.5, airline_q: float = 0.5,
                    split: bool = True, access_cost: float = 0, segments: int = 2) -> Itinerary:
    segs = [
        FlightSegment(origin="PVG", destination="JFK",
                       departure_time=datetime(2026,6,15,17,0),
                       arrival_time=datetime(2026,6,15,20,0),
                       airline="MU", cabin=CabinClass.ECONOMY),
    ]
    if segments >= 2:
        segs.append(FlightSegment(origin="JFK", destination="PIT",
                                   departure_time=datetime(2026,6,16,8,0),
                                   arrival_time=datetime(2026,6,16,9,30),
                                   airline="DL", cabin=CabinClass.ECONOMY))
    return Itinerary(
        id=f"test-{price}", type="hub_split" if split else "direct",
        segments=segs, total_price_usd=price,
        total_access_cost_usd=access_cost,
        total_estimated_time_hours=20,
        number_of_segments=len(segs),
        split_ticket_count=2 if split else 0,
    )


class TestScoringProfiles:
    def test_airline_priority_boosts_premium_airline(self):
        scoring = ScoringService(profile="airline_priority")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        # Premium airline
        rec_high = scoring.score(_make_itinerary(1000), risk, 2000, airline_quality_score=0.9)
        # Budget airline with same price
        rec_budget = scoring.score(_make_itinerary(1000), risk, 2000, airline_quality_score=0.4)
        assert rec_high.final_score > rec_budget.final_score

    def test_cheapest_still_prefers_low_price(self):
        scoring = ScoringService(profile="cheapest")
        risk_low = RiskAssessment(risk_level="low", risk_score=0.1)
        risk_med = RiskAssessment(risk_level="medium", risk_score=0.5)
        # Cheap + low risk + direct vs expensive + medium risk + split + budget airline
        rec_cheap = scoring.score(_make_itinerary(600, risk_score=0.1, split=False, segments=1), risk_low, 2000, airline_quality_score=0.6)
        rec_expensive = scoring.score(_make_itinerary(1400, risk_score=0.5, split=True, segments=2, access_cost=100), risk_med, 2000, airline_quality_score=0.35)
        assert rec_cheap.final_score > rec_expensive.final_score, \
            f"cheap={rec_cheap.final_score:.3f} expensive={rec_expensive.final_score:.3f}"

    def test_low_risk_penalizes_split_tickets(self):
        scoring_balanced = ScoringService(profile="balanced")
        scoring_low = ScoringService(profile="low_risk")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)

        split = _make_itinerary(1000, split=True)
        direct = _make_itinerary(1100, split=False)

        # In low_risk mode, direct should gain advantage over split
        rec_split_bal = scoring_balanced.score(split, risk, 2000)
        rec_direct_bal = scoring_balanced.score(direct, risk, 2000)
        rec_split_low = scoring_low.score(split, risk, 2000)
        rec_direct_low = scoring_low.score(direct, risk, 2000)

        # Low risk should improve direct relative to split
        bal_diff = rec_direct_bal.final_score - rec_split_bal.final_score
        low_diff = rec_direct_low.final_score - rec_split_low.final_score
        assert low_diff > bal_diff, f"low_risk should favor direct more. bal_diff={bal_diff:.3f} low_diff={low_diff:.3f}"

    def test_low_risk_penalizes_ground_transfer(self):
        scoring_balanced = ScoringService(profile="balanced")
        scoring_low = ScoringService(profile="low_risk")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)

        with_access = _make_itinerary(1000, access_cost=200)
        without = _make_itinerary(1000, access_cost=0)

        rec_with_bal = scoring_balanced.score(with_access, risk, 2000)
        rec_with_low = scoring_low.score(with_access, risk, 2000)
        rec_without_low = scoring_low.score(without, risk, 2000)

        # Ground transfer should get penalized more in low_risk
        assert rec_without_low.final_score > rec_with_low.final_score

    def test_all_profile_labels_have_chinese_names(self):
        from travel_agent.models.preference import get_profile_label
        for profile in SCORING_PROFILES:
            label = get_profile_label(profile)
            assert label, f"Missing label for {profile}"
            assert label != profile, f"Label for {profile} should be Chinese, not key"


# ── FollowUp agent reranking commands ────────────────────────────────

class TestFollowUpReranking:
    def test_airline_priority_command(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("主流航司优先", {})))
        assert result.intent_type == "rerank"
        assert result.constraint_updates.get("scoring_profile") == "airline_priority"

    def test_no_budget_command(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("不要廉航", {})))
        assert result.intent_type == "rerank"
        assert result.constraint_updates.get("scoring_profile") == "airline_priority"

    def test_low_risk_command(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("少折腾", {})))
        assert result.intent_type == "rerank"
        assert result.constraint_updates.get("scoring_profile") == "low_risk"

    def test_cheapest_command(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("只要便宜", {})))
        assert result.intent_type == "rerank"
        assert result.constraint_updates.get("scoring_profile") == "cheapest"

    def test_fastest_command(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("时间短一点", {})))
        assert result.intent_type == "rerank"
        assert result.constraint_updates.get("scoring_profile") == "fastest"

    def test_explain_command_still_works(self):
        agent = FollowUpAgent()
        result = asyncio.run(agent.execute(("解释第1个", {})))
        assert result.intent_type == "explain_option"
        assert result.selected_option_index == 1


# ── Ground transfer display ──────────────────────────────────────────

class TestGroundTransferDisplay:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_ground_transfer_uses_ground_arrow(self, display):
        """Itinerary with access cost should use ⇢ for ground leg."""
        it = Itinerary(
            id="test", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
            ],
            origin_airport="WNZ", destination_airport="JFK",
            total_access_cost_usd=50,
        )
        route = display.format_route_codes(it)
        assert "⇢" in route, f"Ground arrow ⇢ should appear in route with access cost, got: {route}"

    def test_flight_only_uses_flight_arrow(self, display):
        """Direct flight should only use → arrow."""
        it = Itinerary(
            id="test", type="direct",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
            ],
            origin_airport="PVG", destination_airport="JFK",
        )
        route = display.format_route_codes(it)
        assert "⇢" not in route, f"Flight-only route should not have ground arrow: {route}"
        assert "→" in route


# ── Detail view ──────────────────────────────────────────────────────

class TestDetailView:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_detail_view_includes_ground_warning(self, display):
        it = Itinerary(
            id="test", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="MU", cabin=CabinClass.ECONOMY),
            ],
            origin_airport="WNZ", destination_airport="JFK",
            total_access_cost_usd=50,
        )
        lines = display.format_leg_details_zh(it)
        ground_line = [l for l in lines if "接驳" in l or "地面" in l or "自行安排" in l]
        assert len(ground_line) >= 1, f"Detail should show ground transfer warning"
        combined = " ".join(lines)
        assert "自行安排" in combined, "Should mention self-arrangement"


# ── Recommendation reason ────────────────────────────────────────────

class TestRecommendationReason:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_cheapest_reason_mentions_savings(self, display):
        from travel_agent.models.ranking import RankedRecommendation
        it = _make_itinerary(900, split=True, access_cost=50)
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        rec = RankedRecommendation(
            itinerary=it, risk_assessment=risk,
            savings_vs_baseline_usd=200, savings_percentage=18,
            recommendation_type="cheapest_reasonable", final_score=0.8,
        )
        reason = display.recommendation_reason(rec)
        assert "便宜" in reason or "最低价" in reason

    def test_low_risk_reason_is_meanr(self, display):
        from travel_agent.models.ranking import RankedRecommendation
        it = _make_itinerary(1100, split=False)
        risk = RiskAssessment(risk_level="low", risk_score=0.1)
        rec = RankedRecommendation(
            itinerary=it, risk_assessment=risk,
            recommendation_type="lowest_risk", final_score=0.7,
        )
        reason = display.recommendation_reason(rec)
        assert "折腾" in reason or "风险" in reason or "拆分少" in reason

    def test_airline_priority_mode_shows_tier_in_reason(self, display):
        from travel_agent.models.ranking import RankedRecommendation
        it = _make_itinerary(1000, split=False)
        risk = RiskAssessment(risk_level="low", risk_score=0.1)
        rec = RankedRecommendation(
            itinerary=it, risk_assessment=risk,
            savings_vs_baseline_usd=60, savings_percentage=5,
            airline_quality_score=0.9,
            recommendation_type="best_overall", final_score=0.8,
        )
        reason = display.recommendation_reason(rec)
        assert "顶级" in reason or "主流" in reason or "标准" in reason or "航司品质" in reason
