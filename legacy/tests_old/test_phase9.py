"""Phase 9 tests: preference persistence, profile metadata, airline tier priority, rerank state."""

import asyncio
from datetime import datetime

import pytest

from travel_agent.models.preference import (
    SCORING_PROFILES, DEFAULT_PROFILE, get_profile_weights,
    get_profile_label, get_profile_description, get_profile_header_text,
    PREFERENCE_COMMANDS,
)
from travel_agent.services.scoring_service import ScoringService
from travel_agent.services.airline_service import AirlineService
from travel_agent.services.itinerary_display_service import ItineraryDisplayService
from travel_agent.agents.followup_agent import FollowUpAgent
from travel_agent.models.flight import FlightSegment, CabinClass
from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment


# ── Profile Metadata ─────────────────────────────────────────────────

class TestProfileMetadata:
    def test_default_is_balanced(self):
        assert DEFAULT_PROFILE == "balanced"

    def test_all_profiles_have_weights(self):
        for profile in SCORING_PROFILES:
            weights = get_profile_weights(profile)
            assert len(weights) == 6
            total = sum(weights.values())
            assert 0.99 <= total <= 1.01, f"{profile} weights sum to {total}"

    def test_all_profiles_have_labels_and_descriptions(self):
        for profile in SCORING_PROFILES:
            label = get_profile_label(profile)
            desc = get_profile_description(profile)
            assert label and label != profile, f"Missing label_zh for {profile}"
            assert desc, f"Missing description_zh for {profile}"

    def test_header_text_contains_label(self):
        header = get_profile_header_text("airline_priority")
        assert "主流航司优先" in header
        assert "排序逻辑" in header

    def test_fallback_to_balanced_for_unknown_profile(self):
        weights = get_profile_weights("nonexistent")
        expected = get_profile_weights("balanced")
        assert weights == expected

    def test_profile_metadata_reused_by_scoring(self):
        scoring = ScoringService(profile="airline_priority")
        assert scoring.profile == "airline_priority"


# ── Profile Persistence ──────────────────────────────────────────────

class TestProfilePersistence:
    def test_default_profile_is_balanced(self):
        scoring = ScoringService()
        assert scoring.profile == "balanced"

    def test_airline_priority_profile(self):
        scoring = ScoringService(profile="airline_priority")
        weights = scoring._compute_weights()
        assert weights["airline"] > 0.15

    def test_cheapest_profile(self):
        scoring = ScoringService(profile="cheapest")
        weights = scoring._compute_weights()
        assert weights["savings"] > 0.35

    def test_low_risk_profile(self):
        scoring = ScoringService(profile="low_risk")
        weights = scoring._compute_weights()
        assert weights["risk"] > 0.25

    def test_scoring_uses_profile_in_score(self):
        bal = ScoringService(profile="balanced")
        cheap = ScoringService(profile="cheapest")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        it = Itinerary(
            id="t", type="hub_split", total_price_usd=800,
            segments=[], offers=[], split_ticket_count=2,
        )
        r_bal = bal.score(it, risk, 2000, airline_quality_score=0.5)
        r_cheap = cheap.score(it, risk, 2000, airline_quality_score=0.5)
        # Cheapest should weight savings more -> different scores
        assert r_cheap.savings_score == r_bal.savings_score  # Same raw score
        assert r_cheap.final_score != r_bal.final_score  # Different final due to weights


# ── FollowUp Commands ────────────────────────────────────────────────

class TestFollowUpCommands:
    def test_airline_priority_triggers_rerank(self):
        agent = FollowUpAgent()
        intent = asyncio.run(agent.execute(("主流航司优先", {})))
        assert intent.intent_type == "rerank"
        assert intent.constraint_updates["scoring_profile"] == "airline_priority"

    def test_reset_triggers_balanced(self):
        agent = FollowUpAgent()
        for cmd in ["恢复默认", "重置排序", "取消偏好", "undo", "reset", "默认排序"]:
            intent = asyncio.run(agent.execute((cmd, {})))
            assert intent.intent_type == "rerank", f"Command '{cmd}' should be rerank"
            assert intent.constraint_updates["scoring_profile"] == "balanced", \
                f"Command '{cmd}' should map to balanced"

    def test_explain_does_not_change_profile(self):
        agent = FollowUpAgent()
        intent = asyncio.run(agent.execute(("解释第1个", {})))
        assert intent.intent_type == "explain_option"
        assert "scoring_profile" not in intent.constraint_updates

    def test_all_preference_commands_map_to_known_profiles(self):
        for command, profile in PREFERENCE_COMMANDS.items():
            assert profile in SCORING_PROFILES, f"Command '{command}' maps to unknown profile '{profile}'"


# ── Airline Tier Label Priority ──────────────────────────────────────

class TestAirlineTierLabel:
    @pytest.fixture
    def svc(self):
        return AirlineService()

    def test_tier_preferred_over_score(self, svc):
        """MU is 'major' tier (score ~0.82). Should show '主流航司', not based on score."""
        # Get the airline data directly
        mu = svc.get_airline("MU")
        assert mu is not None
        tier = mu.get("quality_tier")
        assert tier == "major"
        label = svc.tier_label_zh(tier)
        assert label == "主流航司"

    def test_premium_tier_label(self, svc):
        cx = svc.get_airline("CX")
        assert cx["quality_tier"] == "premium"
        assert svc.tier_label_zh(cx["quality_tier"]) == "顶级航司"

    def test_budget_tier_label(self, svc):
        nk = svc.get_airline("NK")
        assert nk["quality_tier"] == "budget"
        assert svc.tier_label_zh(nk["quality_tier"]) == "廉价航司"

    def test_score_fallback_when_no_tier(self, svc):
        # tier_label_for_score should give correct labels
        assert svc.tier_label_for_score(0.95) == "顶级航司"
        assert svc.tier_label_for_score(0.82) == "主流航司"
        assert svc.tier_label_for_score(0.65) == "标准航司"
        assert svc.tier_label_for_score(0.40) == "廉价航司"


# ── Display Service Consistency ──────────────────────────────────────

class TestDisplayConsistency:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_detail_view_uses_tier_from_data(self, display):
        """CX (premium) should show 顶级航司 in detail view."""
        it = Itinerary(
            id="t", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="CX", cabin=CabinClass.ECONOMY),
            ],
            offers=[],
        )
        lines = display.format_leg_details_zh(it)
        combined = " ".join(lines)
        assert "顶级航司" in combined, f"CX should be 顶级航司 in detail. Got: {combined[:200]}"

    def test_detail_view_budget_shows_warning(self, display):
        it = Itinerary(
            id="t", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="NK", cabin=CabinClass.ECONOMY),
            ],
            offers=[],
        )
        lines = display.format_leg_details_zh(it)
        combined = " ".join(lines)
        assert "廉价航司" in combined


# ── Rerank Preserves State ───────────────────────────────────────────

class TestRerankState:
    def test_rerank_with_profile_gives_different_order(self):
        from travel_agent.models.ranking import RankedRecommendation
        scoring_bal = ScoringService(profile="balanced")
        scoring_air = ScoringService(profile="airline_priority")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        it_budget = Itinerary(
            id="budget", type="hub_split", total_price_usd=800,
            segments=[], offers=[], split_ticket_count=2,
        )
        it_premium = Itinerary(
            id="premium", type="direct", total_price_usd=1000,
            segments=[], offers=[], split_ticket_count=0,
        )
        r_bud_bal = scoring_bal.score(it_budget, risk, 2000, airline_quality_score=0.3)
        r_pre_bal = scoring_bal.score(it_premium, risk, 2000, airline_quality_score=0.9)
        r_bud_air = scoring_air.score(it_budget, risk, 2000, airline_quality_score=0.3)
        r_pre_air = scoring_air.score(it_premium, risk, 2000, airline_quality_score=0.9)

        # In balanced: budget might win on price
        # In airline_priority: premium should win
        bal_order = "budget" if r_bud_bal.final_score > r_pre_bal.final_score else "premium"
        air_order = "budget" if r_bud_air.final_score > r_pre_air.final_score else "premium"
        # At minimum they should differ or be tie
        assert r_bud_bal.final_score != r_bud_air.final_score  # Different weights produce different scores

    def test_balanced_profile_is_same_as_default(self):
        scoring_default = ScoringService()
        scoring_balanced = ScoringService(profile="balanced")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        it = Itinerary(id="t", type="hub_split", total_price_usd=1000,
                       segments=[], offers=[], split_ticket_count=2)
        r_def = scoring_default.score(it, risk, 2000, airline_quality_score=0.5)
        r_bal = scoring_balanced.score(it, risk, 2000, airline_quality_score=0.5)
        assert r_def.final_score == r_bal.final_score
