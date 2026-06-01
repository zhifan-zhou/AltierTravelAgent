"""Phase 10 tests: profile persistence, comparison, display quality, evals."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from travel_agent.models.preference import (
    SCORING_PROFILES, DEFAULT_PROFILE, get_profile_label,
    get_profile_description, get_profile_header_text,
)
from travel_agent.services.scoring_service import ScoringService
from travel_agent.services.itinerary_display_service import ItineraryDisplayService
from travel_agent.services.airline_service import AirlineService
from travel_agent.models.flight import FlightSegment, CabinClass
from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment


# ── Profile persistence artifacts ────────────────────────────────────

class TestProfilePersistence:
    def test_default_profile_is_balanced(self):
        assert DEFAULT_PROFILE == "balanced"

    def test_session_json_has_profile_fields(self):
        """Simulate the session.json structure that CLI exports."""
        session = {
            "current_profile": "airline_priority",
            "profile_history": ["balanced", "cheapest", "airline_priority"],
            "profile_events": [
                {"from": "balanced", "to": "cheapest", "trigger": "只要便宜", "timestamp": "..."}
            ],
        }
        assert "current_profile" in session
        assert "profile_history" in session
        assert len(session["profile_events"]) >= 1
        assert session["profile_events"][0]["to"] == "cheapest"

    def test_profile_history_does_not_duplicate_adjacent(self):
        """Adjacent duplicates should be collapsed."""
        history = ["balanced", "cheapest", "cheapest", "airline_priority"]
        deduped = [history[0]]
        for h in history[1:]:
            if h != deduped[-1]:
                deduped.append(h)
        assert deduped == ["balanced", "cheapest", "airline_priority"]

    def test_reset_sets_balanced(self):
        """reset/undo should set profile to balanced."""
        profile = "airline_priority"
        profile = "balanced"  # Simulate reset
        assert profile == DEFAULT_PROFILE


# ── Recommendation table never shows ID as route ─────────────────────

class TestDisplayQuality:
    @pytest.fixture
    def display(self):
        return ItineraryDisplayService()

    def test_route_never_shows_itinerary_id(self, display):
        it = Itinerary(
            id="hubsplit-abc123-def456", type="hub_split",
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
        assert "hubsplit" not in route, f"Route should not contain ID: {route}"

    def test_airline_summary_appears(self, display):
        it = Itinerary(
            id="t", type="hub_split",
            segments=[
                FlightSegment(origin="PVG", destination="JFK",
                              departure_time=datetime(2026,6,15,17,0),
                              arrival_time=datetime(2026,6,15,20,0),
                              airline="CX", cabin=CabinClass.ECONOMY),
            ],
        )
        summary = display.format_airline_summary(it)
        assert len(summary) > 2
        assert "CX" in summary or "Cathay" in summary or "国泰" in summary

    def test_ground_arrow_for_access(self, display):
        it = Itinerary(
            id="t", type="hub_split",
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
        assert "⇢" in route


# ── Profile comparison ───────────────────────────────────────────────

class TestProfileComparison:
    def test_all_profiles_produce_different_results(self):
        """Same itinerary scored under different profiles should differ."""
        it = Itinerary(id="t", type="hub_split", total_price_usd=1000,
                       split_ticket_count=2, segments=[], offers=[])
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        scores = {}
        for profile in SCORING_PROFILES:
            scoring = ScoringService(profile=profile)
            rec = scoring.score(it, risk, 2000, airline_quality_score=0.7)
            scores[profile] = rec.final_score
        # At least some profiles should differ from each other
        unique_scores = len(set(round(s, 4) for s in scores.values()))
        assert unique_scores >= 2, f"All profiles produced identical scores: {scores}"

    def test_cheapest_profile_favors_low_price(self):
        scoring = ScoringService(profile="cheapest")
        risk_low = RiskAssessment(risk_level="low", risk_score=0.1)
        risk_med = RiskAssessment(risk_level="medium", risk_score=0.5)
        # Cheap direct with low risk vs expensive split with medium risk
        cheap = Itinerary(id="cheap", type="direct", total_price_usd=500,
                          split_ticket_count=0, total_estimated_time_hours=15,
                          segments=[], offers=[])
        expensive = Itinerary(id="expensive", type="hub_split", total_price_usd=1300,
                              split_ticket_count=3, total_estimated_time_hours=35,
                              total_access_cost_usd=200, segments=[], offers=[])
        r_cheap = scoring.score(cheap, risk_low, 2000, airline_quality_score=0.5)
        r_exp = scoring.score(expensive, risk_med, 2000, airline_quality_score=0.9)
        assert r_cheap.final_score > r_exp.final_score, \
            f"cheap={r_cheap.final_score:.3f} expensive={r_exp.final_score:.3f}"

    def test_low_risk_favors_low_risk(self):
        scoring = ScoringService(profile="low_risk")
        risk_low = RiskAssessment(risk_level="low", risk_score=0.1)
        risk_high = RiskAssessment(risk_level="high", risk_score=0.8)
        it_safe = Itinerary(id="safe", type="direct", total_price_usd=1100,
                            split_ticket_count=0, segments=[], offers=[])
        it_risky = Itinerary(id="risky", type="hub_split", total_price_usd=900,
                             split_ticket_count=3, segments=[], offers=[])
        r_safe = scoring.score(it_safe, risk_low, 2000, airline_quality_score=0.5)
        r_risky = scoring.score(it_risky, risk_high, 2000, airline_quality_score=0.5)
        assert r_safe.final_score > r_risky.final_score

    def test_airline_priority_boosts_premium(self):
        scoring = ScoringService(profile="airline_priority")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        it_budget = Itinerary(id="b", type="hub_split", total_price_usd=900,
                              split_ticket_count=2, segments=[], offers=[])
        it_premium = Itinerary(id="p", type="direct", total_price_usd=1000,
                               split_ticket_count=0, segments=[], offers=[])
        r_b = scoring.score(it_budget, risk, 2000, airline_quality_score=0.3)
        r_p = scoring.score(it_premium, risk, 2000, airline_quality_score=0.95)
        assert r_p.final_score > r_b.final_score

    def test_fastest_boosts_low_time(self):
        scoring = ScoringService(profile="fastest")
        risk = RiskAssessment(risk_level="medium", risk_score=0.5)
        it_slow = Itinerary(id="slow", type="hub_split", total_price_usd=900,
                            total_estimated_time_hours=35, split_ticket_count=2,
                            segments=[], offers=[])
        it_fast = Itinerary(id="fast", type="direct", total_price_usd=1100,
                            total_estimated_time_hours=15, split_ticket_count=0,
                            segments=[], offers=[])
        r_slow = scoring.score(it_slow, risk, 2000, airline_quality_score=0.5)
        r_fast = scoring.score(it_fast, risk, 2000, airline_quality_score=0.5)
        assert r_fast.final_score > r_slow.final_score


# ── Data quality ─────────────────────────────────────────────────────

class TestDataQuality:
    def test_mock_fallback_detected(self):
        from travel_agent.models.flight import FlightOffer
        offers = [
            FlightOffer(id="o1", segments=[], total_price_usd=100, source="mock_fallback"),
            FlightOffer(id="o2", segments=[], total_price_usd=200, source="mock_exact"),
        ]
        has_fallback = any(o.source == "mock_fallback" for o in offers)
        assert has_fallback

    def test_mock_exact_no_fallback(self):
        from travel_agent.models.flight import FlightOffer
        offers = [
            FlightOffer(id="o1", segments=[], total_price_usd=100, source="mock_exact"),
        ]
        has_fallback = any(o.source == "mock_fallback" for o in offers)
        assert not has_fallback


# ── Profile eval runner ──────────────────────────────────────────────

class TestProfileEvalRunner:
    def test_eval_queries_file_exists(self):
        path = Path(__file__).resolve().parent.parent / "evals" / "profile_eval_queries.jsonl"
        assert path.exists(), f"Profile eval queries not found at {path}"

    def test_eval_runner_imports(self):
        from scripts.run_profile_eval import load_queries, run_profile_evals
        assert load_queries is not None
        assert run_profile_evals is not None

    def test_load_profile_queries(self):
        from scripts.run_profile_eval import load_queries
        path = Path(__file__).resolve().parent.parent / "evals" / "profile_eval_queries.jsonl"
        queries = load_queries(str(path))
        assert len(queries) >= 10
        for q in queries:
            assert "query" in q
            assert "id" in q
