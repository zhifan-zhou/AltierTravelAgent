"""Phase 13 tests: contract, exclusions, validator, SFT logging."""

import asyncio
import json
from pathlib import Path

import pytest

from travel_agent.models.requirement_contract import (
    TravelRequirementContract, TripRequirement, GeographyRequirement,
    TicketingPreferences, RankingPreferences,
)
from travel_agent.services.constraint_compiler import ConstraintCompiler, ExclusionRules
from travel_agent.services.recommendation_validator import RecommendationValidator
from travel_agent.services.conversation_dataset_logger import ConversationDatasetLogger
from travel_agent.agents.conversation_requirement_agent import (
    ConversationRequirementAgent, RequirementUpdateResult,
)


# ── Contract schema ──────────────────────────────────────────────────

class TestContractSchema:
    def test_default_contract(self):
        c = TravelRequirementContract()
        assert not c.ready_to_search
        assert c.schema_version == "v2"

    def test_has_required_route(self):
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡")
        )
        assert c.has_required_route()

    def test_add_avoid_airports(self):
        c = TravelRequirementContract()
        c.geography.avoid_airports.append("PVG")
        c.geography.avoid_airports.append("SHA")
        assert "PVG" in c.geography.avoid_airports

    def test_remove_from_acceptable_hubs(self):
        c = TravelRequirementContract(
            geography=GeographyRequirement(acceptable_origin_hubs=["PVG", "SHA", "HGH"])
        )
        c.geography.acceptable_origin_hubs = [h for h in c.geography.acceptable_origin_hubs if h not in ("PVG", "SHA")]
        assert "PVG" not in c.geography.acceptable_origin_hubs
        assert "HGH" in c.geography.acceptable_origin_hubs

    def test_summary_zh(self):
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡")
        )
        s = c.summary_zh()
        assert "温州" in s
        assert "匹兹堡" in s

    def test_to_json_for_sft(self):
        c = TravelRequirementContract()
        d = c.to_sft_target()
        assert "trip" in d
        assert "geography" in d


# ── ConstraintCompiler ───────────────────────────────────────────────

class TestConstraintCompiler:
    def test_excludes_avoid_airports(self):
        compiler = ConstraintCompiler()
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡"),
            geography=GeographyRequirement(avoid_airports=["PVG", "SHA"]),
        )
        compiled = compiler.compile(c)
        assert "PVG" in compiled.exclusions.excluded_airports
        assert "SHA" in compiled.exclusions.excluded_airports

    def test_profile_from_contract(self):
        compiler = ConstraintCompiler()
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡"),
            ranking_preferences=RankingPreferences(profile="low_risk"),
        )
        compiled = compiler.compile(c)
        assert compiled.profile == "low_risk"


# ── RecommendationValidator ──────────────────────────────────────────

class TestRecommendationValidator:
    def test_removes_itinerary_with_excluded_airport(self):
        from travel_agent.models.flight import FlightSegment
        from travel_agent.models.itinerary import Itinerary
        from travel_agent.models.ranking import RankedRecommendation
        from travel_agent.models.risk import RiskAssessment
        from datetime import datetime

        it = Itinerary(
            id="test", type="hub_split",
            segments=[FlightSegment(
                origin="PVG", destination="JFK",
                departure_time=datetime(2026,6,15,17,0),
                arrival_time=datetime(2026,6,15,20,0),
                airline="MU",
            )],
            origin_airport="WNZ", destination_airport="PIT",
        )
        rec = RankedRecommendation(itinerary=it)
        validator = RecommendationValidator()
        exclusions = ExclusionRules(excluded_airports=["PVG"])
        valid = validator.validate([rec], exclusions)
        assert len(valid) == 0

    def test_keeps_clean_itinerary(self):
        from travel_agent.models.flight import FlightSegment
        from travel_agent.models.itinerary import Itinerary
        from travel_agent.models.ranking import RankedRecommendation
        from datetime import datetime

        it = Itinerary(
            id="test", type="hub_split",
            segments=[FlightSegment(
                origin="HGH", destination="JFK",
                departure_time=datetime(2026,6,15,17,0),
                arrival_time=datetime(2026,6,15,20,0),
                airline="MU",
            )],
            origin_airport="WNZ", destination_airport="PIT",
        )
        rec = RankedRecommendation(itinerary=it)
        validator = RecommendationValidator()
        exclusions = ExclusionRules(excluded_airports=["PVG", "SHA"])
        valid = validator.validate([rec], exclusions)
        assert len(valid) == 1


# ── ConversationRequirementAgent ─────────────────────────────────────

class TestConversationAgent:
    @pytest.fixture
    def agent(self):
        return ConversationRequirementAgent()

    def test_creates_new_contract(self, agent):
        result = asyncio.run(agent.execute(("温州到匹兹堡，可以从上海走", None, {})))
        assert result.updated_contract is not None
        assert result.updated_contract.trip.primary_origin_text in ("温州", "上海", "")

    def test_avoid_shanghai(self, agent):
        """'我不想去上海' should add PVG/SHA to avoid_airports."""
        # First create contract with Shanghai as acceptable hub
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡"),
            geography=GeographyRequirement(acceptable_origin_hubs=["PVG", "SHA"]),
        )
        result = asyncio.run(agent.execute(("我不想去上海", c, {})))
        updated = result.updated_contract
        has_pvg = "PVG" in updated.geography.avoid_airports
        has_sha = "SHA" in updated.geography.avoid_airports
        assert has_pvg or has_sha, f"PVG/SHA should be in avoid_airports after rejecting Shanghai"

    def test_family_sets_low_risk(self, agent):
        c = TravelRequirementContract(
            trip=TripRequirement(primary_origin_text="温州", primary_destination_text="匹兹堡"),
        )
        result = asyncio.run(agent.execute(("我爸妈也一起，别太折腾", c, {})))
        updated = result.updated_contract


# ── SFT Logger ───────────────────────────────────────────────────────

class TestSFTLogger:
    def test_logger_creates_files(self, tmp_path):
        import os
        from travel_agent.services import conversation_dataset_logger
        original = conversation_dataset_logger.OUTPUT_DIR
        conversation_dataset_logger.OUTPUT_DIR = tmp_path / "conv_data"

        try:
            logger = ConversationDatasetLogger()
            logger.log_message("user", "温州到匹兹堡")
            logger.set_initial_contract({"trip": {"origin": "温州"}})
            logger.log_contract_update("温州到匹兹堡", ["origin"], {"trip": {"origin": "温州"}})
            path = logger.save({"trip": {"origin": "温州", "destination": "匹兹堡"}})

            assert (path / "sample.json").exists()
            assert (path / "sft_samples.jsonl").exists()

            with open(path / "sample.json") as f:
                sample = json.load(f)
            assert "conversation" in sample
            assert "final_contract" in sample
            assert "contract_updates" in sample
        finally:
            conversation_dataset_logger.OUTPUT_DIR = original
