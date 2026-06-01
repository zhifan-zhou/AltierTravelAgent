"""End-to-end orchestrator tests."""

import pytest

from tests.conftest import SAMPLE_QUERY


class TestOrchestrator:
    def test_full_pipeline_runs_without_error(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY, debug=True))
        assert result.error is None, f"Pipeline error: {result.error}"
        assert result.intake is not None
        assert result.constraints is not None
        assert result.hub_split is not None
        assert result.flight_retrieval is not None
        assert result.route_composer is not None
        assert result.risk is not None
        assert result.ranking is not None
        assert result.explanation is not None

    def test_intake_parses_origin_destination(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))
        assert result.intake.origin_text == "宁波"
        assert result.intake.destination_text == "匹兹堡"

    def test_hub_split_discovers_alternative_hubs(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))
        assert len(result.hub_split.plan.candidate_hub_pairs) > 0

    def test_split_route_is_cheaper_than_direct(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))

        direct_price = None
        split_prices = []
        for rec in result.ranking.rankings:
            if rec.itinerary.type == "direct":
                direct_price = rec.itinerary.total_price_usd
            else:
                split_prices.append(rec.itinerary.total_price_usd)

        if direct_price and split_prices:
            avg_split = sum(split_prices) / len(split_prices)
            assert avg_split < direct_price, \
                f"Split route avg ${avg_split} should be cheaper than direct ${direct_price}"

    def test_ranking_has_three_categories(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))

        if len(result.ranking.rankings) >= 2:
            assert result.ranking.best_overall is not None
            assert result.ranking.cheapest_reasonable is not None
            assert result.ranking.lowest_risk is not None

    def test_direct_itinerary_is_low_risk(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))

        for rec in result.ranking.rankings:
            if rec.itinerary.type == "direct":
                assert rec.risk_assessment.risk_level == "low", \
                    f"Direct itinerary should be low risk, got {rec.risk_assessment.risk_level}"

    def test_split_itinerary_has_warnings(self, orchestrator):
        import asyncio
        result = asyncio.run(orchestrator.run(SAMPLE_QUERY))

        split_found = False
        for rec in result.ranking.rankings:
            if rec.itinerary.type == "hub_split":
                split_found = True
                assert len(rec.risk_assessment.warnings) > 0, \
                    "Split ticket itinerary should have warnings"

        assert split_found, "No hub_split itinerary found in results"
