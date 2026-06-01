"""Travel Agent Orchestrator: coordinates the full agent pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from travel_agent.agents.booking_stub_agent import BookingStubAgent
from travel_agent.agents.constraint_agent import ConstraintAgent
from travel_agent.agents.explanation_agent import ExplanationAgent
from travel_agent.agents.flight_retrieval_agent import FlightRetrievalAgent
from travel_agent.agents.hubsplit_agent import HubSplitAgent
from travel_agent.agents.intake_agent import IntakeAgent
from travel_agent.agents.ranking_agent import RankingAgent
from travel_agent.agents.risk_compliance_agent import RiskComplianceAgent
from travel_agent.agents.route_composer_agent import RouteComposerAgent
from travel_agent.agents.search_strategy_agent import SearchStrategyAgent
from travel_agent.models.agent_outputs import TravelAgentResult
from travel_agent.providers.provider_router import ProviderRouter
from travel_agent.services.airport_service import AirportService
from travel_agent.services.risk_service import RiskService
from travel_agent.core.config import get_settings

logger = logging.getLogger("travel_agent.orchestrator")


class TravelAgentOrchestrator:
    """Central orchestrator for the Travel Agent pipeline.

    Coordinates all agents in sequence:
      Intake -> Constraint -> HubSplit -> SearchStrategy ->
      FlightRetrieval -> RouteComposer -> RiskCompliance ->
      Ranking -> Explanation -> (optional BookingStub)
    """

    def __init__(
        self,
        router: ProviderRouter | None = None,
        airport_service: AirportService | None = None,
        risk_service: RiskService | None = None,
    ):
        self._airport_service = airport_service or AirportService()
        self._risk_service = risk_service or RiskService()
        self._router = router or ProviderRouter()

        # Initialize agents
        self.intake = IntakeAgent(airport_service=self._airport_service)
        self.constraint = ConstraintAgent()
        self.hubsplit = HubSplitAgent(airport_service=self._airport_service)
        self.search_strategy = SearchStrategyAgent()
        self.flight_retrieval = FlightRetrievalAgent(
            router=self._router,
            airport_service=self._airport_service,
        )
        self.route_composer = RouteComposerAgent()
        self.risk_compliance = RiskComplianceAgent(risk_service=self._risk_service)
        self.ranking = RankingAgent()
        self.explanation = ExplanationAgent()
        self.booking_stub = BookingStubAgent()

    async def run(self, query: str, debug: bool = False) -> TravelAgentResult:
        """Execute the full pipeline for a user query."""
        result = TravelAgentResult(query=query)
        debug_artifacts: dict = {}

        try:
            # Step 1: Intake
            intake_output = await self.intake.execute(query)
            result.intake = intake_output
            if debug:
                debug_artifacts["intake"] = intake_output.model_dump()

            # Step 2: Constraints
            constraint_output = await self.constraint.execute(intake_output)
            result.constraints = constraint_output
            if debug:
                debug_artifacts["constraints"] = constraint_output.model_dump()

            # Step 3: HubSplit
            hub_split_output = await self.hubsplit.execute((constraint_output, None))
            result.hub_split = hub_split_output
            if debug:
                debug_artifacts["hub_split"] = hub_split_output.model_dump()

            # Step 4: Search Strategy
            search_strategy_output = await self.search_strategy.execute((hub_split_output, None))
            result.search_strategy = search_strategy_output
            if debug:
                debug_artifacts["search_strategy"] = search_strategy_output.model_dump()

            # Step 5: Flight Retrieval
            flight_retrieval_output = await self.flight_retrieval.execute((search_strategy_output, None))
            result.flight_retrieval = flight_retrieval_output
            if debug:
                debug_artifacts["flight_retrieval"] = {
                    "total_offers": len(flight_retrieval_output.all_offers),
                    "direct": len(flight_retrieval_output.direct_offers),
                    "hub_split": len(flight_retrieval_output.hub_split_offers),
                    "domestic": len(flight_retrieval_output.domestic_offers),
                }

            # Step 6: Route Composer
            route_composer_output = await self.route_composer.execute(
                (flight_retrieval_output, hub_split_output, None)
            )
            result.route_composer = route_composer_output
            if debug:
                debug_artifacts["route_composer"] = {
                    "itineraries": len(route_composer_output.output.itineraries),
                }

            # Step 7: Risk & Compliance
            risk_output = await self.risk_compliance.execute(route_composer_output)
            result.risk = risk_output
            if debug:
                debug_artifacts["risk"] = {
                    it_id: a.model_dump()
                    for it_id, a in risk_output.assessments.items()
                }

            # Step 8: Ranking
            ranking_output = await self.ranking.execute(
                (route_composer_output, risk_output, constraint_output)
            )
            result.ranking = ranking_output
            if debug:
                debug_artifacts["ranking"] = {
                    "top_3": [
                        {"rank": r.rank, "id": r.itinerary.id, "score": r.final_score}
                        for r in ranking_output.rankings[:3]
                    ]
                }

            # Step 9: Explanation
            explanation_output = await self.explanation.execute(
                (ranking_output, hub_split_output, constraint_output)
            )
            result.explanation = explanation_output
            if debug:
                debug_artifacts["explanation"] = explanation_output.model_dump()

        except Exception as e:
            logger.exception("Pipeline execution failed")
            result.error = str(e)

        result.debug_artifacts = debug_artifacts
        return result

    async def run_and_save(self, query: str, debug: bool = False, output_dir: str | None = None) -> TravelAgentResult:
        """Run pipeline and save results to disk."""
        result = await self.run(query, debug=debug)

        if output_dir is None:
            from travel_agent.core.config import get_settings
            settings = get_settings()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{settings.runs_dir}/{ts}"

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save full result
        result_path = out_path / "result.json"
        result_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Save debug artifacts separately
        if debug:
            artifacts_path = out_path / "artifacts"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            for step_name, artifact in result.debug_artifacts.items():
                file_path = artifacts_path / f"{step_name}.json"
                file_path.write_text(
                    json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )

        logger.info(f"Results saved to {out_path}")
        return result
