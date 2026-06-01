"""Risk & Compliance Agent: assess risk for each itinerary."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import RiskOutput, RouteComposerResult
from travel_agent.models.risk import RiskAssessment
from travel_agent.services.risk_service import RiskService


class RiskComplianceAgent(BaseAgent[RouteComposerResult, RiskOutput]):
    """Evaluate all risk factors for every itinerary.

    Risk factors include: split ticket, short connection, baggage recheck,
    overnight stay, airport transfer, visa/entry, hidden city, price expiration.
    """

    name = "risk_compliance"

    def __init__(self, risk_service: RiskService | None = None):
        super().__init__()
        self._risk_service = risk_service or RiskService()

    async def execute(self, data: RouteComposerResult) -> RiskOutput:
        assessments: dict[str, RiskAssessment] = {}

        for it in data.output.itineraries:
            assessment = self._risk_service.assess(it)
            # Update the itinerary's risk info in-place for downstream use
            it.risk_level = assessment.risk_level
            it.warnings = assessment.warnings
            assessments[it.id] = assessment
            self.logger.info(f"Itinerary {it.id}: risk={assessment.risk_level}, score={assessment.risk_score}")

        return RiskOutput(assessments=assessments)
