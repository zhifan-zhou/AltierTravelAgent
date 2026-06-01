"""Shared fixtures for all tests."""

import pytest
from travel_agent.providers.mock_flight_provider import MockFlightProvider
from travel_agent.providers.provider_router import ProviderRouter
from travel_agent.services.airport_service import AirportService
from travel_agent.services.risk_service import RiskService
from travel_agent.agents.intake_agent import IntakeAgent
from travel_agent.agents.constraint_agent import ConstraintAgent
from travel_agent.agents.hubsplit_agent import HubSplitAgent
from travel_agent.agents.search_strategy_agent import SearchStrategyAgent
from travel_agent.agents.flight_retrieval_agent import FlightRetrievalAgent
from travel_agent.agents.route_composer_agent import RouteComposerAgent
from travel_agent.agents.risk_compliance_agent import RiskComplianceAgent
from travel_agent.agents.ranking_agent import RankingAgent
from travel_agent.agents.explanation_agent import ExplanationAgent
from travel_agent.core.orchestrator import TravelAgentOrchestrator


@pytest.fixture
def airport_service():
    return AirportService()


@pytest.fixture
def risk_service():
    return RiskService()


@pytest.fixture
def mock_provider():
    return MockFlightProvider()


@pytest.fixture
def mock_router():
    """ProviderRouter forced to mock mode for testing."""
    return ProviderRouter()


@pytest.fixture
def intake_agent():
    return IntakeAgent()


@pytest.fixture
def constraint_agent():
    return ConstraintAgent()


@pytest.fixture
def hubsplit_agent(airport_service):
    return HubSplitAgent(airport_service=airport_service)


@pytest.fixture
def search_strategy_agent():
    return SearchStrategyAgent()


@pytest.fixture
def flight_retrieval_agent(mock_router):
    return FlightRetrievalAgent(router=mock_router)


@pytest.fixture
def route_composer_agent():
    return RouteComposerAgent()


@pytest.fixture
def empty_exclusions():
    from travel_agent.services.constraint_compiler import ExclusionRules
    return ExclusionRules()


@pytest.fixture
def risk_compliance_agent(risk_service):
    return RiskComplianceAgent(risk_service=risk_service)


@pytest.fixture
def ranking_agent():
    return RankingAgent()


@pytest.fixture
def explanation_agent():
    return ExplanationAgent()


@pytest.fixture
def orchestrator(mock_router, airport_service, risk_service):
    return TravelAgentOrchestrator(
        router=mock_router,
        airport_service=airport_service,
        risk_service=risk_service,
    )


SAMPLE_QUERY = "我要从宁波飞匹兹堡，便宜点，可以从上海走，也可以纽约或者华盛顿转"
