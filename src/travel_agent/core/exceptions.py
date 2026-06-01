"""Domain-specific exceptions."""


class TravelAgentError(Exception):
    """Base exception for travel agent errors."""


class AgentExecutionError(TravelAgentError):
    """Raised when an agent fails during execution."""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")


class ProviderError(TravelAgentError):
    """Raised when a flight provider fails."""

    def __init__(self, provider_name: str, message: str):
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class DataNotFoundError(TravelAgentError):
    """Raised when required data is not found."""


class ValidationError(TravelAgentError):
    """Raised when input validation fails."""


class NoFlightsFoundError(TravelAgentError):
    """Raised when no flights are found for a search."""
