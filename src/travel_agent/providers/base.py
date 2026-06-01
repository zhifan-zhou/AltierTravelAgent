"""Base provider interface for flight search — provider-agnostic design."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from travel_agent.core.exceptions import ProviderError
from travel_agent.models.flight import FlightOffer, FlightSearchRequest


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is misconfigured (missing keys, invalid settings)."""
    pass


@dataclass
class ProviderCapabilities:
    """Declares what a provider can and cannot do."""
    provider_name: str = "base"
    supports_search: bool = False
    supports_booking: bool = False
    supports_price_verify: bool = False
    requires_api_key: bool = False
    is_real_provider: bool = False  # False = mock/synthetic data
    max_results_per_search: int = 10
    typical_latency_ms: int = 2000
    rate_limit_per_minute: int = 30


class BaseFlightProvider(ABC):
    """Abstract interface for flight search providers.

    Provider-agnostic: mock, Duffel, SerpApi, SearchApi, Skyscanner, Kiwi.
    All providers must implement validate_config() at minimum.
    """

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return this provider's capability declaration."""
        ...

    @abstractmethod
    def validate_config(self) -> None:
        """Validate that required credentials/config are present.

        Raises ProviderConfigurationError if misconfigured.
        """
        ...

    @abstractmethod
    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        """Search for flights matching the request."""
        ...

    async def verify_price(self, offer_id: str) -> FlightOffer | None:
        """Re-verify a previously fetched offer price. Optional."""
        return None

    async def health_check(self) -> bool:
        """Check if the provider API is reachable and authenticated."""
        try:
            self.validate_config()
            return True
        except ProviderConfigurationError:
            return False
