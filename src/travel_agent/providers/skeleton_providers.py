"""Optional provider skeletons — placeholder implementations.

Each provider validates config and raises clear errors when used without keys.
Only DuffelProvider is fully implemented for search.
"""

from __future__ import annotations

import logging

from travel_agent.core.config import get_settings
from travel_agent.models.flight import FlightOffer, FlightSearchRequest
from travel_agent.providers.base import (
    BaseFlightProvider,
    ProviderCapabilities,
    ProviderConfigurationError,
)

logger = logging.getLogger("travel_agent.providers.skeletons")


# ── SerpApi Google Flights ────────────────────────────────────────────

class SerpApiGoogleFlightsProvider(BaseFlightProvider):
    """SerpApi Google Flights provider — skeleton only.

    Requires SERPAPI_API_KEY env var.
    Docs: https://serpapi.com/google-flights-api
    """

    def __init__(self) -> None:
        self._key = get_settings().serpapi_api_key or ""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="serpapi_google_flights",
            supports_search=True, supports_booking=False,
            supports_price_verify=False, requires_api_key=True,
            is_real_provider=True, max_results_per_search=20,
        )

    def validate_config(self) -> None:
        if not self._key:
            raise ProviderConfigurationError(
                "serpapi_google_flights",
                "SERPAPI_API_KEY is not set. Get a key from https://serpapi.com",
            )

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        logger.warning("SerpApiGoogleFlightsProvider.search_flights not yet implemented")
        return []


# ── SearchApi Google Flights ──────────────────────────────────────────

class SearchApiGoogleFlightsProvider(BaseFlightProvider):
    """SearchApi Google Flights provider — skeleton only.

    Requires SEARCHAPI_API_KEY env var.
    Docs: https://www.searchapi.io/docs/google-flights
    """

    def __init__(self) -> None:
        self._key = get_settings().searchapi_api_key or ""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="searchapi_google_flights",
            supports_search=True, supports_booking=False,
            supports_price_verify=False, requires_api_key=True,
            is_real_provider=True, max_results_per_search=20,
        )

    def validate_config(self) -> None:
        if not self._key:
            raise ProviderConfigurationError(
                "searchapi_google_flights",
                "SEARCHAPI_API_KEY is not set. Get a key from https://www.searchapi.io",
            )

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        logger.warning("SearchApiGoogleFlightsProvider.search_flights not yet implemented")
        return []


# ── Skyscanner ────────────────────────────────────────────────────────

class SkyscannerProvider(BaseFlightProvider):
    """Skyscanner provider — skeleton only.

    Requires SKYSCANNER_API_KEY env var.
    Docs: https://developers.skyscanner.net
    """

    def __init__(self) -> None:
        self._key = get_settings().skyscanner_api_key or ""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="skyscanner",
            supports_search=True, supports_booking=False,
            supports_price_verify=False, requires_api_key=True,
            is_real_provider=True, max_results_per_search=20,
        )

    def validate_config(self) -> None:
        if not self._key:
            raise ProviderConfigurationError(
                "skyscanner",
                "SKYSCANNER_API_KEY is not set. Get a key from https://developers.skyscanner.net",
            )

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        logger.warning("SkyscannerProvider.search_flights not yet implemented")
        return []


# ── Kiwi ──────────────────────────────────────────────────────────────

class KiwiProvider(BaseFlightProvider):
    """Kiwi.com provider — skeleton only.

    Requires KIWI_API_KEY env var.
    Docs: https://docs.kiwi.com
    """

    def __init__(self) -> None:
        self._key = get_settings().kiwi_api_key or ""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="kiwi",
            supports_search=True, supports_booking=False,
            supports_price_verify=False, requires_api_key=True,
            is_real_provider=True, max_results_per_search=20,
        )

    def validate_config(self) -> None:
        if not self._key:
            raise ProviderConfigurationError(
                "kiwi",
                "KIWI_API_KEY is not set. Get a key from https://docs.kiwi.com",
            )

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        logger.warning("KiwiProvider.search_flights not yet implemented")
        return []


# ── Amadeus (legacy/optional) ─────────────────────────────────────────

class AmadeusLegacyProvider(BaseFlightProvider):
    """Amadeus provider — legacy skeleton.

    Only used as fallback. Duffel is the preferred real provider.
    Requires AMADEUS_API_KEY and AMADEUS_API_SECRET.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._key = settings.amadeus_api_key or ""
        self._secret = settings.amadeus_api_secret or ""

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="amadeus",
            supports_search=True, supports_booking=False,
            supports_price_verify=False, requires_api_key=True,
            is_real_provider=True, max_results_per_search=10,
        )

    def validate_config(self) -> None:
        if not self._key or not self._secret:
            raise ProviderConfigurationError(
                "amadeus",
                "AMADEUS_API_KEY and AMADEUS_API_SECRET must be set. "
                "Get keys from https://developers.amadeus.com/register",
            )

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        logger.warning("AmadeusLegacyProvider.search_flights not yet implemented")
        return []


# ── Registry ──────────────────────────────────────────────────────────

SKELETON_PROVIDERS: dict[str, type[BaseFlightProvider]] = {
    "serpapi_google_flights": SerpApiGoogleFlightsProvider,
    "searchapi_google_flights": SearchApiGoogleFlightsProvider,
    "skyscanner": SkyscannerProvider,
    "kiwi": KiwiProvider,
    "amadeus": AmadeusLegacyProvider,
}
