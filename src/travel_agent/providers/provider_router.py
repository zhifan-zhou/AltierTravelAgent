"""Provider Router — selects and coordinates flight providers.

Supports three modes:
  - mock: MockFlightProvider only
  - duffel: DuffelProvider only (fails if no token)
  - hybrid: tries real providers in priority order, falls back to mock

Records per-search diagnostics for every task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from travel_agent.core.config import get_settings
from travel_agent.core.exceptions import ProviderError
from travel_agent.models.flight import FlightOffer, FlightSearchRequest
from travel_agent.providers.base import BaseFlightProvider, ProviderConfigurationError
from travel_agent.providers.mock_flight_provider import MockFlightProvider
from travel_agent.providers.provider_cache import ProviderCache

logger = logging.getLogger("travel_agent.providers.router")


@dataclass
class TaskDiagnostic:
    """Per-search-task diagnostic record."""
    origin: str
    destination: str
    cabin: str
    providers_tried: list[str] = field(default_factory=list)
    providers_skipped: list[str] = field(default_factory=list)
    provider_used: str = ""
    offers_returned: int = 0
    offers_from: str = ""  # "exact", "fallback", "real_api"
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cached: bool = False


class ProviderRouter:
    """Routes flight searches to the appropriate provider(s).

    Mode selection via TRAVEL_AGENT_PROVIDER env var:
      - "mock": mock only
      - "duffel": Duffel only
      - "hybrid": real providers + mock fallback
    """

    def __init__(self):
        settings = get_settings()
        self._mode = settings.travel_agent_provider.lower()
        self._hybrid_fallback = getattr(settings, 'hybrid_enable_mock_fallback', True)
        self._cache = ProviderCache(ttl_seconds=300)
        self._diagnostics: list[TaskDiagnostic] = []

        # Parse hybrid provider priority
        priority_str = getattr(settings, 'real_provider_priority', '')
        self._priority = [p.strip() for p in priority_str.split(",") if p.strip()] if priority_str else [
            "duffel", "serpapi_google_flights", "searchapi_google_flights",
            "skyscanner", "kiwi", "amadeus",
        ]

        # Lazy-initialized providers
        self._mock: MockFlightProvider | None = None
        self._real_providers: dict[str, BaseFlightProvider] = {}
        self._provider_available: dict[str, bool] = {}

        logger.info(f"ProviderRouter mode={self._mode} priority={self._priority}")

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def diagnostics(self) -> list[TaskDiagnostic]:
        return self._diagnostics

    def clear_diagnostics(self) -> None:
        self._diagnostics.clear()

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        """Search flights through the appropriate provider(s)."""
        diag = TaskDiagnostic(
            origin=request.origin,
            destination=request.destination,
            cabin=request.cabin.value,
        )
        self._diagnostics.append(diag)

        t0 = datetime.now()

        # Check cache
        cached = self._cache.get(
            self._mode, request.origin, request.destination,
            request.departure_date.strftime("%Y-%m-%d") if request.departure_date else "any",
            request.cabin.value, request.passengers,
        )
        if cached is not None:
            diag.cached = True
            diag.provider_used = "cache"
            diag.offers_returned = len(cached) if isinstance(cached, list) else 0
            diag.latency_ms = (datetime.now() - t0).total_seconds() * 1000
            return cached if isinstance(cached, list) else []

        offers: list[FlightOffer] = []

        if self._mode == "mock":
            offers = await self._search_mock(request, diag)
        elif self._mode == "duffel":
            offers = await self._search_single_real("duffel", request, diag)
        elif self._mode == "hybrid":
            offers = await self._search_hybrid(request, diag)
        else:
            # Unknown mode — fall back to mock
            logger.warning(f"Unknown provider mode '{self._mode}', falling back to mock")
            offers = await self._search_mock(request, diag)

        diag.latency_ms = (datetime.now() - t0).total_seconds() * 1000
        diag.offers_returned = len(offers)

        # Cache results
        self._cache.set(
            self._mode, request.origin, request.destination,
            request.departure_date.strftime("%Y-%m-%d") if request.departure_date else "any",
            request.cabin.value, request.passengers, offers,
        )

        return offers

    async def _search_mock(self, request: FlightSearchRequest, diag: TaskDiagnostic) -> list[FlightOffer]:
        if self._mock is None:
            self._mock = MockFlightProvider()
        diag.providers_tried.append("mock")
        diag.provider_used = "mock"
        offers = await self._mock.search_flights(request)
        diag.offers_from = "mock"
        return offers

    async def _search_single_real(self, name: str, request: FlightSearchRequest, diag: TaskDiagnostic) -> list[FlightOffer]:
        provider = self._get_real_provider(name)
        if provider is None:
            diag.providers_skipped.append(name)
            diag.errors.append(f"{name}: unavailable")
            return []
        diag.providers_tried.append(name)
        try:
            provider.validate_config()
            offers = await provider.search_flights(request)
            diag.provider_used = name
            diag.offers_from = "real_api"
            return offers
        except ProviderConfigurationError as e:
            diag.providers_skipped.append(name)
            diag.errors.append(str(e))
            logger.info(f"Skipping {name}: {e}")
            return []
        except Exception as e:
            diag.errors.append(f"{name}: {e}")
            logger.warning(f"Provider {name} search error: {e}")
            return []

    async def _search_hybrid(self, request: FlightSearchRequest, diag: TaskDiagnostic) -> list[FlightOffer]:
        """Try real providers in priority order, collect results, fall back to mock."""
        all_offers: list[FlightOffer] = []

        for name in self._priority:
            offers = await self._search_single_real(name, request, diag)
            all_offers.extend(offers)

        # If no real results, fall back to mock
        if not all_offers and self._hybrid_fallback:
            logger.info("Hybrid: no real results, falling back to mock")
            mock_offers = await self._search_mock(request, TaskDiagnostic(
                origin=request.origin, destination=request.destination, cabin=request.cabin.value,
            ))
            for o in mock_offers:
                o.source = "mock_fallback"  # Ensure labeled correctly
                o.is_real = False
                o.confidence = "estimated"
            all_offers = mock_offers
            diag.offers_from = "mock_fallback"
        elif all_offers:
            diag.offers_from = "real_api"
        else:
            diag.offers_from = "none"

        diag.offers_returned = len(all_offers)
        return all_offers

    def _get_real_provider(self, name: str) -> BaseFlightProvider | None:
        """Get or create a real provider by name. Returns None if unavailable."""
        if name in self._provider_available and not self._provider_available[name]:
            return None
        if name in self._real_providers:
            return self._real_providers[name]

        provider = self._instantiate_provider(name)
        if provider is None:
            self._provider_available[name] = False
            return None

        try:
            provider.validate_config()
            self._real_providers[name] = provider
            self._provider_available[name] = True
            return provider
        except ProviderConfigurationError:
            self._provider_available[name] = False
            return None

    def _instantiate_provider(self, name: str) -> BaseFlightProvider | None:
        """Instantiate a provider by name. Returns None if unknown."""
        if name == "duffel":
            from travel_agent.providers.duffel_provider import DuffelProvider
            return DuffelProvider()
        if name in ("serpapi_google_flights", "serpapi"):
            from travel_agent.providers.skeleton_providers import SerpApiGoogleFlightsProvider
            return SerpApiGoogleFlightsProvider()
        if name in ("searchapi_google_flights", "searchapi"):
            from travel_agent.providers.skeleton_providers import SearchApiGoogleFlightsProvider
            return SearchApiGoogleFlightsProvider()
        if name == "skyscanner":
            from travel_agent.providers.skeleton_providers import SkyscannerProvider
            return SkyscannerProvider()
        if name == "kiwi":
            from travel_agent.providers.skeleton_providers import KiwiProvider
            return KiwiProvider()
        if name == "amadeus":
            from travel_agent.providers.skeleton_providers import AmadeusLegacyProvider
            return AmadeusLegacyProvider()
        return None

    @property
    def provider_summary(self) -> dict:
        """Summary of available and configured providers."""
        return {
            "mode": self._mode,
            "available_real": [k for k, v in self._provider_available.items() if v],
            "unavailable_real": [k for k, v in self._provider_available.items() if not v],
            "priority": self._priority,
            "hybrid_fallback": self._hybrid_fallback,
            "cache_size": self._cache.size,
        }
