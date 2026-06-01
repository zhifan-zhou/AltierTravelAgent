"""Duffel flight provider — search-only implementation.

Connects to Duffel API for flight offer search.
Requires DUFFEL_API_TOKEN environment variable.

API docs: https://duffel.com/docs/api
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from travel_agent.core.config import get_settings
from travel_agent.models.flight import CabinClass, FlightOffer, FlightSearchRequest, FlightSegment
from travel_agent.providers.base import (
    BaseFlightProvider,
    ProviderCapabilities,
    ProviderConfigurationError,
)

logger = logging.getLogger("travel_agent.providers.duffel")


class DuffelProvider(BaseFlightProvider):
    """Duffel API flight search provider — search only, no booking."""

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.duffel_api_token or settings.duffel_api_key
        self._base_url = getattr(settings, 'duffel_base_url', 'https://api.duffel.com')
        self._api_version = getattr(settings, 'duffel_api_version', 'v2')
        self._timeout = getattr(settings, 'duffel_timeout_seconds', 20)
        self._max_retries = getattr(settings, 'duffel_max_retries', 2)
        self._http: httpx.AsyncClient | None = None

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="duffel",
            supports_search=True,
            supports_booking=False,
            supports_price_verify=False,
            requires_api_key=True,
            is_real_provider=True,
            max_results_per_search=50,
            typical_latency_ms=2000,
            rate_limit_per_minute=50,
        )

    def validate_config(self) -> None:
        if not self._token:
            raise ProviderConfigurationError(
                "duffel",
                "DUFFEL_API_TOKEN is not set. Get a token from https://duffel.com/register",
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Duffel-Version": self._api_version,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        """Search flight offers via Duffel API.

        Uses POST /air/offer_requests to create a search, then
        GET /air/offers?offer_request_id=... to retrieve results.
        """
        self.validate_config()
        client = await self._get_client()

        # Build search request
        passengers = [{"type": "adult"} for _ in range(max(request.passengers, 1))]
        slices = [{
            "origin": request.origin.upper(),
            "destination": request.destination.upper(),
            "departure_date": (request.departure_date or datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
        }]

        cabin_map = {
            CabinClass.ECONOMY: "economy",
            CabinClass.PREMIUM_ECONOMY: "premium_economy",
            CabinClass.BUSINESS: "business",
            CabinClass.FIRST: "first",
        }
        cabin_class = cabin_map.get(request.cabin, "economy")

        body = {
            "data": {
                "slices": slices,
                "passengers": passengers,
                "cabin_class": cabin_class,
                "max_connections": 2,
            }
        }

        # Step 1: Create offer request
        try:
            resp = await client.post(f"{self._base_url}/air/offer_requests", json=body)
            resp.raise_for_status()
            offer_request = resp.json()
            offer_request_id = offer_request["data"]["id"]
        except httpx.HTTPError as e:
            logger.error(f"Duffel offer request failed: {e}")
            return []

        # Step 2: Poll for offers
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.get(
                    f"{self._base_url}/air/offers",
                    params={"offer_request_id": offer_request_id, "limit": 10},
                )
                resp.raise_for_status()
                data = resp.json()
                return self._parse_offers(data.get("data", []))
            except httpx.HTTPError as e:
                if attempt < self._max_retries:
                    logger.warning(f"Duffel offer fetch retry {attempt+1}: {e}")
                    import asyncio
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.error(f"Duffel offer fetch failed after retries: {e}")
                    return []

        return []

    def _parse_offers(self, raw_offers: list[dict]) -> list[FlightOffer]:
        """Normalize Duffel offers into FlightOffer models."""
        results = []
        for raw in raw_offers:
            segments = []
            for slice_data in raw.get("slices", []):
                for seg_raw in slice_data.get("segments", []):
                    segments.append(FlightSegment(
                        origin=seg_raw.get("origin", {}).get("iata_code", ""),
                        destination=seg_raw.get("destination", {}).get("iata_code", ""),
                        departure_time=datetime.fromisoformat(
                            seg_raw.get("departing_at", datetime.now().isoformat())
                        ) if seg_raw.get("departing_at") else datetime.now(),
                        arrival_time=datetime.fromisoformat(
                            seg_raw.get("arriving_at", datetime.now().isoformat())
                        ) if seg_raw.get("arriving_at") else datetime.now(),
                        airline=seg_raw.get("marketing_carrier", {}).get("iata_code", ""),
                        airline_name=seg_raw.get("marketing_carrier", {}).get("name", ""),
                        flight_number=str(seg_raw.get("marketing_carrier_flight_number", "")),
                        cabin=CabinClass.ECONOMY,
                        aircraft=seg_raw.get("aircraft", {}).get("name", ""),
                    ))

            price = raw.get("total_amount") or raw.get("base_amount") or "0.00"
            currency = raw.get("total_currency", "USD")

            results.append(FlightOffer(
                id=raw.get("id", ""),
                segments=segments,
                total_price_usd=float(price) if currency == "USD" else float(price),
                currency=currency,
                provider_name="duffel",
                source="duffel_api",
                is_real=True,
                confidence="verified",
                booking_available=raw.get("live_mode", False),
                baggage_included=any(
                    s.get("passengers", [{}])[0].get("baggages", [{}])[0].get("type") == "checked"
                    for sl in raw.get("slices", [])
                    for s in sl.get("segments", [])
                ) if raw.get("slices") else True,
                raw_provider_payload=raw,
            ))

        return results

    async def verify_price(self, offer_id: str) -> FlightOffer | None:
        return None  # Not implemented for MVP

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
