"""Mock flight provider with exact data + deterministic fallback pricing."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from travel_agent.core.config import get_settings
from travel_agent.models.flight import CabinClass, FlightOffer, FlightSearchRequest, FlightSegment
from travel_agent.providers.base import BaseFlightProvider, ProviderCapabilities, ProviderConfigurationError
from travel_agent.services.airport_service import AirportService


FALLBACK_BASE_PRICES: dict[str, float] = {
    "international": 800.0,
    "domestic": 180.0,
}

FALLBACK_AIRLINES: dict[str, list[dict]] = {
    "international": [
        {"code": "MU", "name": "China Eastern"}, {"code": "CZ", "name": "China Southern"},
        {"code": "CA", "name": "Air China"}, {"code": "UA", "name": "United Airlines"},
        {"code": "DL", "name": "Delta Air Lines"}, {"code": "HU", "name": "Hainan Airlines"},
        {"code": "CX", "name": "Cathay Pacific"}, {"code": "NH", "name": "ANA"},
        {"code": "JL", "name": "Japan Airlines"}, {"code": "KE", "name": "Korean Air"},
    ],
    "domestic": [
        {"code": "UA", "name": "United Airlines"}, {"code": "DL", "name": "Delta Air Lines"},
        {"code": "AA", "name": "American Airlines"}, {"code": "B6", "name": "JetBlue"},
        {"code": "AS", "name": "Alaska Airlines"}, {"code": "WN", "name": "Southwest Airlines"},
    ],
}


class MockFlightProvider(BaseFlightProvider):
    """Mock provider with exact data + deterministic fallback pricing."""

    def __init__(self, data_path: str | Path | None = None,
                 airport_service: AirportService | None = None):
        if data_path is None:
            settings = get_settings()
            data_path = settings.data_path / "mock_flights.json"
        self._data_path = Path(data_path)
        self._airport_service = airport_service or AirportService()
        settings = get_settings()
        self._enable_fallback = getattr(settings, 'mock_provider_enable_fallback', True)
        self._data: dict = {}
        self._load_data()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="mock",
            supports_search=True,
            supports_booking=False,
            supports_price_verify=False,
            requires_api_key=False,
            is_real_provider=False,
            max_results_per_search=20,
            typical_latency_ms=5,
        )

    def validate_config(self) -> None:
        if not self._data_path.exists():
            raise ProviderConfigurationError("mock", f"Data file not found: {self._data_path}")

    def _load_data(self) -> None:
        with open(self._data_path) as f:
            self._data = json.load(f)

    async def search_flights(self, request: FlightSearchRequest) -> list[FlightOffer]:
        origin = request.origin.upper()
        dest = request.destination.upper()
        route_key = f"{origin}->{dest}"
        results: list[FlightOffer] = []

        for category in ("direct", "international", "domestic_us"):
            cat_data = self._data.get("flights", {}).get(category, {})
            if route_key in cat_data:
                for entry in cat_data[route_key].get("offers", []):
                    if self._cabin_matches(entry, request.cabin):
                        offer = self._parse_offer(entry)
                        offer.source = "mock_exact"
                        offer.is_real = False
                        offer.confidence = "demo"
                        offer.provider_name = "mock"
                        results.append(offer)

        if not results and self._enable_fallback:
            fb = self._generate_fallback(origin, dest, request.cabin)
            if fb:
                results.append(fb)

        if request.cabin != CabinClass.ECONOMY:
            results = [r for r in results
                       if any(s.cabin == request.cabin for s in r.segments)]

        return results

    def _generate_fallback(self, origin: str, dest: str, cabin: CabinClass) -> FlightOffer | None:
        origin_apt = self._airport_service.get_airport(origin)
        dest_apt = self._airport_service.get_airport(dest)
        if not origin_apt or not dest_apt:
            return None

        is_intl = origin_apt.country != dest_apt.country
        route_type = "international" if is_intl else "domestic"
        base = FALLBACK_BASE_PRICES.get(route_type, 500.0)
        avg_hub = (self._get_hub_score(origin) + self._get_hub_score(dest)) / 2.0
        hub_mod = 1.0 + (1.0 - avg_hub) * 0.6
        cabin_mult = {"economy": 1.0, "premium_economy": 1.8, "business": 4.0, "first": 8.0}.get(cabin.value, 1.0)
        route_hash = hashlib.md5(f"{origin}{dest}{cabin.value}".encode()).hexdigest()
        hash_val = int(route_hash[:4], 16) / 65536.0
        price = round(base * hub_mod * cabin_mult * (0.85 + hash_val * 0.30), 2)

        departure_day = 15 + (int(route_hash[4:6], 16) % 3)
        departure_hour = 8 + (int(route_hash[6:8], 16) % 14)
        dep_time = datetime(2026, 6, departure_day, departure_hour, 0)
        flight_hours = 12.0 if is_intl else 2.5
        arr_time = dep_time + timedelta(hours=flight_hours)

        airlines = FALLBACK_AIRLINES[route_type]
        airline = airlines[int(route_hash[8:10], 16) % len(airlines)]
        flight_num = f"{airline['code']}{1000 + int(route_hash[10:14], 16) % 9000}"

        segment = FlightSegment(
            origin=origin, destination=dest,
            departure_time=dep_time, arrival_time=arr_time,
            airline=airline["code"], airline_name=airline["name"],
            flight_number=flight_num, cabin=cabin,
        )

        return FlightOffer(
            id=f"mock-fallback-{origin}-{dest}-{cabin.value}-{route_hash[:8]}",
            segments=[segment], total_price_usd=price, currency="USD",
            provider_name="mock", source="mock_fallback",
            is_real=False, confidence="estimated",
            booking_available=False, baggage_included=True,
        )

    def _get_hub_score(self, code: str) -> float:
        airport = self._airport_service.get_airport(code)
        if not airport:
            return 0.3
        return 0.85 if airport.is_international_hub else 0.4

    def _cabin_matches(self, entry: dict, cabin: CabinClass) -> bool:
        return any(seg.get("cabin") == cabin.value for seg in entry.get("segments", []))

    def _parse_offer(self, entry: dict) -> FlightOffer:
        segments = [
            FlightSegment(
                origin=s["origin"], destination=s["destination"],
                departure_time=datetime.fromisoformat(s["departure_time"]),
                arrival_time=datetime.fromisoformat(s["arrival_time"]),
                airline=s.get("airline", ""), airline_name=s.get("airline_name", ""),
                flight_number=s.get("flight_number", ""),
                cabin=CabinClass(s.get("cabin", "economy")),
            )
            for s in entry.get("segments", [])
        ]
        return FlightOffer(
            id=entry["id"], segments=segments,
            total_price_usd=entry["total_price_usd"],
            currency=entry.get("currency", "USD"),
            provider_name="mock", source=entry.get("source", "mock_exact"),
            is_real=False, confidence="demo",
            booking_available=False,
            baggage_included=entry.get("baggage_included", True),
            refundable=entry.get("refundable", False),
        )

    async def verify_price(self, offer_id: str) -> FlightOffer | None:
        return None
