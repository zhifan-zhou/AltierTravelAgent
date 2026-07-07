"""Deterministic mock flight provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.config import load_settings
from travel_agent.contract.compiler import ExclusionRules
from travel_agent.pipeline.types import FlightOffer, FlightSegment, SearchTask
from travel_agent.services.airline_service import AirlineService


SECTION_BY_LEG = {
    "direct": "direct",
    "international": "international",
    "domestic_us": "domestic_us",
    "ground_cn": "domestic_cn",
}


class MockFlightProvider:
    """Reads mock data first, then creates clearly marked deterministic estimates."""

    def __init__(self, data_dir: Path | None = None, airline_service: AirlineService | None = None):
        settings = load_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.airlines = airline_service or AirlineService(self.data_dir)
        self.data = json.loads((self.data_dir / "mock_flights.json").read_text(encoding="utf-8"))["flights"]
        self.calls: list[SearchTask] = []

    def reset_calls(self) -> None:
        self.calls = []

    def search(self, task: SearchTask, exclusions: ExclusionRules) -> list[FlightOffer]:
        if exclusions.airport_is_excluded(task.origin) or exclusions.airport_is_excluded(task.destination):
            return []
        self.calls.append(task)
        key = f"{task.origin}->{task.destination}"
        section = SECTION_BY_LEG.get(task.leg_type, "direct")
        raw = self.data.get(section, {}).get(key)
        if raw and raw.get("offers"):
            return [self._from_raw(task, item) for item in raw["offers"][:3] if self._raw_allowed(item, exclusions)]
        if raw and raw.get("access_estimate_only"):
            return [
                FlightOffer(
                    id=f"mock-access-{task.origin.lower()}-{task.destination.lower()}",
                    task_id=task.task_id,
                    leg_type=task.leg_type,
                    origin=task.origin,
                    destination=task.destination,
                    segments=[
                        FlightSegment(
                            origin=task.origin,
                            destination=task.destination,
                            airline="GROUND",
                            airline_name=raw.get("mode", "ground access"),
                            mode="ground",
                        )
                    ],
                    total_price_usd=float(raw.get("estimated_cost_usd", 40)),
                    estimated_time_hours=3.0,
                    source="mock_access_estimate",
                    confidence="estimated",
                    booking_available=False,
                )
            ]
        return [self._fallback(task)]

    def _from_raw(self, task: SearchTask, raw: dict[str, Any]) -> FlightOffer:
        segments = [
            FlightSegment(
                origin=s["origin"].upper(),
                destination=s["destination"].upper(),
                airline=s.get("airline"),
                airline_name=s.get("airline_name"),
                flight_number=s.get("flight_number"),
                mode="flight",
                departure_time=s.get("departure_time"),
                arrival_time=s.get("arrival_time"),
            )
            for s in raw.get("segments", [])
        ]
        return FlightOffer(
            id=raw.get("id", f"mock-{task.origin}-{task.destination}"),
            task_id=task.task_id,
            leg_type=task.leg_type,
            origin=task.origin,
            destination=task.destination,
            segments=segments,
            total_price_usd=float(raw.get("total_price_usd", 0)),
            estimated_time_hours=_estimate_offer_hours(task, segments),
            source=raw.get("source", "mock"),
            confidence="known",
            booking_available=False,
        )

    def _raw_allowed(self, raw: dict[str, Any], exclusions: ExclusionRules) -> bool:
        for segment in raw.get("segments", []):
            if exclusions.airport_is_excluded(segment.get("origin")):
                return False
            if exclusions.airport_is_excluded(segment.get("destination")):
                return False
            if exclusions.airline_is_excluded(segment.get("airline")):
                return False
        return True

    def _fallback(self, task: SearchTask) -> FlightOffer:
        price = _fallback_price(task)
        airline = _fallback_airline(task)
        return FlightOffer(
            id=f"mock-fallback-{task.leg_type}-{task.origin.lower()}-{task.destination.lower()}",
            task_id=task.task_id,
            leg_type=task.leg_type,
            origin=task.origin,
            destination=task.destination,
            segments=[
                FlightSegment(
                    origin=task.origin,
                    destination=task.destination,
                    airline=airline,
                    airline_name=self.airlines.name(airline) if airline != "GROUND" else "ground access estimate",
                    flight_number=None,
                    mode="ground" if task.leg_type == "ground_cn" else "flight",
                )
            ],
            total_price_usd=price,
            estimated_time_hours=_estimate_offer_hours(task, []),
            source="mock_fallback",
            confidence="estimated",
            booking_available=False,
        )


def _fallback_price(task: SearchTask) -> float:
    seed = sum(ord(ch) for ch in f"{task.leg_type}:{task.origin}:{task.destination}")
    if task.leg_type == "ground_cn":
        return 25 + seed % 45
    if task.leg_type == "domestic_us":
        return 120 + seed % 150
    if task.leg_type == "international":
        return 650 + seed % 360
    return 1500 + seed % 650


def _fallback_airline(task: SearchTask) -> str:
    if task.leg_type == "ground_cn":
        return "GROUND"
    if task.leg_type == "domestic_us":
        return ["UA", "DL", "AA"][sum(ord(c) for c in task.destination) % 3]
    if task.leg_type == "international":
        return ["MU", "CA", "UA", "DL"][sum(ord(c) for c in task.origin + task.destination) % 4]
    return "MOCK"


def _estimate_offer_hours(task: SearchTask, segments: list[FlightSegment]) -> float:
    if task.leg_type == "ground_cn":
        return 3.0
    if task.leg_type == "domestic_us":
        return 2.0
    if task.leg_type == "international":
        return 14.0
    return max(2.0, len(segments) * 3.0)
