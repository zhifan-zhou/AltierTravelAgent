"""Risk assessment service for itineraries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from travel_agent.core.config import get_settings
from travel_agent.models.itinerary import Itinerary
from travel_agent.models.risk import RiskAssessment


class RiskService:
    """Evaluates travel risk for itineraries."""

    def __init__(self, data_path: str | Path | None = None):
        if data_path is None:
            settings = get_settings()
            data_path = settings.data_path
        self._data_path = Path(data_path)
        self._rules: list[dict] = []
        self._risk_levels: dict = {}
        self._load_data()

    def _load_data(self) -> None:
        with open(self._data_path / "risk_rules.json") as f:
            data = json.load(f)
        self._rules = data.get("risk_factors", [])
        self._risk_levels = data.get("risk_levels", {})

    def assess(self, itinerary: Itinerary) -> RiskAssessment:
        """Evaluate all risk factors for an itinerary."""
        total_score = 0.0
        warnings: list[str] = []
        flags: list[str] = []
        details: dict = {}
        has_split = itinerary.split_ticket_count > 0
        is_mock = any(o.provider_name == "mock" for o in itinerary.offers)

        for rule in self._rules:
            triggered = False
            rid = rule["id"]

            if rid == "split_ticket" and has_split:
                triggered = True
            elif rid == "short_connection":
                triggered = self._has_short_connection(itinerary)
            elif rid == "baggage_recheck" and has_split:
                triggered = True
            elif rid == "overnight_stay":
                triggered = self._has_overnight(itinerary)
            elif rid == "airport_transfer":
                triggered = self._has_airport_change(itinerary)
            elif rid == "visa_entry":
                triggered = self._transit_visa_needed(itinerary)
            elif rid == "hidden_city":
                triggered = False  # We don't actively create these
            elif rid == "price_expiration" and is_mock:
                triggered = True

            if triggered:
                total_score += rule.get("base_score", 0.1)
                flags.append(rid)
                if "warning_template" in rule:
                    warnings.append(rule["warning_template"])
                details[rid] = rule.get("description", "")

        total_score = min(total_score, 1.0)

        risk_level = "low"
        for level_name, level_config in sorted(
            self._risk_levels.items(),
            key=lambda x: x[1].get("max_score", 0),
        ):
            if total_score <= level_config.get("max_score", 1.0):
                risk_level = level_name
                break

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=round(total_score, 4),
            warnings=warnings,
            flags=flags,
            split_ticket_risk="split_ticket" in flags,
            short_connection_risk="short_connection" in flags,
            baggage_recheck_risk="baggage_recheck" in flags,
            overnight_stay_risk="overnight_stay" in flags,
            airport_transfer_risk="airport_transfer" in flags,
            visa_entry_risk="visa_entry" in flags,
            hidden_city_risk="hidden_city" in flags,
            price_expiration_risk="price_expiration" in flags,
            details=details,
        )

    def _has_short_connection(self, it: Itinerary) -> bool:
        """Check if any connection is under 120 minutes (for international-to-domestic)."""
        segs = it.segments
        for i in range(len(segs) - 1):
            gap = segs[i + 1].departure_time - segs[i].arrival_time
            if gap.total_seconds() / 60 < 120:
                return True
        return False

    def _has_overnight(self, it: Itinerary) -> bool:
        """Check if connection has a very long layover suggesting overnight."""
        segs = it.segments
        for i in range(len(segs) - 1):
            gap = segs[i + 1].departure_time - segs[i].arrival_time
            if gap.total_seconds() / 3600 > 12:
                return True
        return False

    def _has_airport_change(self, it: Itinerary) -> bool:
        """Check if the itinerary involves a change of airport in the same city."""
        # Simplified: check if we have JFK->EWR or LGA type transfers
        # For MVP, we check if origin of next segment differs from destination of previous
        # when they serve the same city
        return False  # MVP: not common in our generated itineraries

    def _transit_visa_needed(self, it: Itinerary) -> bool:
        """Check if transit visa might be needed."""
        # Simplified: flights through Canada (YYZ, YVR) might need transit visa
        for seg in it.segments:
            if seg.destination in ("YYZ", "YVR") or seg.origin in ("YYZ", "YVR"):
                return True
        return False
