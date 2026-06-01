"""Airport and nearby hub data access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.config import load_settings
from travel_agent.contract.normalization import expand_location


class AirportService:
    def __init__(self, data_dir: Path | None = None):
        settings = load_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.airports = self._load_airports()
        self.nearby = self._load_nearby_hubs()
        self.alias_map = self._build_alias_map()

    def _load_airports(self) -> dict[str, dict[str, Any]]:
        rows = json.loads((self.data_dir / "airports.json").read_text(encoding="utf-8"))
        return {row["code"].upper(): row for row in rows}

    def _load_nearby_hubs(self) -> dict[str, Any]:
        path = self.data_dir / "nearby_hubs_us_china.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _build_alias_map(self) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}

        def add(alias: str, code: str) -> None:
            key = str(alias).strip().lower()
            if not key:
                return
            values = aliases.setdefault(key, [])
            if code not in values:
                values.append(code)

        for code, row in self.airports.items():
            add(code, code)
            add(row.get("city", ""), code)
            add(row.get("city_cn", ""), code)
            for alias in row.get("aliases", []):
                add(str(alias), code)
        for special in ["上海", "上海浦东", "上海虹桥", "纽约", "new york", "nyc", "成都", "chengdu"]:
            matches = expand_location(special)
            if matches:
                aliases[special.lower()] = matches
        for key, codes in list(aliases.items()):
            aliases[key] = self.preferred_first(codes)
        return aliases

    def exists(self, code: str | None) -> bool:
        return bool(code) and code.upper() in self.airports

    def get(self, code: str | None) -> dict[str, Any] | None:
        return self.airports.get(code.upper()) if code else None

    def city(self, code: str | None) -> str:
        row = self.get(code)
        return row.get("city", "") if row else ""

    def city_cn(self, code: str | None) -> str:
        row = self.get(code)
        return row.get("city_cn", "") if row else ""

    def country(self, code: str | None) -> str:
        row = self.get(code)
        return row.get("country", "") if row else ""

    def resolve(self, text: str | None) -> list[str]:
        if not text:
            return []
        raw = text.strip()
        if len(raw) == 3 and raw.upper() in self.airports:
            return [raw.upper()]
        lowered = raw.lower()
        if lowered in self.alias_map:
            return list(self.alias_map[lowered])
        matches = expand_location(raw)
        if matches:
            return self.preferred_first(matches)
        return []

    def resolve_location(self, text: str | None) -> list[str]:
        return self.resolve(text)

    def preferred_first(self, codes: list[str]) -> list[str]:
        unique: list[str] = []
        for code in codes:
            code = str(code).upper()
            if code and code not in unique:
                unique.append(code)
        return sorted(unique, key=lambda code: (not bool(self.airports.get(code, {}).get("preferred")), code))

    def preferred_airport(self, codes: list[str]) -> str | None:
        ordered = self.preferred_first(codes)
        return ordered[0] if ordered else None

    def nearby_origin_hubs(self, origin: str) -> list[dict[str, Any]]:
        return list(self.nearby.get("origin_hubs", {}).get(origin.upper(), []))

    def nearby_destination_hubs(self, destination: str) -> list[dict[str, Any]]:
        return list(self.nearby.get("destination_hubs", {}).get(destination.upper(), []))

    def all_us_hub_defaults(self) -> list[str]:
        return ["JFK", "EWR", "IAD", "ORD", "ATL", "DFW", "LAX", "SFO", "SEA", "BOS", "PHL"]

    def all_china_hub_defaults(self) -> list[str]:
        return ["PVG", "SHA", "HGH", "NGB", "WNZ"]
