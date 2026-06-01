"""Airline metadata for ranking and display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.config import load_settings


class AirlineService:
    def __init__(self, data_dir: Path | None = None):
        settings = load_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        path = self.data_dir / "airlines.json"
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.airlines: dict[str, dict[str, Any]] = raw.get("airlines", {})
        self.quality_map = raw.get("quality_score_map", {})

    def name(self, code: str | None) -> str:
        if not code:
            return "未知"
        row = self.airlines.get(code.upper())
        return row.get("name_zh") or row.get("name") if row else code.upper()

    def display_name(self, code: str | None, *, compact: bool = False) -> str:
        if not code:
            return "未知"
        code = code.upper()
        if code == "GROUND":
            return "地面接驳"
        row = self.airlines.get(code)
        if not row:
            return code
        zh = row.get("name_zh") or code
        en = row.get("name")
        if compact:
            return zh
        return f"{zh} ({en})" if en and en != zh else zh

    def quality_score(self, code: str | None) -> float:
        if not code:
            return 0.5
        row = self.airlines.get(code.upper())
        if not row:
            return 0.5
        return float(row.get("reliability_score") or self.quality_map.get(row.get("quality_tier"), 0.5))

    def is_low_cost(self, code: str | None) -> bool:
        row = self.airlines.get((code or "").upper())
        return bool(row and row.get("airline_type") == "low_cost")

    def is_major(self, code: str | None) -> bool:
        row = self.airlines.get((code or "").upper())
        return bool(row and row.get("quality_tier") in {"major", "premium"})
