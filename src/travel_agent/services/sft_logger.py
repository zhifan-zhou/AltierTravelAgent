"""SFT data logger for chat sessions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from travel_agent.config import load_settings


class SFTLogger:
    def __init__(self, root: Path | None = None):
        settings = load_settings()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root or settings.runs_dir / "conversation_data" / ts)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.messages: list[dict[str, str]] = []
        self.turns_path = self.session_dir / "turns.jsonl"
        self.session_path = self.session_dir / "session.jsonl"

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def log_turn(
        self,
        *,
        previous_contract: dict[str, Any] | None,
        target_update: dict[str, Any],
        target_contract_after_update: dict[str, Any],
        result_summary: dict[str, Any] | None = None,
        quality_flags: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "messages": list(self.messages),
            "previous_contract": previous_contract,
            "target_update": target_update,
            "target_contract_after_update": target_contract_after_update,
            "decision_trace": target_update.get("decision_trace", []),
            "result_summary": result_summary or {},
            "quality_flags": quality_flags or {},
        }
        with self.turns_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_session(self, *, target_final_contract: dict[str, Any]) -> None:
        row = {
            "messages": list(self.messages),
            "target_final_contract": target_final_contract,
        }
        with self.session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
