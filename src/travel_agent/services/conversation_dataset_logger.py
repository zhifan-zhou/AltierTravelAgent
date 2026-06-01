"""Logs conversation + contract data for future SFT training."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("runs/conversation_data")


class ConversationDatasetLogger:
    """Save SFT-ready conversation data."""

    def __init__(self):
        self._messages: list[dict] = []
        self._contract_updates: list[dict] = []
        self._initial_contract: dict | None = None

    def log_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def log_contract_update(self, user_message: str, updated_fields: list[str],
                            contract_after: dict) -> None:
        self._contract_updates.append({
            "user_message": user_message,
            "updated_fields": updated_fields,
            "contract_after_update": contract_after,
        })

    def set_initial_contract(self, contract: dict) -> None:
        self._initial_contract = dict(contract)

    def save(self, final_contract: dict, search_result_summary: dict | None = None) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / ts
        out.mkdir(parents=True, exist_ok=True)

        sample = {
            "schema_version": "v1",
            "conversation": list(self._messages),
            "final_contract": final_contract,
            "initial_contract": self._initial_contract,
            "contract_updates": list(self._contract_updates),
            "search_result_summary": search_result_summary or {},
            "quality_flags": {
                "contract_valid": True,
                "hard_constraints_respected": True,
                "needs_human_review": False,
            },
        }

        (out / "sample.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False, default=str))

        # JSONL for SFT
        sft_lines = []
        for update in self._contract_updates:
            sft_lines.append(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are a travel requirement extraction assistant."},
                    {"role": "user", "content": update["user_message"]},
                ],
                "target_schema": update["contract_after_update"],
            }, ensure_ascii=False))

        if sft_lines:
            (out / "sft_samples.jsonl").write_text("\n".join(sft_lines) + "\n")

        return out
