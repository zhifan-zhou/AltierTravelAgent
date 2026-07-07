"""Local JSON session persistence for the web prototype.

The store intentionally keeps only user-visible transcript data, sanitized UI
state, and the minimum contract JSON needed to resume a planning session. It
does not persist API keys, debug traces, prompts, raw tool requests, or raw tool
results.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from travel_agent.config import load_settings


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
SENSITIVE_KEY_PARTS = ("api_key", "token", "secret", "bearer", "password")


class SessionStore:
    def __init__(self, root: Path | str | None = None):
        if root is None:
            settings = load_settings()
            root = settings.project_root / ".local" / "sessions"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> dict[str, Any]:
        now = _now()
        session_id = uuid4().hex
        record = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "contract_json": {},
            "contract_summary": {},
            "cards": [],
            "sources": [],
            "warnings": [],
        }
        self._write(record)
        return record

    def list_sessions(self) -> list[dict[str, Any]]:
        records = [self.get_session(path.stem) for path in self.root.glob("*.json")]
        return sorted(
            [item for item in records if item],
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        if not self._valid_session_id(session_id):
            return None
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if data.get("session_id") != session_id:
            return None
        return data

    def delete_session(self, session_id: str) -> bool:
        if not self._valid_session_id(session_id):
            return False
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def save_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        contract_json: dict[str, Any] | None,
        contract_summary: dict[str, Any],
        cards: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        warnings: list[str],
    ) -> dict[str, Any]:
        record = self.get_session(session_id)
        if record is None:
            raise KeyError(f"unknown session: {session_id}")
        now = _now()
        record.setdefault("messages", []).append(
            {"role": "user", "content": user_message, "created_at": now}
        )
        record["messages"].append(
            {"role": "assistant", "content": assistant_response, "created_at": now}
        )
        record["contract_json"] = _strip_sensitive(contract_json or {})
        record["contract_summary"] = _strip_sensitive(contract_summary)
        record["cards"] = _strip_sensitive(cards)
        record["sources"] = _strip_sensitive(sources)
        record["warnings"] = _strip_sensitive(warnings)
        record["updated_at"] = now
        self._write(record)
        return record

    def save_state(
        self,
        *,
        session_id: str,
        contract_json: dict[str, Any] | None = None,
        contract_summary: dict[str, Any] | None = None,
        cards: list[dict[str, Any]] | None = None,
        sources: list[dict[str, Any]] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        record = self.get_session(session_id)
        if record is None:
            raise KeyError(f"unknown session: {session_id}")
        if contract_json is not None:
            record["contract_json"] = _strip_sensitive(contract_json)
        if contract_summary is not None:
            record["contract_summary"] = _strip_sensitive(contract_summary)
        if cards is not None:
            record["cards"] = _strip_sensitive(cards)
        if sources is not None:
            record["sources"] = _strip_sensitive(sources)
        if warnings is not None:
            record["warnings"] = _strip_sensitive(warnings)
        record["updated_at"] = _now()
        self._write(record)
        return record

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(record["session_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _valid_session_id(self, session_id: str) -> bool:
        return bool(SESSION_ID_RE.fullmatch(session_id or ""))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _strip_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                continue
            result[str(key)] = _strip_sensitive(item)
        return result
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    return value
