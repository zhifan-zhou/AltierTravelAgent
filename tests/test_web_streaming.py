from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.helpers_v03 import make_v03_session
from travel_agent.server.api_models import FORBIDDEN_WEB_MARKERS
from travel_agent.server.app import create_app
from travel_agent.server.session_store import SessionStore


def test_streaming_chat_returns_token_and_final_events(tmp_path):
    app = create_app(
        store=SessionStore(tmp_path / "sessions"),
        session_factory=lambda debug=False: make_v03_session(debug=debug),
    )
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["session_id"]

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"session_id": session_id, "message": "帮我安排奥斯丁三天行程，预算低一点"},
    ) as response:
        assert response.status_code == 200
        text = response.read().decode("utf-8")

    events = _parse_sse(text)
    token_text = "".join(event["data"]["text"] for event in events if event["event"] == "token")
    final = next(event["data"] for event in events if event["event"] == "final")

    assert token_text
    assert token_text == final["assistant_response"]
    assert "itinerary" in {card["type"] for card in final["cards"]}
    assert final["contract_summary"]
    for marker in FORBIDDEN_WEB_MARKERS:
        assert marker not in text


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in [item for item in text.split("\n\n") if item.strip()]:
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append({"event": event, "data": json.loads(data)})
    return events
