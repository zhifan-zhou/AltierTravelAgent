from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers_v03 import make_v03_session
from travel_agent.server.app import create_app
from travel_agent.server.api_models import FORBIDDEN_WEB_MARKERS
from travel_agent.server.session_store import SessionStore


def _client(tmp_path, *, broken: bool = False) -> TestClient:
    if broken:
        factory = lambda debug=False: BrokenSession()
    else:
        factory = lambda debug=False: make_v03_session(debug=debug)
    app = create_app(store=SessionStore(tmp_path / "sessions"), session_factory=factory)
    return TestClient(app)


class BrokenSession:
    debug = False
    messages = []
    contract = None

    async def handle_user_message(self, _message: str):
        raise RuntimeError("Traceback should not leak")


def test_healthcheck_and_web_index(tmp_path):
    client = _client(tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "version": "v0.4",
        "service": "AltierTravelAgent",
    }

    index = client.get("/")
    assert index.status_code == 200
    assert "AltierTravelAgent" in index.text
    assert "No booking/payment" in index.text


def test_create_session_non_stream_chat_and_resume(tmp_path):
    client = _client(tmp_path)
    created = client.post("/api/sessions").json()

    response = client.post(
        "/api/chat",
        json={"session_id": created["session_id"], "message": "我想从成都飞奥斯丁", "stream": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == created["session_id"]
    assert body["assistant_response"]
    assert body["contract_summary"]["route"]["origin"].endswith("(TFU)")
    assert body["cards"]
    assert_no_forbidden_markers(body)

    snapshot = client.get(f"/api/sessions/{created['session_id']}").json()
    assert len(snapshot["messages"]) == 2
    assert snapshot["contract_summary"]["route"]["destination"].endswith("(AUS)")


def test_itinerary_chat_returns_product_cards(tmp_path):
    client = _client(tmp_path)

    body = client.post(
        "/api/chat",
        json={"message": "帮我安排奥斯丁三天行程，预算低一点", "stream": False},
    ).json()

    card_types = {card["type"] for card in body["cards"]}
    assert "itinerary" in card_types
    assert "constraint_check" in card_types
    assert "source" in card_types
    assert "safety" in card_types
    assert_no_forbidden_markers(body)


def test_invalid_session_and_api_failure_are_clean(tmp_path):
    client = _client(tmp_path)

    missing = client.post(
        "/api/chat",
        json={"session_id": "missing-session", "message": "hello", "stream": False},
    )
    assert missing.status_code == 404
    assert missing.json() == {"message": "Session not found."}

    broken = _client(tmp_path / "broken", broken=True)
    created = broken.post("/api/sessions").json()
    response = broken.post(
        "/api/chat",
        json={"session_id": created["session_id"], "message": "trigger", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert "Could not process request" in body["assistant_response"]
    assert "Traceback" not in str(body)
    assert "RuntimeError" not in str(body)


def assert_no_forbidden_markers(payload):
    text = str(payload)
    for marker in FORBIDDEN_WEB_MARKERS:
        assert marker not in text
