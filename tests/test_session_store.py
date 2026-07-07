from __future__ import annotations

from travel_agent.server.session_store import SessionStore


def test_session_store_create_append_load_and_delete(tmp_path):
    store = SessionStore(tmp_path)
    record = store.create_session()

    assert record["session_id"]
    assert store.get_session(record["session_id"])["messages"] == []

    saved = store.save_turn(
        session_id=record["session_id"],
        user_message="我想从成都飞奥斯丁",
        assistant_response="什么时候出发？",
        contract_json={"trip": {"origin_airport": "TFU"}, "api_key": "should-not-persist"},
        contract_summary={"route": {"origin": "Chengdu (TFU)"}, "token": "should-not-persist"},
        cards=[{"type": "safety", "title": "Safety"}],
        sources=[],
        warnings=[],
    )

    assert len(saved["messages"]) == 2
    loaded = store.get_session(record["session_id"])
    assert loaded["contract_json"] == {"trip": {"origin_airport": "TFU"}}
    assert "api_key" not in str(loaded)
    assert "should-not-persist" not in str(loaded)

    assert store.delete_session(record["session_id"]) is True
    assert store.get_session(record["session_id"]) is None


def test_session_store_temp_dir_isolated(tmp_path):
    store_a = SessionStore(tmp_path / "a")
    store_b = SessionStore(tmp_path / "b")
    session_a = store_a.create_session()

    assert store_a.get_session(session_a["session_id"]) is not None
    assert store_b.get_session(session_a["session_id"]) is None
