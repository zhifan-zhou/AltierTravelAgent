"""FastAPI app for the v0.4 product-grade web prototype."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from travel_agent.config import load_settings
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.deepseek_client import DeepSeekClient, DeepSeekRequirementAgent
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.pipeline.types import ChatTurnResult
from travel_agent.planning.models import ConstraintCheckResult, SourceRef, UserResponse
from travel_agent.server.api_models import (
    SERVICE_NAME,
    VERSION,
    ChatRequest,
    ChatResponse,
    DeleteSessionResponse,
    HealthResponse,
    SessionCreateResponse,
    SessionListItem,
    SessionSnapshot,
    contains_forbidden_marker,
)
from travel_agent.server.session_store import SessionStore
from travel_agent.server.stream import sse_event, user_response_token_events


WEB_DIR = Path(__file__).resolve().parents[1] / "web"
SAFE_ERROR = "Could not process request. Please try again with a simpler travel request."
SessionFactory = Callable[[bool], LLMFirstChatSession]


def create_app(
    *,
    store: SessionStore | None = None,
    session_factory: SessionFactory | None = None,
) -> FastAPI:
    session_store = store or SessionStore()
    runtime_sessions: dict[str, LLMFirstChatSession] = {}

    app = FastAPI(title=SERVICE_NAME, version=VERSION)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    def build_session(debug: bool = False) -> LLMFirstChatSession:
        if session_factory is not None:
            return session_factory(debug)
        return _default_session(debug=debug)

    def ensure_record(session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return session_store.create_session()
        record = session_store.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return record

    def runtime_for(record: dict[str, Any], *, debug: bool = False) -> LLMFirstChatSession:
        session_id = record["session_id"]
        session = runtime_sessions.get(session_id)
        if session is None or (debug and not session.debug):
            session = build_session(debug)
            _hydrate_session(session, record)
            runtime_sessions[session_id] = session
        return session

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(status_code=exc.status_code, content={"message": detail})

    @app.exception_handler(Exception)
    async def exception_handler(_request, _exc: Exception):
        return JSONResponse(status_code=500, content={"message": SAFE_ERROR})

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    @app.post("/api/sessions", response_model=SessionCreateResponse)
    async def create_session():
        record = session_store.create_session()
        runtime_sessions[record["session_id"]] = build_session(False)
        return SessionCreateResponse(
            session_id=record["session_id"],
            created_at=record["created_at"],
            contract_summary=record.get("contract_summary") or {},
        )

    @app.get("/api/sessions", response_model=list[SessionListItem])
    async def list_sessions():
        return [
            SessionListItem(
                session_id=item["session_id"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                message_count=len(item.get("messages") or []),
                contract_summary=item.get("contract_summary") or {},
            )
            for item in session_store.list_sessions()
        ]

    @app.get("/api/sessions/{session_id}", response_model=SessionSnapshot)
    async def get_session(session_id: str):
        record = session_store.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return SessionSnapshot(
            session_id=record["session_id"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            messages=record.get("messages") or [],
            contract_summary=record.get("contract_summary") or {},
            cards=record.get("cards") or [],
        )

    @app.delete("/api/sessions/{session_id}", response_model=DeleteSessionResponse)
    async def delete_session(session_id: str):
        runtime_sessions.pop(session_id, None)
        if not session_store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        return DeleteSessionResponse(status="deleted")

    @app.get("/api/sessions/{session_id}/contract")
    async def get_contract_summary(session_id: str):
        record = session_store.get_session(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"session_id": session_id, "contract_summary": record.get("contract_summary") or {}}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, debug: bool = Query(False)):
        record = ensure_record(payload.session_id)
        session = runtime_for(record, debug=debug)
        try:
            result = await session.handle_user_message(payload.message)
            response = _build_chat_response(
                session_id=record["session_id"],
                session=session,
                result=result,
                previous_cards=record.get("cards") or [],
            )
        except HTTPException:
            raise
        except Exception:
            response = _error_chat_response(record["session_id"], record)
        _persist_chat_response(session_store, payload.message, response, session)
        return response

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest, debug: bool = Query(False)):
        record = ensure_record(payload.session_id)
        session = runtime_for(record, debug=debug)

        async def events():
            try:
                result = await session.handle_user_message(payload.message)
                response = _build_chat_response(
                    session_id=record["session_id"],
                    session=session,
                    result=result,
                    previous_cards=record.get("cards") or [],
                )
                _persist_chat_response(session_store, payload.message, response, session)
                user_response = result.user_response or UserResponse(
                    text=response.assistant_response,
                    response_type=response.response_type,
                )
                for event in user_response_token_events(user_response):
                    yield event
                    await asyncio.sleep(0)
                yield sse_event(
                    "final",
                    {
                        "assistant_response": response.assistant_response,
                        "response_type": response.response_type,
                        "contract_summary": response.contract_summary,
                        "cards": response.cards,
                        "sources": response.sources,
                        "warnings": response.warnings,
                    },
                )
            except Exception:
                yield sse_event("error", {"message": SAFE_ERROR})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _default_session(*, debug: bool = False) -> LLMFirstChatSession:
    settings = load_settings()
    llm = DeepSeekClient(settings) if settings.deepseek_configured else FakeRequirementLLM()
    return LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(llm),
        debug=debug,
        debug_dir=(settings.project_root / ".local" / "debug") if debug else None,
    )


def _hydrate_session(session: LLMFirstChatSession, record: dict[str, Any]) -> None:
    contract_json = record.get("contract_json") or {}
    if contract_json:
        try:
            session.contract = TravelRequirementContract.model_validate(contract_json).normalize()
            session.history_summary = session.contract.summary_zh()
        except Exception:
            session.contract = None
    for message in record.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            session.messages.append({"role": role, "content": content})


def _build_chat_response(
    *,
    session_id: str,
    session: LLMFirstChatSession,
    result: ChatTurnResult,
    previous_cards: list[dict[str, Any]],
) -> ChatResponse:
    user_response = result.user_response or UserResponse(text=result.message)
    contract = result.contract or session.contract
    summary = contract_summary(contract)
    new_cards = cards_from_session(session, result)
    cards = merge_cards(previous_cards, new_cards)
    sources = source_refs_to_dicts(user_response.sources)
    warnings = list(dict.fromkeys([*user_response.warnings, *summary.get("warnings", [])]))
    response = ChatResponse(
        session_id=session_id,
        assistant_response=result.message,
        response_type=user_response.response_type,
        contract_summary=summary,
        cards=cards,
        sources=sources,
        warnings=warnings,
    )
    if contains_forbidden_marker(response.model_dump(mode="json")):
        response.assistant_response = "这次响应包含不应展示的诊断内容，已安全隐藏。请重试。"
        response.cards = [card for card in response.cards if not contains_forbidden_marker(card)]
        response.sources = [source for source in response.sources if not contains_forbidden_marker(source)]
        response.warnings = [warning for warning in response.warnings if not contains_forbidden_marker(warning)]
    return response


def _error_chat_response(session_id: str, record: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        assistant_response=SAFE_ERROR,
        response_type="error",
        contract_summary=record.get("contract_summary") or {},
        cards=record.get("cards") or [],
        sources=record.get("sources") or [],
        warnings=["No booking or payment action was taken."],
    )


def _persist_chat_response(
    store: SessionStore,
    user_message: str,
    response: ChatResponse,
    session: LLMFirstChatSession,
) -> None:
    contract_json = session.contract.model_dump(mode="json") if session.contract else {}
    store.save_turn(
        session_id=response.session_id,
        user_message=user_message,
        assistant_response=response.assistant_response,
        contract_json=contract_json,
        contract_summary=response.contract_summary,
        cards=response.cards,
        sources=response.sources,
        warnings=response.warnings,
    )


def contract_summary(contract: TravelRequirementContract | None) -> dict[str, Any]:
    if contract is None:
        return {
            "route": {},
            "dates": {},
            "budget": {},
            "companions": {},
            "preferences": {},
            "missing_fields": ["trip.origin_airport", "trip.destination_airport", "time.departure_window"],
            "warnings": [],
        }
    pets = [
        {"kind": pet.kind, "count": pet.count, "size": pet.size or "unknown"}
        for pet in contract.companions.pets
        if pet.active
    ]
    warnings = []
    if contract.pending.pending_question:
        warnings.append(contract.pending.pending_question)
    warnings.extend(question for question in contract.unresolved_questions if question)
    for req in contract.special_requirements:
        if req.active and req.requires_clarification and req.clarification_question_zh:
            warnings.append(req.clarification_question_zh)
    summary = {
        "route": {
            "origin": _location_label(
                code=contract.trip.origin_airport,
                city=contract.trip.origin_city,
                text=contract.trip.origin_text,
            ),
            "destination": _location_label(
                code=contract.trip.destination_airport,
                city=contract.trip.destination_city,
                text=contract.trip.destination_text,
            ),
        },
        "dates": {
            "departure": _first_text(
                contract.time.departure_window_text,
                contract.time.departure_text,
                contract.time.departure_start_date,
            ),
            "return_date": contract.time.return_date or "",
            "duration_days": contract.time.duration_days or "",
            "flexibility": contract.time.date_flexibility
            or ("flexible" if contract.time.flexible else "fixed"),
        },
        "budget": {
            "amount": contract.budget.amount or "",
            "currency": contract.budget.currency or "USD",
            "level": contract.budget.priority or "medium",
            "preference": contract.budget.preference or "",
        },
        "companions": {
            "adults": contract.companions.adults,
            "children": contract.companions.children,
            "seniors": contract.companions.seniors,
            "pets": pets,
        },
        "preferences": {
            "avoid_red_eye": contract.preferences.avoid_red_eye,
            "nonstop_preferred": contract.preferences.nonstop_preferred,
            "max_stops": contract.preferences.max_stops if contract.preferences.max_stops is not None else "",
            "ranking_profile": contract.ranking.profile,
            "prefer_major_airlines": contract.airline_preferences.prefer_major_airlines,
            "avoid_airports": contract.geography.avoid_airports,
        },
        "missing_fields": contract.pending.missing_fields or contract.missing_mandatory_search_fields(),
        "warnings": list(dict.fromkeys(warnings)),
    }
    return _drop_empty(summary)


def cards_from_session(
    session: LLMFirstChatSession,
    result: ChatTurnResult,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if result.pipeline_result and result.pipeline_result.recommendations:
        cards.append(_flight_card(result.pipeline_result.recommendations))
    if session.last_itinerary:
        cards.append(_itinerary_card(session.last_itinerary))
    if session.last_cost_estimate:
        cards.append(_cost_card(session.last_cost_estimate))
    constraint_card = _constraint_card(session)
    if constraint_card:
        cards.append(constraint_card)
    cards.extend(_source_cards(session, result))
    cards.append(
        {
            "type": "safety",
            "title": "Safety boundary",
            "items": [
                "Flight prices are demo/mock only unless explicitly labeled live.",
                "No booking, payment, ticketing, or price lock.",
                "Live tools: Open-Meteo / Frankfurter where available.",
            ],
        }
    )
    return [card for card in cards if not contains_forbidden_marker(card)]


def merge_cards(
    previous_cards: list[dict[str, Any]],
    new_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    replace_types = {"flight_demo", "itinerary", "cost_estimate", "constraint_check", "safety"}
    for card in previous_cards:
        if card.get("type") not in replace_types and card.get("type") != "source":
            merged.append(card)
    source_keys: set[tuple[str, str]] = set()
    for card in [*previous_cards, *new_cards]:
        if card.get("type") == "source":
            key = (str(card.get("title", "")), str(card.get("source", "")))
            if key not in source_keys:
                source_keys.add(key)
                merged.append(card)
    for card in new_cards:
        if card.get("type") in replace_types:
            merged = [item for item in merged if item.get("type") != card.get("type")]
            merged.append(card)
    return [card for card in merged if not contains_forbidden_marker(card)]


def source_refs_to_dicts(sources: list[SourceRef]) -> list[dict[str, Any]]:
    return [
        {
            "label": item.label,
            "source": item.source,
            "is_live": item.is_live,
            "note": item.note or "",
        }
        for item in sources
    ]


def _flight_card(recommendations) -> dict[str, Any]:
    items = []
    for rec in recommendations[:3]:
        itinerary = rec.itinerary
        items.append(
            {
                "rank": rec.rank,
                "route": " → ".join(itinerary.route),
                "airlines": ", ".join(itinerary.airlines) or "mixed/mock",
                "price_usd": round(itinerary.total_price_usd, 2),
                "risk": rec.risk.risk_level,
                "classification": "mock_demo",
                "bookable": False,
            }
        )
    return {
        "type": "flight_demo",
        "title": "Demo flight options",
        "classification": "mock_demo",
        "safety_label": "Demo/mock only. Not real price, not bookable.",
        "items": items,
    }


def _itinerary_card(plan) -> dict[str, Any]:
    return {
        "type": "itinerary",
        "title": f"{plan.duration_days}-day {plan.destination} draft",
        "classification": "estimate",
        "items": [
            {
                "day": day.day,
                "title": day.title,
                "morning": day.morning,
                "afternoon": day.afternoon,
                "evening": day.evening,
                "notes": day.notes,
                "weather_considerations": day.weather_considerations,
                "budget_level": day.budget_level,
            }
            for day in plan.days
        ],
        "warnings": plan.warnings,
    }


def _cost_card(estimate) -> dict[str, Any]:
    return {
        "type": "cost_estimate",
        "title": "Rough budget estimate",
        "classification": "estimate",
        "currency": estimate.currency,
        "total": {
            "min": estimate.total_min,
            "max": estimate.total_max,
            "currency": estimate.currency,
        },
        "items": [
            {
                "category": item.category,
                "min": item.amount_min,
                "max": item.amount_max,
                "currency": item.currency,
                "confidence": item.confidence,
                "source_type": item.source_type,
                "note": item.note,
            }
            for item in estimate.items
        ],
        "warnings": estimate.warnings,
    }


def _constraint_card(session: LLMFirstChatSession) -> dict[str, Any] | None:
    if not session.contract:
        return None
    checks: ConstraintCheckResult = session.constraint_checker.check(
        session.contract,
        cost_estimate=session.last_cost_estimate,
        weather_result=session.last_tool_results.get("weather"),
        pipeline_result=session.last_result,
    )
    if not checks.findings:
        return None
    return {
        "type": "constraint_check",
        "title": "Travel constraints and reminders",
        "items": [
            {
                "category": item.category,
                "level": item.level,
                "message": item.message,
                "evidence_type": item.evidence_type,
            }
            for item in checks.findings
        ],
    }


def _source_cards(session: LLMFirstChatSession, result: ChatTurnResult) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    tool_results = list(session.last_tool_results.values())
    for raw in result.tool_results:
        if not any(item.tool_name == raw.get("tool_name") for item in tool_results):
            continue
    for item in tool_results:
        if not item.source:
            continue
        cards.append(
            {
                "type": "source",
                "title": f"{item.tool_name} source",
                "source": item.source,
                "status": item.status,
                "is_live": item.is_live,
                "classification": "live" if item.is_live else "mock_demo" if item.is_mock else "tool",
            }
        )
    if session.last_result and session.last_result.recommendations:
        cards.append(
            {
                "type": "source",
                "title": "Flight source",
                "source": "mock flight provider",
                "status": "ok",
                "is_live": False,
                "classification": "mock_demo",
                "note": "Demo/mock only. Not real price, not bookable.",
            }
        )
    return cards


def _location_label(*, code: str | None, city: str | None, text: str | None) -> str:
    label = city or text or ""
    if code and label:
        return f"{label} ({code})"
    return code or label


def _first_text(*items: str | None) -> str:
    for item in items:
        if item:
            return item
    return ""


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = _drop_empty(item)
            if cleaned in ({}, [], None):
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if _drop_empty(item) not in ({}, [], None)]
    if value is None:
        return ""
    return value


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("travel_agent.server.app:app", host="127.0.0.1", port=8000, reload=True)
