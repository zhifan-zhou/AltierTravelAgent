"""Final acceptance script for the LLM-first TravelAgent architecture with UI/UX validation."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import httpx

from travel_agent.config import load_settings
from travel_agent.llm.deepseek_client import DeepSeekClient, DeepSeekRequirementAgent
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.rendering.response_streamer import ResponseStreamer
from travel_agent.services.display_service import DisplayService
from travel_agent.services.sft_logger import SFTLogger
from travel_agent.tools.http_client import HttpClient
from travel_agent.tools.tool_router import ToolRouter


SCENARIO = [
    "我要从温州去匹兹堡",
    "温州到匹兹堡，六月初，可以从上海走，越便宜越好",
    "我其实想看看宁波到迈阿密，7月初",
    "其实可以去上海浦东，但不去虹桥",
    "主流航司优先",
    "不要纽约转",
    "解释第1个",
    "导出",
]

V03_SCENARIO = [
    "帮我安排奥斯丁三天行程，预算低一点",
    "估算一下预算",
    "我想带猫一起去",
    "检查一下当前约束和风险",
]

FORBIDDEN = [
    "[debug]",
    "ToolRequest",
    "ToolResult",
    "Contract(",
    "schema update",
    "traceback",
    "route_semantics",
    "LLM prompt",
    "raw JSON",
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-deepseek", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = settings.runs_dir / "final_codex_acceptance" / ts
    out.mkdir(parents=True, exist_ok=True)

    llm = DeepSeekClient(settings) if args.real_deepseek else FakeRequirementLLM()
    display = DisplayService()
    session = LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(llm),
        logger=SFTLogger(out / "conversation_data"),
        display=display,
        tool_router=_acceptance_tool_router(settings),
    )

    transcript: list[str] = []
    updates: list[dict] = []
    contracts: list[dict] = []
    validations: list[dict] = []

    # Validate opening screen
    opening_ok = validate_opening_screen(display)

    for idx, message in enumerate(SCENARIO, start=1):
        result = await session.handle_user_message(message)
        transcript.append(f"> {message}\n{result.message}\n")
        updates.append(result.update.model_dump(mode="json") if result.update else {})
        contracts.append(result.contract.model_dump(mode="json") if result.contract else {})
        validations.append(validate_step(idx, result, session, display))

    for offset, message in enumerate(V03_SCENARIO, start=1):
        result = await session.handle_user_message(message)
        transcript.append(f"> {message}\n{result.message}\n")
        updates.append(result.update.model_dump(mode="json") if result.update else {})
        contracts.append(result.contract.model_dump(mode="json") if result.contract else {})
        validations.append(validate_v03_step(8 + offset, result))

    validation = {
        "ok": opening_ok["ok"] and all(item["ok"] for item in validations),
        "opening_screen": opening_ok,
        "steps": validations,
        "mode": "real_deepseek" if args.real_deepseek else "fake_llm",
    }
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "transcript.txt").write_text("\n".join(transcript), encoding="utf-8")
    (out / "updates.json").write_text(json.dumps(updates, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "contracts.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"acceptance_dir: {out}")
    print(f"opening_screen: {'PASS' if opening_ok['ok'] else 'FAIL'}")
    for check in opening_ok.get("checks", []):
        print(f"  {check['name']}: {'PASS' if check['ok'] else 'FAIL'}")
    print(f"validation: {'PASS' if validation['ok'] else 'FAIL'}")
    for item in validations:
        print(f"step {item['step']}: {'PASS' if item['ok'] else 'FAIL'} - {item['name']}")
    if not validation["ok"]:
        raise SystemExit(1)


def validate_opening_screen(display: DisplayService) -> dict:
    text = display.opening_screen_text()
    tips = display.opening_tips()
    checks = [
        ("title shown", "AI 出行管家" in text and "Planning Travel Demo" in text),
        ("subtitle shown", "旅行草案" in text),
        ("numbered examples", all(str(i) in text for i in range(1, 8))),
        ("example 1 route", "温州到匹兹堡" in text),
        ("example 6 explain", "解释第1个" in text),
        ("example 7 export", "导出" in text),
        ("tips shown", "quit" in tips and "改变目的地" in tips),
        (
            "no internal implementation details",
            "DeepSeekRequirementAgent" not in text
            and "TravelRequirementContract" not in text
            and "schema update" not in text,
        ),
    ]
    return {
        "name": "opening_screen",
        "ok": all(ok for _, ok in checks),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }


def validate_step(idx, result, session, display) -> dict:
    checks: list[tuple[str, bool]] = []
    contract = result.contract
    pipeline = result.pipeline_result
    msg = result.message
    update = result.update

    if idx == 1:
        checks.extend(
            [
                ("contract WNZ->PIT", contract.trip.origin_airport == "WNZ" and contract.trip.destination_airport == "PIT"),
                ("asks departure window", "什么时候出发" in msg),
                ("no recommendations", pipeline is None),
                ("not ready", not contract.ready_to_search),
                ("no trace in normal mode", "需求决策" not in msg and "LLM: DeepSeek" not in msg),
            ]
        )
        name = "missing date asks clarification"
    elif idx == 2:
        checks.extend(
            [
                ("contract WNZ->PIT", contract.trip.origin_airport == "WNZ" and contract.trip.destination_airport == "PIT"),
                ("PVG/SHA acceptable", {"PVG", "SHA"}.issubset(set(contract.geography.acceptable_origin_hubs))),
                ("recommendations WNZ/PIT", _all_recs_match(pipeline, "WNZ", "PIT")),
                ("no generic value question", "你更看重" not in msg and "偏好问题" not in msg and "你的 value" not in msg),
                ("date captured", contract.time.departure_window_text),
            ]
        )
        name = "initial WNZ/PIT search"
    elif idx == 3:
        checks.extend(
            [
                ("create_new", update.update_type == "create_new"),
                ("trace detect_new_search", any(t.step == "detect_new_search" for t in update.decision_trace)),
                ("contract NGB->MIA", contract.trip.origin_airport == "NGB" and contract.trip.destination_airport == "MIA"),
                ("old route cleared", "WNZ" not in msg and "PIT" not in msg),
                ("recommendations NGB/MIA", _all_recs_match(pipeline, "NGB", "MIA")),
            ]
        )
        name = "new NGB/MIA search clears stale WNZ/PIT"
    elif idx == 4:
        downstream = _downstream_codes(pipeline)
        checks.extend(
            [
                ("PVG allowed", "PVG" in contract.geography.acceptable_origin_hubs),
                ("SHA excluded", "SHA" in contract.geography.avoid_airports),
                ("no SHA downstream", "SHA" not in downstream),
            ]
        )
        name = "PVG allowed and SHA excluded"
    elif idx == 5:
        checks.extend(
            [
                ("add_preference", update.update_type == "add_preference"),
                ("rerank only", update.should_rerank_only and result.rerank_only),
                ("no provider rerun", result.provider_call_count == 0),
                ("constraints preserved", "SHA" in contract.geography.avoid_airports),
                ("rerank status shown", "重新排序" in msg),
            ]
        )
        name = "airline priority rerank only"
    elif idx == 6:
        downstream = _downstream_codes(pipeline)
        blocked = {"JFK", "EWR", "LGA"}
        checks.extend(
            [
                ("NYC excluded", blocked.issubset(set(contract.geography.avoid_airports))),
                ("no NYC downstream", not (blocked & downstream)),
                ("suggestion not repeated", "不要纽约转" not in _suggestion_block(msg)),
            ]
        )
        name = "avoid New York transfer"
    elif idx == 7:
        checks.extend(
            [
                ("explain option", update.update_type == "explain_option"),
                ("detail route title", "方案" in msg and "路线明细" in msg),
                ("total price", "总价" in msg and "$" in msg),
                ("risk shown", "风险" in msg),
                ("data quality section", "数据说明" in msg),
                ("leg-by-leg detail", "接驳" in msg or "国际段" in msg or "国内段" in msg),
            ]
        )
        name = "detail view"
    else:
        export_dir = Path(result.export_dir or "")
        expected = [
            "transcript.txt",
            "updates.json",
            "contracts.json",
            "final_contract.json",
            "recommendations.json",
            "sft_turn_samples.jsonl",
            "sft_session_sample.json",
        ]
        checks.extend(
            [
                ("export update", update.update_type == "export"),
                ("export dir exists", export_dir.exists()),
                ("all export files", all((export_dir / name).exists() for name in expected)),
                ("export path shown", str(export_dir) in msg),
            ]
        )
        name = "export artifacts"

    # Common UI/UX checks for search/rerank turns
    if idx in {2, 3, 4, 5, 6}:
        checks.extend(
            [
                ("no LLM marker in normal mode", "LLM: DeepSeek" not in msg),
                ("no decision trace in normal mode", "需求决策" not in msg),
                ("recommendation fields shown", all(
                    h in msg for h in ["路线", "航司", "价格", "风险", "推荐理由"]
                )),
                ("no route IDs in normal mode", "itin-" not in msg and "hubsplit-" not in msg and "direct-" not in msg),
                ("mock warning present", "mock fallback" in msg.lower() or "估算" in msg),
                ("contract summary shown", "当前需求" in msg or "路线" in msg),
                ("no agent progress", "Agent 进度" not in msg and "候选枢纽组合" not in msg),
                ("follow-up prompt shown", "你可以继续追问" in msg),
            ]
        )

    checks.append(("response-only", _clean_response(msg)))
    checks.append(("user response model", result.user_response is not None))

    return {
        "step": idx,
        "name": name,
        "ok": all(ok for _, ok in checks),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }


def validate_v03_step(idx, result) -> dict:
    checks = [
        ("response-only", _clean_response(result.message)),
        ("user response model", result.user_response is not None),
        (
            "stream joins exactly",
            "".join(ResponseStreamer(chunk_size=12).stream_response(result.user_response)) == result.message,
        ),
    ]
    if idx == 9:
        checks.extend(
            [
                ("itinerary response", result.user_response.response_type == "itinerary"),
                ("three days", result.message.count("Day ") == 3),
                ("budget aware", result.contract.budget.preference == "lower"),
                ("weather source", "Open-Meteo" in result.message or "open_meteo" in result.message),
            ]
        )
        name = "v0.3 itinerary and streaming"
    elif idx == 10:
        checks.extend(
            [
                ("cost response", result.user_response.response_type == "cost_estimate"),
                ("rough estimates labeled", "rough estimate" in result.message),
                ("not a quote", "不是实时报价" in result.message),
            ]
        )
        name = "v0.3 cost estimate"
    elif idx == 11:
        checks.extend(
            [
                ("pet constraint recorded", any(pet.active for pet in result.contract.companions.pets)),
                ("policy reminder", "政策" in result.message),
            ]
        )
        name = "v0.3 generic pet constraint"
    else:
        checks.extend(
            [
                ("constraint response", result.user_response.response_type == "constraint_check"),
                ("official policy reminder", "官方" in result.message),
            ]
        )
        name = "v0.3 constraint check"
    return {
        "step": idx,
        "name": name,
        "ok": all(ok for _, ok in checks),
        "checks": [{"name": name, "ok": ok} for name, ok in checks],
    }


def _clean_response(message: str) -> bool:
    lowered = message.casefold()
    return all(marker.casefold() not in lowered for marker in FORBIDDEN)


def _acceptance_tool_router(settings) -> ToolRouter:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Austin",
                            "country": "United States",
                            "country_code": "US",
                            "admin1": "Texas",
                            "latitude": 30.2672,
                            "longitude": -97.7431,
                            "timezone": "America/Chicago",
                        }
                    ]
                },
            )
        if path.endswith("/v1/forecast"):
            return httpx.Response(
                200,
                json={
                    "current": {"temperature_2m": 30, "weather_code": 1},
                    "daily": {
                        "time": ["2026-07-06", "2026-07-07", "2026-07-08"],
                        "weather_code": [1, 2, 61],
                        "temperature_2m_max": [36, 32, 30],
                        "temperature_2m_min": [24, 23, 22],
                        "precipitation_probability_max": [10, 20, 60],
                    },
                },
            )
        if path.endswith("/w/api.php"):
            return httpx.Response(200, json={"query": {"search": [{"title": "奥斯汀"}]}})
        if "/api/rest_v1/page/summary/" in path:
            return httpx.Response(200, json={"extract": "奥斯汀目的地简介。"})
        if path.endswith("/v1/latest"):
            return httpx.Response(200, json={"date": "2026-07-03", "rates": {"CNY": 6.78}})
        return httpx.Response(404, json={})

    configured = replace(
        settings,
        enabled_tools=("weather", "airport_lookup", "time", "currency", "destination_brief"),
    )
    return ToolRouter(
        settings=configured,
        http_client=HttpClient(
            transport=httpx.MockTransport(handler),
            retries=0,
            backoff_seconds=0,
        ),
    )


def _all_recs_match(pipeline, origin: str, destination: str) -> bool:
    return bool(pipeline and pipeline.recommendations) and all(
        rec.itinerary.route[0] == origin and rec.itinerary.route[-1] == destination
        for rec in pipeline.recommendations
    )


def _downstream_codes(pipeline) -> set[str]:
    if not pipeline:
        return set()
    codes: set[str] = set()
    for pair in pipeline.hub_pairs:
        codes.update([pair.origin_airport, pair.origin_hub, pair.destination_hub, pair.destination_airport])
    for task in pipeline.search_tasks:
        codes.update([task.origin, task.destination])
    for offer in pipeline.offers:
        codes.update([offer.origin, offer.destination])
        for segment in offer.segments:
            codes.update([segment.origin, segment.destination])
    for rec in pipeline.recommendations:
        codes.update(rec.itinerary.route)
    return codes


def _suggestion_block(message: str) -> str:
    marker = "你可以继续追问"
    if marker not in message:
        return ""
    return message.split(marker, 1)[1]


if __name__ == "__main__":
    asyncio.run(main())
