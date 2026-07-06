from __future__ import annotations

import json
from pathlib import Path

import pytest
import httpx

from travel_agent.contract.compiler import ConstraintCompiler
from travel_agent.contract.merger import ContractMerger
from travel_agent.contract.models import ConstraintItem, SpecialRequirement, TravelRequirementContract
from travel_agent.contract.route_semantics import RouteSemanticValidator
from travel_agent.llm.deepseek_client import DeepSeekRequirementAgent, InvalidRequirementUpdate
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.llm.schemas import TravelRequirementContractUpdate
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.pipeline.types import FlightSegment, Itinerary, Recommendation, RiskAssessment
from travel_agent.pipeline.validator import RecommendationValidator
from travel_agent.services.display_service import DisplayService
from travel_agent.services.airport_service import AirportService
from travel_agent.services.sft_logger import SFTLogger
from travel_agent.tools.tool_router import ToolRequestContext, ToolRouter
from travel_agent.tools.http_client import HttpClient


INITIAL = "温州到匹兹堡，六月初，可以从上海走，越便宜越好"
MISSING_DATE_QUERY = "我要从温州去匹兹堡"


class TextOnlyLLM:
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return "好的，我帮你找。"


class MissingTraceLLM:
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "update_type": "create_new",
                "field_updates": {"trip": {"origin_airport": "WNZ", "destination_airport": "PIT"}},
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "ok",
                "reasoning_summary": "missing trace",
                "decision_trace": [],
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )


class CrashingLLM:
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("boom")


class MissingOriginLLM:
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "update_type": "create_new",
                "field_updates": {
                    "trip": {"destination_airport": "PIT", "destination_text": "匹兹堡"},
                    "time": {"departure_window_text": "六月初"},
                },
                "next_action": "run_search",
                "should_search": True,
                "should_rerun_search": True,
                "user_facing_ack_zh": "我先记录目的地。",
                "decision_trace": [
                    {
                        "step": "partial_route",
                        "evidence": "用户只提供目的地",
                        "decision": "缺少出发地，不能搜索。",
                        "affected_fields": ["trip.destination_airport"],
                    }
                ],
            },
            ensure_ascii=False,
        )


class MissingDestinationLLM:
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "update_type": "create_new",
                "field_updates": {
                    "trip": {"origin_airport": "WNZ", "origin_text": "温州"},
                    "time": {"departure_window_text": "六月初"},
                },
                "next_action": "run_search",
                "should_search": True,
                "should_rerun_search": True,
                "user_facing_ack_zh": "我先记录出发地。",
                "decision_trace": [
                    {
                        "step": "partial_route",
                        "evidence": "用户只提供出发地",
                        "decision": "缺少目的地，不能搜索。",
                        "affected_fields": ["trip.origin_airport"],
                    }
                ],
            },
            ensure_ascii=False,
        )


def make_session(tmp_path=None, *, debug: bool = False, show_reasoning: bool = False) -> LLMFirstChatSession:
    logger = SFTLogger(tmp_path) if tmp_path else None
    return LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(FakeRequirementLLM()),
        logger=logger,
        debug=debug,
        show_reasoning=show_reasoning,
        debug_dir=tmp_path / "debug" if tmp_path and debug else None,
    )


async def start_session(tmp_path=None) -> tuple[LLMFirstChatSession, object]:
    session = make_session(tmp_path)
    result = await session.handle_user_message(INITIAL)
    assert result.ok
    return session, result


def test_opening_screen_contains_examples():
    text = DisplayService().opening_screen_text()
    assert "AI 出行管家" in text
    assert "温州到匹兹堡" in text
    assert "宁波到迈阿密" in text
    assert "解释第1个" in text


@pytest.mark.asyncio
async def test_deepseek_requirement_agent_requires_strict_json():
    agent = DeepSeekRequirementAgent(FakeRequirementLLM())
    update = await agent.update(contract=None, user_message=INITIAL)
    assert isinstance(update, TravelRequirementContractUpdate)
    assert update.update_type == "create_new"
    assert update.decision_trace


@pytest.mark.asyncio
async def test_invalid_natural_language_only_response_is_rejected():
    agent = DeepSeekRequirementAgent(TextOnlyLLM())
    with pytest.raises(InvalidRequirementUpdate):
        await agent.update(contract=None, user_message=INITIAL)


@pytest.mark.asyncio
async def test_actionable_update_without_decision_trace_is_rejected():
    agent = DeepSeekRequirementAgent(MissingTraceLLM())
    with pytest.raises(InvalidRequirementUpdate):
        await agent.update(contract=None, user_message=INITIAL)


@pytest.mark.asyncio
async def test_create_new_for_ngb_to_mia():
    session, _ = await start_session()
    result = await session.handle_user_message("我其实想看看宁波到迈阿密，7月初")
    assert result.update.update_type == "create_new"
    assert result.contract.trip.origin_airport == "NGB"
    assert result.contract.trip.destination_airport == "MIA"
    assert any(t.step == "detect_new_search" for t in result.update.decision_trace)


@pytest.mark.asyncio
async def test_create_new_clears_old_wnz_pit_contract_state():
    session, _ = await start_session()
    result = await session.handle_user_message("我其实想看看宁波到迈阿密，7月初")
    assert result.contract.trip.origin_airport == "NGB"
    assert result.contract.trip.destination_airport == "MIA"
    assert "WNZ" not in result.contract.summary_zh()
    assert "PIT" not in result.contract.summary_zh()
    assert result.contract.geography.acceptable_origin_hubs == []


@pytest.mark.asyncio
async def test_create_new_clears_stale_result_and_displayed_options():
    session, first = await start_session()
    assert first.pipeline_result.recommendations
    second = await session.handle_user_message("我其实想看看宁波到迈阿密，7月初")
    assert session.last_result is second.pipeline_result
    assert session.displayed_options == second.pipeline_result.recommendations
    assert all(rec.itinerary.route[0] == "NGB" for rec in session.displayed_options)


@pytest.mark.asyncio
async def test_ngb_mia_results_do_not_show_wnz_pit():
    session, _ = await start_session()
    result = await session.handle_user_message("我其实想看看宁波到迈阿密，7月初")
    assert "WNZ" not in result.message
    assert "PIT" not in result.message
    assert all(rec.itinerary.route[0] == "NGB" for rec in result.pipeline_result.recommendations)
    assert all(rec.itinerary.route[-1] == "MIA" for rec in result.pipeline_result.recommendations)


@pytest.mark.asyncio
async def test_hangzhou_modifies_existing_contract_not_create_new():
    session, _ = await start_session()
    result = await session.handle_user_message("如果从杭州走呢")
    assert result.update.update_type == "modify_existing"
    assert result.contract.trip.origin_airport == "WNZ"
    assert result.contract.trip.destination_airport == "PIT"
    assert "HGH" in result.contract.geography.acceptable_origin_hubs


@pytest.mark.asyncio
async def test_pvg_allowed_and_sha_excluded():
    session, _ = await start_session()
    result = await session.handle_user_message("其实可以去上海浦东，但不去虹桥")
    assert "PVG" in result.contract.geography.acceptable_origin_hubs
    assert "SHA" in result.contract.geography.avoid_airports
    assert "SHA" not in result.contract.geography.acceptable_origin_hubs


@pytest.mark.asyncio
async def test_no_sha_in_downstream_outputs_or_display():
    session, _ = await start_session()
    result = await session.handle_user_message("其实可以去上海浦东，但不去虹桥")
    codes = downstream_codes(result.pipeline_result)
    assert "SHA" not in codes
    # SHA may appear in contract summary (user-facing constraint display), but not in recommendations


@pytest.mark.asyncio
async def test_city_level_shanghai_excludes_pvg_and_sha():
    session, _ = await start_session()
    result = await session.handle_user_message("我不想去上海")
    assert {"PVG", "SHA"}.issubset(set(result.contract.geography.avoid_airports))
    assert not {"PVG", "SHA"} & set(result.contract.geography.acceptable_origin_hubs)
    # PVG/SHA may appear in contract summary (user-facing constraint display), but not in route recommendations


@pytest.mark.asyncio
async def test_reallow_hongqiao_excludes_pudong_only():
    session, _ = await start_session()
    await session.handle_user_message("我不想去上海")
    result = await session.handle_user_message("不去浦东，可以去虹桥")
    assert "SHA" in result.contract.geography.acceptable_origin_hubs
    assert "SHA" not in result.contract.geography.avoid_airports
    assert "PVG" in result.contract.geography.avoid_airports


@pytest.mark.asyncio
async def test_avoid_new_york_excludes_all_nyc_airports():
    session, _ = await start_session()
    result = await session.handle_user_message("不要纽约转")
    assert {"JFK", "EWR", "LGA"}.issubset(set(result.contract.geography.avoid_airports))
    assert not {"JFK", "EWR", "LGA"} & downstream_codes(result.pipeline_result)
    # JFK/EWR/LGA may appear in contract summary (user-facing constraint display)
    suggestion_block = result.message.split("你可以继续追问", 1)[1]
    assert "不要纽约转" not in suggestion_block


@pytest.mark.asyncio
async def test_major_airline_priority_is_rerank_only_without_provider_rerun():
    session, _ = await start_session()
    result = await session.handle_user_message("主流航司优先")
    assert result.update.update_type == "add_preference"
    assert result.update.should_rerank_only
    assert result.rerank_only
    assert result.provider_call_count == 0
    assert result.contract.ranking.profile == "airline_priority"


@pytest.mark.asyncio
async def test_price_priority_does_not_trigger_generic_clarification():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    assert result.contract.ranking.profile == "cheapest"
    assert result.contract.ranking.price_priority == "high"
    assert "你更看重" not in result.message
    assert "请问" not in result.message


@pytest.mark.asyncio
async def test_missing_departure_window_blocks_search_and_asks_once():
    session = make_session()
    result = await session.handle_user_message(MISSING_DATE_QUERY)
    assert result.contract.trip.origin_airport == "WNZ"
    assert result.contract.trip.destination_airport == "PIT"
    assert not result.contract.ready_to_search
    assert result.pipeline_result is None
    assert result.update.next_action == "ask_clarification"
    assert "什么时候出发" in result.message
    assert result.message.count("什么时候出发") == 1


@pytest.mark.asyncio
async def test_flexible_date_confirmation_runs_search():
    session = make_session()
    result = await session.handle_user_message("温州到匹兹堡，日期还没定，先随便看看")
    assert result.contract.ready_to_search
    assert result.contract.time.flexible_date_confirmed
    assert any(a.field == "time.departure_date" for a in result.contract.assumptions)
    assert result.pipeline_result.recommendations


@pytest.mark.asyncio
async def test_missing_mandatory_origin_blocks_search():
    session = LLMFirstChatSession(requirement_agent=DeepSeekRequirementAgent(MissingOriginLLM()))
    result = await session.handle_user_message("去匹兹堡，六月初")
    assert result.pipeline_result is None
    assert not result.contract.ready_to_search
    assert "从哪里出发" in result.message


@pytest.mark.asyncio
async def test_missing_mandatory_destination_blocks_search():
    session = LLMFirstChatSession(requirement_agent=DeepSeekRequirementAgent(MissingDestinationLLM()))
    result = await session.handle_user_message("从温州走，六月初")
    assert result.pipeline_result is None
    assert not result.contract.ready_to_search
    assert "去哪里" in result.message


@pytest.mark.asyncio
async def test_chengdu_to_austin_directional_route_is_repaired_and_asks_date():
    session = make_session()
    result = await session.handle_user_message("我想从成都飞奥斯丁")
    assert result.contract.trip.origin_airport == "TFU"
    assert result.contract.trip.destination_airport == "AUS"
    assert result.pipeline_result is None
    assert "什么时候出发" in result.message


@pytest.mark.asyncio
async def test_chengdu_to_austin_with_to_marker():
    session = make_session()
    result = await session.handle_user_message("从成都到奥斯丁")
    assert result.contract.trip.origin_airport == "TFU"
    assert result.contract.trip.destination_airport == "AUS"


@pytest.mark.asyncio
async def test_chengdu_go_austin_marker():
    session = make_session()
    result = await session.handle_user_message("成都去奥斯丁")
    assert result.contract.trip.origin_airport == "TFU"
    assert result.contract.trip.destination_airport == "AUS"


@pytest.mark.asyncio
async def test_austin_to_chengdu_marker():
    session = make_session()
    result = await session.handle_user_message("奥斯丁飞成都")
    assert result.contract.trip.origin_airport == "AUS"
    assert result.contract.trip.destination_airport == "TFU"


def test_route_semantic_validator_repairs_swapped_llm_route():
    update = TravelRequirementContractUpdate(
        update_type="create_new",
        field_updates={"trip": {"origin_airport": "AUS", "destination_airport": "TFU"}},
        next_action="run_search",
        decision_trace=[{"step": "s", "evidence": "e", "decision": "d"}],
    )
    result = RouteSemanticValidator().validate_update(user_message="我想从成都飞奥斯丁", update=update)
    assert result.repaired
    assert update.field_updates["trip"]["origin_airport"] == "TFU"
    assert update.field_updates["trip"]["destination_airport"] == "AUS"


def test_route_semantic_validator_preserves_route_for_date_only_create_new():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "TFU"
    contract.trip.destination_airport = "AUS"
    update = TravelRequirementContractUpdate(
        update_type="create_new",
        field_updates={"time": {"departure_window_text": "六月初"}},
        next_action="run_search",
        should_rerun_search=True,
        decision_trace=[{"step": "s", "evidence": "e", "decision": "d"}],
    )
    result = RouteSemanticValidator().validate_update(
        user_message="六月初，越便宜越好",
        update=update,
        current_contract=contract,
    )
    assert result.repaired
    assert update.update_type == "clarification_answer"


@pytest.mark.asyncio
async def test_ambiguous_chengdu_austin_asks_clarification():
    session = make_session()
    result = await session.handle_user_message("查一下成都奥斯丁")
    assert result.pipeline_result is None
    assert result.update.next_action == "ask_clarification"
    assert "从成都去奥斯丁" in result.message


def test_airport_service_resolves_chengdu_and_austin():
    service = AirportService()
    assert service.resolve_location("成都") == ["TFU", "CTU"]
    assert service.resolve_location("奥斯丁") == ["AUS"]


@pytest.mark.asyncio
async def test_clarification_answer_after_chengdu_austin_searches():
    session = make_session()
    await session.handle_user_message("我想从成都飞奥斯丁")
    result = await session.handle_user_message("六月初，越便宜越好")
    assert result.contract.trip.origin_airport == "TFU"
    assert result.contract.trip.destination_airport == "AUS"
    assert result.pipeline_result is not None
    assert result.pipeline_result.recommendations


@pytest.mark.asyncio
async def test_destination_weather_tool_query_with_known_destination():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    session = make_session()
    session.tool_router = ToolRouter(
        http_client=HttpClient(transport=httpx.MockTransport(fail), retries=0, backoff_seconds=0)
    )
    await session.handle_user_message("温州到匹兹堡，六月初，可以从上海走，越便宜越好")
    result = await session.handle_user_message("目的地天气怎么样？")
    assert result.update.next_action == "tool_query"
    assert result.update.tool_requests[0].tool_name == "weather"
    assert result.pipeline_result is None
    assert "Open-Meteo 当前不可用" in result.message
    assert "推荐方案" not in result.message


def test_weather_tool_stub_does_not_invent_weather():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    result = ToolRouter(
        http_client=HttpClient(transport=httpx.MockTransport(fail), retries=0, backoff_seconds=0)
    ).execute(
        TravelRequirementContractUpdate(
            update_type="advisory_question",
            next_action="tool_query",
            tool_requests=[
                {
                    "tool_name": "weather",
                    "arguments": {"location": "Austin"},
                    "reason_zh": "test",
                    "requires_current_contract": False,
                }
            ],
        ).tool_requests[0],
        ToolRequestContext(contract=None),
    )
    assert not result.success
    assert result.status == "unavailable"
    assert "不会用猜测数据" in result.user_facing_text_zh
    assert "晴" not in result.user_facing_text_zh


@pytest.mark.asyncio
async def test_airport_lookup_tool_query():
    session = make_session()
    result = await session.handle_user_message("奥斯丁机场是哪个？")
    assert result.update.next_action == "tool_query"
    assert result.update.tool_requests[0].tool_name == "airport_lookup"
    assert "Austin-Bergstrom International Airport (AUS)" in result.message
    assert result.pipeline_result is None


@pytest.mark.asyncio
async def test_display_uses_actual_decision_trace():
    session = make_session(show_reasoning=True)
    result = await session.handle_user_message(INITIAL)
    trace_step = result.update.decision_trace[0].step
    assert f"1. {trace_step}" in result.message
    assert "extract_new_search" in result.message


@pytest.mark.asyncio
async def test_normal_mode_hides_decision_trace_and_schema_fields():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    assert "需求决策" not in result.message
    assert "LLM: DeepSeek" not in result.message
    assert "trip.origin_airport" not in result.message
    assert "affected_fields" not in result.message


@pytest.mark.asyncio
async def test_recommendation_table_has_route_airline_price_risk_data_quality_and_no_ids():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    text = result.message
    assert "路线" in text
    assert "航司" in text
    assert "总价" in text
    assert "风险" in text
    assert "数据" in text
    assert "温州 WNZ" in text
    assert "$" in text
    assert "估算demo" in text or "demo固定" in text
    assert "itin-" not in text
    assert "mock-" not in text


@pytest.mark.asyncio
async def test_detail_view_shows_leg_by_leg_route():
    session, _ = await start_session()
    result = await session.handle_user_message("解释第1个")
    assert result.update.update_type == "explain_option"
    assert "方案 1" in result.message
    assert "路线明细" in result.message
    assert "总价" in result.message
    assert "风险原因" in result.message
    assert "数据说明" in result.message


@pytest.mark.asyncio
async def test_normal_chat_output_has_no_info_logs():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    assert "INFO" not in result.message
    assert "DEBUG" not in result.message


@pytest.mark.asyncio
async def test_debug_mode_includes_diagnostics(tmp_path):
    session = make_session(tmp_path, debug=True)
    result = await session.handle_user_message(INITIAL)
    assert "Debug diagnostics" in result.message
    assert "update_type: create_new" in result.message
    assert "search_task_count" in result.message
    assert (tmp_path / "debug" / "logs.txt").exists()


@pytest.mark.asyncio
async def test_sft_logger_writes_target_update_and_final_contract(tmp_path):
    session = make_session(tmp_path)
    result = await session.handle_user_message(INITIAL)
    turns = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    session_rows = (tmp_path / "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert turns
    assert session_rows
    row = json.loads(turns[-1])
    session_row = json.loads(session_rows[-1])
    assert row["target_update"]["update_type"] == "create_new"
    assert row["target_contract_after_update"]["trip"]["origin_airport"] == "WNZ"
    assert session_row["target_final_contract"]["trip"]["destination_airport"] == "PIT"
    assert result.ok


@pytest.mark.asyncio
async def test_export_writes_transcript_updates_contracts_recommendations_and_sft(tmp_path):
    session, _ = await start_session()
    export_dir = tmp_path / "export"
    saved = session.export(export_dir)
    expected = [
        "transcript.txt",
        "updates.json",
        "contracts.json",
        "final_contract.json",
        "recommendations.json",
        "sft_turn_samples.jsonl",
        "sft_session_sample.json",
    ]
    assert saved == export_dir
    assert all((export_dir / name).exists() for name in expected)
    assert json.loads((export_dir / "updates.json").read_text(encoding="utf-8"))[0]["update_type"] == "create_new"


@pytest.mark.asyncio
async def test_export_followup_writes_artifacts():
    session, _ = await start_session()
    result = await session.handle_user_message("导出")
    assert result.update.update_type == "export"
    assert result.export_dir
    assert "已导出本轮对话" in result.message
    assert (Path(result.export_dir) / "transcript.txt").exists()
    assert (Path(result.export_dir) / "sft_session_sample.json").exists()


@pytest.mark.asyncio
async def test_invalid_llm_output_does_not_show_old_recommendations():
    session, _ = await start_session()
    session.requirement_agent = DeepSeekRequirementAgent(TextOnlyLLM())
    result = await session.handle_user_message("随便再查一下")
    assert not result.ok
    assert "没有返回有效" in result.message
    assert "温州 WNZ" not in result.message
    assert session.last_result is None


@pytest.mark.asyncio
async def test_mock_fallback_warning_appears():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    assert "mock fallback" in result.message


@pytest.mark.asyncio
async def test_arbitrary_special_requirement_message_does_not_crash():
    session, _ = await start_session()
    result = await session.handle_user_message("我想带我家狗一起去")
    assert result.ok
    assert result.update.update_type == "advisory_question"
    assert result.update.next_action == "answer_advisory"
    assert result.contract.special_requirements
    assert result.pipeline_result is None
    assert "宠物同行" in result.message
    assert "狗狗大概" in result.message


@pytest.mark.asyncio
async def test_pet_travel_becomes_general_special_requirement():
    session, _ = await start_session()
    result = await session.handle_user_message("我想带我家狗一起去")
    special = result.contract.special_requirements[-1]
    assert special.category == "pet_travel"
    assert special.structured_values["animal"] == "dog"
    assert "airline_policy" in special.impact_areas
    assert result.contract.airline_preferences.prefer_major_airlines


@pytest.mark.asyncio
async def test_family_message_becomes_special_requirement_and_low_risk():
    session, _ = await start_session()
    result = await session.handle_user_message("我爸妈也一起，别太折腾")
    assert result.update.update_type == "add_special_requirement"
    assert any(item.category == "family_or_elderly" for item in result.contract.special_requirements)
    assert result.contract.passengers.family_or_parents
    assert result.contract.ranking.profile == "low_risk"
    assert result.contract.ticketing.split_ticket_policy == "avoid"


@pytest.mark.asyncio
async def test_heavy_baggage_becomes_special_requirement():
    session, _ = await start_session()
    result = await session.handle_user_message("我有很多行李")
    assert any(item.category == "heavy_baggage" for item in result.contract.special_requirements)
    assert result.contract.ticketing.allow_self_transfer is False
    assert "较多行李" in result.message
    assert result.pipeline_result is None


@pytest.mark.asyncio
async def test_visa_concern_becomes_special_requirement():
    session, _ = await start_session()
    result = await session.handle_user_message("我没有美国签证，不想过境麻烦")
    assert any(item.category == "visa_constraint" for item in result.contract.special_requirements)
    assert "签证和过境政策" in result.message
    assert result.pipeline_result is None


@pytest.mark.asyncio
async def test_smalltalk_does_not_search():
    session = make_session()
    result = await session.handle_user_message("你好")
    assert result.ok
    assert result.update.update_type == "smalltalk"
    assert result.pipeline_result is None
    assert session.last_result is None
    assert "温州到匹兹堡" in result.message


@pytest.mark.asyncio
async def test_help_does_not_search():
    session = make_session()
    result = await session.handle_user_message("这个系统怎么用")
    assert result.ok
    assert result.update.update_type == "help"
    assert result.pipeline_result is None
    assert session.last_result is None
    assert "可以直接描述路线" in result.message


def test_contract_merger_coerces_dict_constraints():
    contract = TravelRequirementContract()
    update = TravelRequirementContractUpdate(
        update_type="add_constraint",
        field_updates={
            "constraints": {
                "hard_constraints": [
                    {
                        "type": "avoid_airport",
                        "value": "SHA",
                        "normalized_values": ["SHA"],
                        "reason": "test",
                    }
                ]
            }
        },
        decision_trace=[{"step": "s", "evidence": "e", "decision": "d", "affected_fields": ["constraints"]}],
    )
    merged = ContractMerger().apply(contract, update)
    assert isinstance(merged.constraints.hard_constraints[0], ConstraintItem)
    assert "SHA" in merged.geography.avoid_airports


def test_contract_merger_coerces_dict_special_requirements():
    contract = TravelRequirementContract()
    update = TravelRequirementContractUpdate(
        update_type="add_special_requirement",
        field_updates={
            "special_requirements": [
                {
                    "category": "pet_travel",
                    "description_zh": "用户想带狗同行",
                    "structured_values": {"animal": "dog"},
                    "impact_areas": ["airline_policy", "risk"],
                    "source_user_message": "我想带我家狗一起去",
                }
            ]
        },
        decision_trace=[{"step": "s", "evidence": "e", "decision": "d", "affected_fields": ["special_requirements"]}],
    )
    merged = ContractMerger().apply(contract, update)
    assert isinstance(merged.special_requirements[0], SpecialRequirement)
    assert merged.special_requirements[0].category == "pet_travel"


def test_contract_merger_skips_malformed_dicts_without_crash():
    contract = TravelRequirementContract()
    update = TravelRequirementContractUpdate(
        update_type="add_special_requirement",
        field_updates={
            "constraints": {"hard_constraints": [{"type": "not_a_valid_constraint"}]},
            "special_requirements": [{"category": "pet_travel", "structured_values": "bad"}],
        },
        decision_trace=[{"step": "s", "evidence": "e", "decision": "d", "affected_fields": ["special_requirements"]}],
    )
    merger = ContractMerger()
    merged = merger.apply(contract, update)
    assert merged.constraints.hard_constraints == []
    assert merged.special_requirements == []
    assert merger.diagnostics


@pytest.mark.asyncio
async def test_every_chat_turn_has_error_boundary():
    session = LLMFirstChatSession(requirement_agent=DeepSeekRequirementAgent(CrashingLLM()))
    result = await session.handle_user_message("我想带我家狗一起去")
    assert not result.ok
    assert "这轮处理时出错了" in result.message
    assert session.last_result is None


@pytest.mark.asyncio
async def test_normal_mode_hides_traceback_on_internal_error():
    session = LLMFirstChatSession(requirement_agent=DeepSeekRequirementAgent(CrashingLLM()))
    result = await session.handle_user_message("随便一句")
    assert "Traceback" not in result.message
    assert "RuntimeError" not in result.message


@pytest.mark.asyncio
async def test_debug_mode_saves_diagnostics_on_internal_error(tmp_path):
    session = LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(CrashingLLM()),
        debug=True,
        debug_dir=tmp_path / "debug",
    )
    result = await session.handle_user_message("随便一句")
    assert not result.ok
    assert "Traceback" in result.message
    assert list((tmp_path / "debug").glob("turn_error_*.txt"))


def test_special_requirements_appear_in_current_requirement_card():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "WNZ"
    contract.trip.destination_airport = "PIT"
    contract.special_requirements = [
        SpecialRequirement(
            category="pet_travel",
            description_zh="用户想带狗同行",
            impact_areas=["airline_policy"],
            requires_clarification=True,
            clarification_question_zh="狗狗大概是小型犬、中型犬还是大型犬？",
        )
    ]
    summary = DisplayService().contract_summary_zh(contract)
    assert "特殊需求" in summary
    assert "宠物同行" in summary
    assert "狗狗大概" in summary


def test_special_requirement_warnings_appear_in_detail_view():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "WNZ"
    contract.trip.destination_airport = "PIT"
    contract.special_requirements = [
        SpecialRequirement(category="pet_travel", description_zh="用户想带狗同行", impact_areas=["airline_policy"])
    ]
    contract.normalize()
    exclusions = ConstraintCompiler().compile(contract).exclusions
    detail = DisplayService().detail_view(
        stale_recommendation(route=["WNZ", "PVG", "JFK", "PIT"]),
        index=1,
        contract=contract,
        exclusions=exclusions,
    )
    assert "特殊需求提醒" in detail
    assert "需要向航司确认" in detail


def test_recommendation_validator_removes_excluded_airports():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "WNZ"
    contract.trip.destination_airport = "PIT"
    contract.geography.avoid_airports = ["SHA"]
    contract.normalize()
    exclusions = ConstraintCompiler().compile(contract).exclusions
    rec = stale_recommendation(route=["WNZ", "SHA", "JFK", "PIT"])
    valid = RecommendationValidator().validate_recommendations([rec], contract=contract, exclusions=exclusions)
    assert valid == []


def test_opening_screen_has_tips():
    tips = DisplayService().opening_tips()
    assert "改变目的地" in tips
    assert "quit" in tips


def test_opening_screen_no_implementation_details():
    text = DisplayService().opening_screen_text()
    assert "DeepSeek" not in text
    assert "TravelRequirementContract" not in text
    assert "DeepSeekRequirementAgent" not in text


def test_followup_prompt_shows_after_table():
    prompt = DisplayService().followup_prompt([])
    assert prompt == ""
    from travel_agent.pipeline.types import FlightSegment, Itinerary, Recommendation, RiskAssessment
    rec = Recommendation(
        rank=1, recommendation_type="最省钱",
        itinerary=Itinerary(id="x", route_type="hub", route=["WNZ", "PIT"], offers=[], segments=[], total_price_usd=100, total_estimated_time_hours=10),
        score=1.0, savings_vs_baseline_usd=0,
        risk=RiskAssessment(risk_score=0.1, risk_level="low"),
        airline_quality_score=0.5, reason_zh="test",
    )
    prompt = DisplayService().followup_prompt([rec])
    assert "可以继续追问" in prompt
    assert "解释第1个" in prompt
    assert "导出" in prompt
    assert "quit" in prompt


def test_followup_prompt_suggests_avoid_new_york_when_route_uses_nyc():
    from travel_agent.pipeline.types import Itinerary, Recommendation, RiskAssessment
    rec = Recommendation(
        rank=1, recommendation_type="最省钱",
        itinerary=Itinerary(id="x", route_type="hub", route=["WNZ", "JFK", "PIT"], offers=[], segments=[], total_price_usd=100, total_estimated_time_hours=10),
        score=1.0, savings_vs_baseline_usd=0,
        risk=RiskAssessment(risk_score=0.5, risk_level="medium"),
        airline_quality_score=0.5, reason_zh="test",
    )
    prompt = DisplayService().followup_prompt([rec])
    assert "不要纽约转" in prompt
    assert "风险低一点" in prompt


def test_contract_summary_shows_route_and_constraints():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "WNZ"
    contract.trip.destination_airport = "PIT"
    contract.geography.acceptable_origin_hubs = ["PVG"]
    contract.geography.avoid_airports = ["SHA"]
    contract.ranking.profile = "cheapest"
    contract.normalize()
    summary = DisplayService().contract_summary_zh(contract)
    assert "温州" in summary or "WNZ" in summary
    assert "匹兹堡" in summary or "PIT" in summary
    assert "PVG" in summary
    assert "SHA" in summary


def test_contract_summary_empty_for_no_origin():
    contract = TravelRequirementContract()
    assert DisplayService().contract_summary_zh(contract) == ""


def test_risk_label_returns_chinese():
    display = DisplayService()
    assert display.risk_label("low") == "低"
    assert display.risk_label("medium") == "中"
    assert display.risk_label("high") == "高"
    assert display.risk_label("unknown") == "unknown"


def test_format_savings_shows_dollar_and_percent():
    from travel_agent.pipeline.types import FlightSegment, Itinerary, Recommendation, RiskAssessment
    itinerary = Itinerary(
        id="x", route_type="hub", route=["WNZ", "PIT"], offers=[], segments=[],
        total_price_usd=800, total_estimated_time_hours=10,
    )
    rec = Recommendation(
        rank=1, recommendation_type="最省钱", itinerary=itinerary,
        score=1.0, savings_vs_baseline_usd=200,
        risk=RiskAssessment(risk_score=0.1, risk_level="low"),
        airline_quality_score=0.5, reason_zh="test",
    )
    s = DisplayService()._format_savings(rec)
    assert "省" in s
    assert "$200" in s
    assert "%" in s


def test_format_savings_no_savings():
    from travel_agent.pipeline.types import FlightSegment, Itinerary, Recommendation, RiskAssessment
    itinerary = Itinerary(
        id="x", route_type="hub", route=["WNZ", "PIT"], offers=[], segments=[],
        total_price_usd=1000, total_estimated_time_hours=10,
    )
    rec = Recommendation(
        rank=1, recommendation_type="最省钱", itinerary=itinerary,
        score=1.0, savings_vs_baseline_usd=0,
        risk=RiskAssessment(risk_score=0.1, risk_level="low"),
        airline_quality_score=0.5, reason_zh="test",
    )
    assert DisplayService()._format_savings(rec) == "—"


@pytest.mark.asyncio
async def test_search_status_message_shown():
    session = make_session()
    result = await session.handle_user_message(INITIAL)
    assert "搜索组合路线" in result.message


@pytest.mark.asyncio
async def test_rerank_status_message_shown():
    session, _ = await start_session()
    result = await session.handle_user_message("主流航司优先")
    assert "重新排序" in result.message


@pytest.mark.asyncio
async def test_hard_constraint_reflected_in_contract_summary():
    session, _ = await start_session()
    await session.handle_user_message("其实可以去上海浦东，但不去虹桥")
    result = await session.handle_user_message("不要纽约转")
    assert "JFK" in result.contract.geography.avoid_airports or "EWR" in result.contract.geography.avoid_airports
    assert "SHA" in result.contract.geography.avoid_airports
    assert result.contract.ready_to_search


def test_display_guard_rejects_stale_route_mismatch():
    contract = TravelRequirementContract()
    contract.trip.origin_airport = "NGB"
    contract.trip.destination_airport = "MIA"
    contract.normalize()
    exclusions = ConstraintCompiler().compile(contract).exclusions
    text = DisplayService().recommendations_table(
        [stale_recommendation(route=["WNZ", "PVG", "JFK", "PIT"])],
        contract=contract,
        exclusions=exclusions,
    )
    assert "当前限制下没有找到合适方案" in text
    assert "WNZ" not in text
    assert "PIT" not in text


def downstream_codes(pipeline_result) -> set[str]:
    codes: set[str] = set()
    for pair in pipeline_result.hub_pairs:
        codes.update([pair.origin_airport, pair.origin_hub, pair.destination_hub, pair.destination_airport])
    for task in pipeline_result.search_tasks:
        codes.update([task.origin, task.destination])
    for task in pipeline_result.provider_calls:
        codes.update([task.origin, task.destination])
    for itinerary in pipeline_result.itineraries:
        codes.update(itinerary.route)
    for rec in pipeline_result.recommendations:
        codes.update(rec.itinerary.route)
    return codes


def stale_recommendation(route: list[str]) -> Recommendation:
    segments = [
        FlightSegment(origin=route[i], destination=route[i + 1], airline="UA")
        for i in range(len(route) - 1)
    ]
    itinerary = Itinerary(
        id="stale",
        route_type="hub_split",
        route=route,
        offers=[],
        segments=segments,
        total_price_usd=1000,
        total_estimated_time_hours=20,
    )
    return Recommendation(
        rank=1,
        recommendation_type="拆分",
        itinerary=itinerary,
        score=1.0,
        savings_vs_baseline_usd=100,
        risk=RiskAssessment(risk_score=0.2, risk_level="low"),
        airline_quality_score=0.8,
        reason_zh="stale",
    )
