"""Deterministic pipeline orchestration and LLM-first chat session."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from travel_agent.contract.compiler import ConstraintCompiler
from travel_agent.contract.completeness import RequirementCompletenessChecker
from travel_agent.contract.merger import ContractMerger
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.deepseek_client import DeepSeekRequirementAgent, InvalidRequirementUpdate
from travel_agent.pipeline.hubsplit_planner import HubSplitPlanner
from travel_agent.pipeline.mock_provider import MockFlightProvider
from travel_agent.pipeline.ranking_service import RankingService
from travel_agent.pipeline.route_composer import RouteComposer
from travel_agent.pipeline.search_task_planner import SearchTaskPlanner
from travel_agent.pipeline.types import ChatTurnResult, PipelineResult
from travel_agent.pipeline.validator import RecommendationValidator
from travel_agent.services.display_service import DisplayService
from travel_agent.services.sft_logger import SFTLogger
from travel_agent.tools.tool_router import ToolRequestContext, ToolRouter


INVALID_UPDATE_MESSAGE = "\n".join(
    [
        "DeepSeek 没有返回有效的可执行需求更新。",
        "我没有继续使用旧结果，以避免展示错误路线。",
        "你可以换一种说法，例如：“换成宁波到迈阿密”。",
    ]
)

INTERNAL_ERROR_MESSAGE = "\n".join(
    [
        "这轮处理时出错了。我没有继续使用可能过期的结果。",
        "你可以换一种说法，或输入 help。",
    ]
)


class SearchPipelineOrchestrator:
    def __init__(
        self,
        *,
        compiler: ConstraintCompiler | None = None,
        completeness: RequirementCompletenessChecker | None = None,
        hub_planner: HubSplitPlanner | None = None,
        task_planner: SearchTaskPlanner | None = None,
        provider: MockFlightProvider | None = None,
        composer: RouteComposer | None = None,
        ranker: RankingService | None = None,
        validator: RecommendationValidator | None = None,
    ):
        self.compiler = compiler or ConstraintCompiler()
        self.completeness = completeness or RequirementCompletenessChecker()
        self.hub_planner = hub_planner or HubSplitPlanner()
        self.task_planner = task_planner or SearchTaskPlanner()
        self.provider = provider or MockFlightProvider()
        self.composer = composer or RouteComposer()
        self.ranker = ranker or RankingService()
        self.validator = validator or RecommendationValidator()

    def run(self, contract: TravelRequirementContract) -> PipelineResult:
        completeness = self.completeness.check(contract)
        compiled = self.compiler.compile(contract)
        if not completeness.ready_to_search:
            return PipelineResult(contract=contract, exclusions=compiled.exclusions, warnings=completeness.clarification_questions)

        self.provider.reset_calls()
        hub_pairs = self.hub_planner.plan(compiled)
        tasks = self.task_planner.plan(constraints=compiled, hub_pairs=hub_pairs)
        tasks = [
            task
            for task in tasks
            if not compiled.exclusions.airport_is_excluded(task.origin)
            and not compiled.exclusions.airport_is_excluded(task.destination)
        ]
        offers = []
        for task in tasks:
            offers.extend(self.provider.search(task, compiled.exclusions))
        itineraries = self.composer.compose(
            hub_pairs=hub_pairs,
            tasks=tasks,
            offers=offers,
            exclusions=compiled.exclusions,
        )
        itineraries = [
            itinerary
            for itinerary in itineraries
            if self.validator.validate_itinerary(itinerary, contract=contract, exclusions=compiled.exclusions)
        ]
        recommendations = self.ranker.rank(itineraries, contract)
        recommendations = self.validator.validate_recommendations(
            recommendations,
            contract=contract,
            exclusions=compiled.exclusions,
        )
        return PipelineResult(
            contract=contract,
            exclusions=compiled.exclusions,
            hub_pairs=hub_pairs,
            search_tasks=tasks,
            provider_calls=list(self.provider.calls),
            offers=offers,
            itineraries=itineraries,
            recommendations=recommendations,
            warnings=[] if recommendations else [self.validator.relaxation_suggestion()],
            provider_call_count=len(self.provider.calls),
        )

    def rerank(
        self,
        *,
        previous_result: PipelineResult,
        contract: TravelRequirementContract,
    ) -> PipelineResult:
        compiled = self.compiler.compile(contract)
        recommendations = self.ranker.rank(previous_result.itineraries, contract)
        recommendations = self.validator.validate_recommendations(
            recommendations,
            contract=contract,
            exclusions=compiled.exclusions,
        )
        return PipelineResult(
            contract=contract,
            exclusions=compiled.exclusions,
            hub_pairs=previous_result.hub_pairs,
            search_tasks=previous_result.search_tasks,
            provider_calls=[],
            offers=previous_result.offers,
            itineraries=[
                it
                for it in previous_result.itineraries
                if self.validator.validate_itinerary(it, contract=contract, exclusions=compiled.exclusions)
            ],
            recommendations=recommendations,
            warnings=[] if recommendations else [self.validator.relaxation_suggestion()],
            provider_call_count=0,
        )


class LLMFirstChatSession:
    """Stateful chat controller: every turn begins with DeepSeekRequirementAgent."""

    def __init__(
        self,
        *,
        requirement_agent: DeepSeekRequirementAgent,
        pipeline: SearchPipelineOrchestrator | None = None,
        merger: ContractMerger | None = None,
        display: DisplayService | None = None,
        logger: SFTLogger | None = None,
        tool_router: ToolRouter | None = None,
        debug: bool = False,
        show_reasoning: bool = False,
        debug_dir: Path | None = None,
    ):
        self.requirement_agent = requirement_agent
        self.pipeline = pipeline or SearchPipelineOrchestrator()
        self.merger = merger or ContractMerger()
        self.display = display or DisplayService()
        self.logger = logger
        self.tool_router = tool_router or ToolRouter()
        self.debug = debug
        self.show_reasoning = show_reasoning or debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug and self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "logs.txt").write_text("", encoding="utf-8")
        self.contract: TravelRequirementContract | None = None
        self.last_result: PipelineResult | None = None
        self.displayed_options = []
        self.history_summary = ""
        self.messages: list[dict[str, str]] = []
        self.updates: list[dict] = []
        self.contract_snapshots: list[dict] = []
        self.turn_samples: list[dict] = []
        self.last_export_dir: Path | None = None

    async def handle_user_message(self, message: str) -> ChatTurnResult:
        try:
            return await self._handle_user_message_inner(message)
        except Exception as exc:
            return self._handle_internal_error(exc, message)

    async def _handle_user_message_inner(self, message: str) -> ChatTurnResult:
        previous_contract = self.contract.model_dump(mode="json") if self.contract else None
        displayed_summary = (
            self.display.recommendations_summary(self.last_result.recommendations)
            if self.last_result
            else ""
        )
        self.messages.append({"role": "user", "content": message})
        if self.logger:
            self.logger.add_message("user", message)
        try:
            update = await self.requirement_agent.update(
                contract=self.contract,
                user_message=message,
                history_summary=self.history_summary,
                displayed_recommendations_summary=displayed_summary,
            )
        except InvalidRequirementUpdate:
            self.last_result = None
            self.displayed_options = []
            self.messages.append({"role": "assistant", "content": INVALID_UPDATE_MESSAGE})
            return ChatTurnResult(ok=False, message=INVALID_UPDATE_MESSAGE)

        self.updates.append(update.model_dump(mode="json"))
        self._save_debug_update(update)

        action = self._next_action(update)

        if action == "explain_result":
            output = self._base_output(update)
            output.append(self.display.action_summary(update.update_type))
            output.append(self._detail_for_update(update))
            if self.last_result:
                output.append(
                    self.display.followup_prompt(
                        self.last_result.recommendations,
                        contract=self.contract,
                        exclusions=self.last_result.exclusions,
                    )
                )
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        if action == "export":
            export_dir = self.export()
            output = self._base_output(update)
            output.append(self.display.action_summary(update.update_type))
            output.append(self.display.export_success_panel(export_dir))
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(
                ok=True,
                message=message_text,
                update=update,
                contract=self.contract,
                pipeline_result=self.last_result,
                export_dir=str(export_dir),
            )

        if action in {"quit", "help", "smalltalk"} or update.update_type in {"quit", "help", "smalltalk"}:
            lines = [update.user_facing_ack_zh or ""]
            if action == "help" or update.update_type == "help":
                lines.append("可以说：温州到匹兹堡，六月初，可以从上海走，越便宜越好；也可以追加：不要纽约转、主流航司优先。")
            message_text = "\n".join(line for line in lines if line)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)
        if action == "no_op" or update.update_type == "unknown":
            message_text = update.user_facing_ack_zh or "这句话我还不能确定如何转成出行需求。你是想修改路线、排除机场，还是调整偏好？"
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        next_contract = self.merger.apply(self.contract, update, user_message=message)
        if update.update_type == "create_new":
            self.last_result = None
            self.displayed_options = []
        self.contract = next_contract

        if action == "tool_query":
            output = self._base_output(update)
            output.extend(self._tool_query_output(update))
            self._log_turn(previous_contract, update, None, {"ready": self.contract.ready_to_search, "action": action})
            self.history_summary = self.contract.summary_zh()
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        completeness = self.pipeline.completeness.check(self.contract)
        if action == "answer_advisory":
            output = self._base_output(update)
            if update.advisory_response_zh:
                output.append(update.advisory_response_zh)
            elif update.clarification_question_zh:
                output.append(update.clarification_question_zh)
            if self.show_reasoning:
                special_warning = self.display.special_requirement_warning_panel(self.contract)
                if special_warning:
                    output.append(special_warning)
            if not update.advisory_response_zh and update.clarification_question_zh and update.clarification_question_zh not in "\n".join(output):
                output.append(update.clarification_question_zh)
            self._log_turn(previous_contract, update, None, {"ready": completeness.ready_to_search, "action": action})
            self.history_summary = self.contract.summary_zh()
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        if action == "ask_clarification" or not completeness.ready_to_search:
            output = self._base_output(update)
            question = update.clarification_question_zh or completeness.clarification_question_zh
            if question and not _question_already_present(question, output):
                output.append(question)
            self._log_turn(previous_contract, update, None, {"ready": False, "action": "ask_clarification"})
            self.history_summary = self.contract.summary_zh()
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        if action not in {"run_search", "rerank"}:
            output = self._base_output(update)
            self._log_turn(previous_contract, update, None, {"ready": completeness.ready_to_search, "action": action})
            self.history_summary = self.contract.summary_zh()
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        if action == "rerank" and self.last_result is None:
            output = self._base_output(update)
            output.append("我已经记下这个偏好。现在还没有可重排的推荐方案；你给我完整路线和出发时间后，我再按这个偏好搜索。")
            self._log_turn(previous_contract, update, None, {"ready": completeness.ready_to_search, "action": "rerank_without_results"})
            message_text = "\n\n".join(part for part in output if part)
            self.messages.append({"role": "assistant", "content": message_text})
            return ChatTurnResult(ok=True, message=message_text, update=update, contract=self.contract)

        full_search = action == "run_search"
        rerank_only = action == "rerank" and self.last_result is not None
        if rerank_only:
            result = self.pipeline.rerank(previous_result=self.last_result, contract=self.contract)
        else:
            result = self.pipeline.run(self.contract)
            full_search = True
        self.last_result = result
        self.displayed_options = list(result.recommendations)

        output = self._base_output(update)
        output.append(self.display.action_summary(update.update_type, rerank_only=rerank_only))
        summary = self.display.contract_summary_zh(self.contract)
        if summary:
            output.append(summary)
        special_warning = self.display.special_requirement_warning_panel(self.contract)
        if special_warning:
            output.append(special_warning)
        output.append(self.display.agent_progress(result, contract=self.contract, rerank_only=rerank_only))
        table = self.display.recommendations_table(
            result.recommendations,
            contract=self.contract,
            exclusions=result.exclusions,
        )
        output.append(table)
        if result.warnings:
            for warning in result.warnings:
                output.append(warning)
        special_questions = [
            item.clarification_question_zh
            for item in self.contract.special_requirements
            if item.active and item.requires_clarification and item.clarification_question_zh
        ]
        if special_questions:
            output.append(f"我还想确认一个细节：{special_questions[0]}")
        if result.recommendations:
            output.append(
                self.display.followup_prompt(
                    result.recommendations,
                    contract=self.contract,
                    exclusions=result.exclusions,
                )
            )
        self._log_turn(
            previous_contract,
            update,
            result,
            {
                "ready": True,
                "full_search_ran": full_search,
                "rerank_only": rerank_only,
                "provider_call_count": result.provider_call_count,
            },
        )
        self.history_summary = self.contract.summary_zh()
        debug_summary = self._debug_summary(update, result, full_search=full_search, rerank_only=rerank_only)
        if self.debug and debug_summary:
            output.append(debug_summary)
            self._write_debug_log(debug_summary)
        message_text = "\n\n".join(part for part in output if part)
        self.messages.append({"role": "assistant", "content": message_text})
        return ChatTurnResult(
            ok=True,
            message=message_text,
            update=update,
            contract=self.contract,
            pipeline_result=result,
            full_search_ran=full_search,
            rerank_only=rerank_only,
            provider_call_count=result.provider_call_count,
            debug_summary=debug_summary,
        )

    def _should_run_full_search(self, update) -> bool:
        if self.last_result is None:
            return True
        if update.update_type == "create_new":
            return True
        if update.should_rerun_search or update.should_search:
            return True
        return False

    def _next_action(self, update) -> str:
        action = getattr(update, "next_action", "no_op") or "no_op"
        if action != "no_op":
            return action
        if update.update_type == "explain_option":
            return "explain_result"
        if update.update_type == "export":
            return "export"
        if update.update_type in {"help", "smalltalk", "quit"}:
            return update.update_type
        if update.next_action == "tool_query":
            return "tool_query"
        if update.should_rerank_only:
            return "rerank"
        if update.should_search or update.should_rerun_search:
            return "run_search"
        if getattr(update, "clarification_question_zh", None):
            return "ask_clarification"
        return "no_op"

    def _base_output(self, update) -> list[str]:
        parts = [update.user_facing_ack_zh]
        if self.show_reasoning:
            parts.append("LLM: DeepSeek ✓ schema update validated")
            trace = self.display.decision_trace_text(
                update.decision_trace,
                max_items=None if self.debug else 3,
            )
            if trace:
                parts.append(trace)
        return parts

    def _log_turn(self, previous_contract, update, result, quality_flags) -> None:
        if not self.contract:
            return
        target_contract = self.contract.model_dump(mode="json")
        turn_sample = {
            "messages": list(self.messages),
            "previous_contract": previous_contract,
            "target_update": update.model_dump(mode="json"),
            "target_contract_after_update": target_contract,
        }
        self.turn_samples.append(turn_sample)
        self.contract_snapshots.append(target_contract)
        if not self.logger:
            return
        self.logger.add_message("assistant", update.user_facing_ack_zh or "")
        self.logger.log_turn(
            previous_contract=previous_contract,
            target_update=turn_sample["target_update"],
            target_contract_after_update=target_contract,
            result_summary=_result_summary(result) if result else {},
            quality_flags=quality_flags,
        )
        self.logger.log_session(target_final_contract=target_contract)

    def _detail_for_update(self, update) -> str:
        if not self.last_result or not self.last_result.recommendations:
            return "还没有可解释的推荐方案。先告诉我你要查哪段行程。"
        index = update.selected_option_index or 1
        if index < 1 or index > len(self.last_result.recommendations):
            return f"当前只展示了 {len(self.last_result.recommendations)} 个方案，请选择其中一个。"
        rec = self.last_result.recommendations[index - 1]
        return self.display.detail_view(
            rec,
            index=index,
            contract=self.contract,
            exclusions=self.last_result.exclusions,
        )

    def _tool_query_output(self, update) -> list[str]:
        if not update.tool_requests:
            return ["当前没有可执行的工具请求。你可以说：目的地天气怎么样，或 奥斯丁机场是哪个。"]
        context = ToolRequestContext(contract=self.contract)
        outputs: list[str] = []
        for request in update.tool_requests:
            result = self.tool_router.execute(request, context)
            outputs.append(result.user_facing_text_zh)
            if self.debug:
                outputs.append(
                    self.display._panel(
                        "Tool Diagnostics",
                        "\n".join(
                            [
                                f"- tool: {result.tool_name}",
                                f"- success: {result.success}",
                                f"- source: {result.source}",
                                f"- error: {result.error_message or '(none)'}",
                            ]
                        ),
                        border_style="magenta",
                    )
                )
        return outputs

    def export(self, export_dir: Path | None = None) -> Path:
        if export_dir is None:
            root = Path("runs") / "chat_exports"
            export_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir.mkdir(parents=True, exist_ok=True)
        recommendations = (
            [rec.model_dump(mode="json") for rec in self.last_result.recommendations]
            if self.last_result
            else []
        )
        final_contract = self.contract.model_dump(mode="json") if self.contract else {}
        (export_dir / "transcript.txt").write_text(_transcript_text(self.messages), encoding="utf-8")
        (export_dir / "updates.json").write_text(json.dumps(self.updates, ensure_ascii=False, indent=2), encoding="utf-8")
        (export_dir / "contracts.json").write_text(
            json.dumps(self.contract_snapshots, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (export_dir / "final_contract.json").write_text(
            json.dumps(final_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (export_dir / "recommendations.json").write_text(
            json.dumps(recommendations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (export_dir / "sft_turn_samples.jsonl").open("w", encoding="utf-8") as f:
            for row in self.turn_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (export_dir / "sft_session_sample.json").write_text(
            json.dumps(
                {"messages": self.messages, "target_final_contract": final_contract},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.last_export_dir = export_dir
        return export_dir

    def _debug_summary(self, update, result, *, full_search: bool, rerank_only: bool) -> str:
        if not self.debug:
            return ""
        meta = getattr(self.requirement_agent, "last_meta", {}) or {}
        update_path = self._save_debug_update(update)
        excluded = result.exclusions.excluded_airports if result else []
        filtered_count = max(0, len(result.itineraries) - len(result.recommendations)) if result else 0
        lines = [
            "Debug diagnostics",
            "Debug Diagnostics",
            "LLM",
            f"- model: {meta.get('model', '(unknown)')}",
            f"- latency: {meta.get('latency_ms', '(unknown)')} ms",
            f"- tokens: {meta.get('token_usage', {})}",
            f"- update_type: {update.update_type}",
            f"- confidence: {update.confidence:.2f}",
            f"- trace count: {len(update.decision_trace)}",
            f"- route validation diagnostics: {getattr(self.requirement_agent, 'last_validation_diagnostics', [])}",
            "",
            "Contract",
            f"- origin: {self.contract.trip.origin_airport if self.contract else '(none)'}",
            f"- destination: {self.contract.trip.destination_airport if self.contract else '(none)'}",
            f"- acceptable hubs: {self.contract.geography.acceptable_origin_hubs if self.contract else []}",
            f"- excluded airports: {excluded}",
            "",
            "Pipeline",
            f"- hub pairs: {len(result.hub_pairs) if result else 0}",
            f"- search tasks: {len(result.search_tasks) if result else 0}",
            f"- search_task_count: {len(result.search_tasks) if result else 0}",
            f"- provider calls: {result.provider_call_count if result else 0}",
            f"- itineraries: {len(result.itineraries) if result else 0}",
            f"- filtered: {filtered_count}",
            f"- full_search_ran: {full_search}",
            f"- rerank_only: {rerank_only}",
            "",
            "Artifacts",
            f"- update JSON: {update_path or '(debug dir unavailable)'}",
            f"- logs: {self.debug_dir / 'logs.txt' if self.debug_dir else '(debug dir unavailable)'}",
        ]
        return self.display._panel("Debug Diagnostics", "\n".join(lines), border_style="magenta")

    def _save_debug_update(self, update) -> str:
        if not self.debug or not self.debug_dir:
            return ""
        path = self.debug_dir / f"turn_{len(self.updates):03d}_update.json"
        path.write_text(json.dumps(update.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _write_debug_log(self, text: str) -> None:
        if not self.debug_dir:
            return
        with (self.debug_dir / "logs.txt").open("a", encoding="utf-8") as f:
            f.write(text + "\n\n")

    def _handle_internal_error(self, exc: Exception, message: str) -> ChatTurnResult:
        self.last_result = None
        self.displayed_options = []
        path = self._save_error_diagnostics(exc, message)
        output = INTERNAL_ERROR_MESSAGE
        debug_summary = ""
        if self.debug:
            tb = traceback.format_exc()
            debug_summary = "\n".join(
                [
                    "Debug traceback",
                    tb,
                    f"diagnostics: {path}",
                ]
            )
            output = f"{output}\n\n{debug_summary}"
            self._write_debug_log(debug_summary)
        self.messages.append({"role": "assistant", "content": output})
        return ChatTurnResult(ok=False, message=output, contract=self.contract, debug_summary=debug_summary)

    def _save_error_diagnostics(self, exc: Exception, message: str) -> str:
        root = self.debug_dir or (Path("runs") / "chat_errors" / datetime.now().strftime("%Y%m%d_%H%M%S"))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"turn_error_{len(self.messages):03d}.txt"
        payload = "\n".join(
            [
                f"user_message: {message}",
                f"exception: {exc.__class__.__name__}: {exc}",
                "",
                traceback.format_exc(),
            ]
        )
        path.write_text(payload, encoding="utf-8")
        return str(path)


def _result_summary(result: PipelineResult | None) -> dict:
    if not result:
        return {}
    return {
        "recommendation_count": len(result.recommendations),
        "top_routes": ["->".join(rec.itinerary.route) for rec in result.recommendations[:5]],
        "provider_call_count": result.provider_call_count,
        "warnings": result.warnings,
    }


def _transcript_text(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{role}: {message['content']}")
    return "\n\n".join(lines)


def _question_already_present(question: str, output_parts: list[str]) -> bool:
    text = "\n".join(output_parts)
    if question in text:
        return True
    topic = question.split("？", 1)[0].split("?", 1)[0].strip()
    return bool(topic and topic in text)
