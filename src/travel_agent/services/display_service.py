"""User-facing Rich-style rendering for chat."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from travel_agent.contract.compiler import ExclusionRules
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.contract.special_requirements import SpecialRequirementInterpreter
from travel_agent.llm.schemas import DecisionTraceItem
from travel_agent.pipeline.types import FlightOffer, PipelineResult, Recommendation
from travel_agent.services.airline_service import AirlineService
from travel_agent.services.airport_service import AirportService


SPECIAL_AIRPORT_LABELS = {
    "PVG": "上海浦东",
    "SHA": "上海虹桥",
    "JFK": "纽约 JFK",
    "EWR": "纽瓦克 EWR",
    "LGA": "纽约拉瓜迪亚 LGA",
    "IAD": "华盛顿杜勒斯 IAD",
    "DCA": "华盛顿里根 DCA",
    "BWI": "巴尔的摩 BWI",
}

TYPE_LABELS = {
    "cheapest_reasonable": "最省钱",
    "best_overall": "综合最优",
    "lowest_risk": "最低风险",
    "fastest": "时间最短",
    "airline_priority": "主流航司优先",
    "baseline": "最低风险",
    "基准": "最低风险",
}

PROFILE_LABELS = {
    "cheapest": "最便宜优先",
    "balanced": "综合均衡",
    "airline_priority": "主流航司优先",
    "low_risk": "低风险优先",
    "fastest": "时间最短",
}

SPECIAL_CATEGORY_LABELS = {
    "pet_travel": "宠物同行",
    "family_or_elderly": "家人同行",
    "child_travel": "儿童同行",
    "accessibility": "无障碍/协助",
    "heavy_baggage": "较多行李",
    "visa_constraint": "签证/过境",
    "alliance_preference": "航司联盟偏好",
    "overnight_avoidance": "避免红眼/过夜",
    "stopover_request": "停留需求",
    "meal_or_medical": "餐食/医疗",
    "unknown": "特殊需求",
}

LEG_LABELS = {
    "ground_cn": "接驳",
    "international": "国际段",
    "domestic_us": "美国国内段",
    "direct": "组合航段",
}

EXPORT_FILES = [
    "transcript.txt",
    "updates.json",
    "contracts.json",
    "final_contract.json",
    "recommendations.json",
    "sft_turn_samples.jsonl",
    "sft_session_sample.json",
]


class DisplayService:
    headers = ["#", "推荐类型", "路线", "航司", "总价", "节省", "风险", "数据", "为什么推荐"]

    def __init__(
        self,
        *,
        airport_service: AirportService | None = None,
        airline_service: AirlineService | None = None,
    ):
        self.airports = airport_service or AirportService()
        self.airlines = airline_service or AirlineService()
        self.specials = SpecialRequirementInterpreter()

    def opening_screen_text(self) -> str:
        return "\n".join(
            [
                "AI 出行管家 · Planning Travel Demo",
                "用自然语言描述路线、预算和偏好，我会给出可行动的旅行草案。",
                "",
                "核心能力：",
                "• 临近大城市组合搜索：温州 → 上海/杭州/南京 → 美国枢纽 → 目的地",
                "• 多轮自然语言修改：排除机场、换目的地、调整航司/风险偏好",
                "• 可解释推荐：价格、风险、航司、数据来源都清楚展示",
                "• 逐日行程、粗略预算和约束风险提醒",
                "",
                "试试这样说：",
                "1. 温州到匹兹堡，六月初，可以从上海走，越便宜越好",
                "2. 我其实想看看宁波到迈阿密",
                "3. 可以去上海浦东，但不去虹桥",
                "4. 帮我安排三天行程，再估算一下预算",
                "5. 不要纽约转",
                "6. 解释第1个",
                "7. 导出",
            ]
        )

    def opening_tips(self) -> str:
        return "输入 quit 退出。你可以随时改变目的地、排除机场、调整偏好。"

    def decision_trace_text(self, trace: list[DecisionTraceItem], *, max_items: int | None = 3) -> str:
        if not trace:
            return ""
        visible = trace if max_items is None else trace[:max_items]
        lines = [f"DeepSeek 判断：{trace[0].decision}", "", "需求决策"]
        for idx, item in enumerate(visible, start=1):
            fields = ", ".join(item.affected_fields) if item.affected_fields else "无"
            lines.extend(
                [
                    f"{idx}. {item.step}",
                    f"   证据：{item.evidence}",
                    f"   决策：{item.decision}",
                    f"   更新字段：{fields}",
                ]
            )
        if max_items is not None and len(trace) > max_items:
            lines.append(f"... 还有 {len(trace) - max_items} 条决策，使用 --debug 查看完整 trace")
        return self._panel("需求理解", "\n".join(lines), border_style="cyan")

    def action_summary(self, update_type: str, *, rerank_only: bool = False) -> str:
        if update_type == "create_new":
            text = "新搜索：正在搜索组合路线，并重新构建路线候选..."
        elif update_type == "clarification_answer":
            text = "信息已补齐：正在搜索组合路线..."
        elif rerank_only or update_type == "add_preference":
            text = "偏好变更：正在重新排序现有方案..."
        elif update_type == "add_special_requirement":
            text = "特殊需求：已记录需求，正在重新评估路线风险和衔接..."
        elif update_type in {"add_constraint", "remove_constraint", "modify_existing"}:
            text = "硬约束变更：正在重新搜索..."
        elif update_type == "explain_option":
            text = "解释方案：正在生成方案详情..."
        elif update_type == "export":
            text = "导出：正在保存会话数据..."
        else:
            text = "正在处理你的请求..."
        return f"▶ {text}"

    def contract_summary_zh(self, contract: TravelRequirementContract) -> str:
        if not contract or not contract.trip.origin_airport:
            return ""
        origin = self.airport_label(contract.trip.origin_airport)
        dest = self.airport_label(contract.trip.destination_airport)
        hubs = self._airport_list_label(contract.geography.acceptable_origin_hubs) or "自动选择"
        avoids = self._airport_list_label(contract.geography.avoid_airports) or "无"
        profile = PROFILE_LABELS.get(contract.ranking.profile, contract.ranking.profile)
        price_pref = "越便宜越好" if contract.ranking.price_priority == "high" else profile
        date_text = (
            contract.time.departure_window_text
            or contract.time.departure_text
            or ("日期未定，按灵活窗口" if contract.time.flexible_date_confirmed else "未指定")
        )
        lines = [
            f"路线：{origin} → {dest}",
            f"偏好：{price_pref}",
            f"排序：{profile}",
            f"可接受出发枢纽：{hubs}",
            f"排除机场：{avoids}",
            f"舱位：{contract.cabin.cabin}",
            f"日期：{date_text}",
            "数据模式：mock demo",
        ]
        special_lines = self._special_requirement_summary_lines(contract)
        if special_lines:
            lines.extend(["", "特殊需求：", *special_lines])
        return self._panel("当前需求", "\n".join(lines), border_style="blue")

    def agent_progress(
        self,
        result: PipelineResult,
        *,
        contract: TravelRequirementContract,
        rerank_only: bool = False,
    ) -> str:
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        for rec in result.recommendations:
            if rec.risk.risk_level in risk_counts:
                risk_counts[rec.risk.risk_level] += 1
        profile = PROFILE_LABELS.get(contract.ranking.profile, contract.ranking.profile)
        retrieval = (
            "复用上一轮候选报价"
            if rerank_only
            else f"找到 {len(result.offers)} 个候选报价"
        )
        lines = [
            "正在运行多 Agent 搜索流程：",
            "✓ 正在生成候选枢纽组合... 已生成搜索约束",
            f"✓ 正在生成候选枢纽组合... 生成 {len(result.hub_pairs)} 组候选组合",
            f"✓ 正在检索候选航段... {retrieval}",
            f"✓ 正在组合完整路线... 生成 {len(result.itineraries)} 条完整路线",
            f"✓ 正在评估风险和排序... 标记 {risk_counts['low']} 条低风险 / {risk_counts['medium']} 条中风险 / {risk_counts['high']} 条高风险",
            f"✓ 完成，找到 {len(result.recommendations)} 个方案。按“{profile}”输出前 {min(5, len(result.recommendations))} 条",
        ]
        return self._panel("Agent 进度", "\n".join(lines), border_style="green")

    def recommendations_table(
        self,
        recommendations: list[Recommendation],
        *,
        contract: TravelRequirementContract,
        exclusions: ExclusionRules,
    ) -> str:
        guarded = [
            rec for rec in recommendations if self._display_guard(rec, contract=contract, exclusions=exclusions)
        ]
        if not guarded:
            return self.no_route_message(contract=contract, exclusions=exclusions)

        chunks = ["推荐方案"]
        for idx, rec in enumerate(guarded[:3], start=1):
            chunks.append(self._recommendation_card(rec, idx, contract))
        if len(guarded) > 3:
            chunks.append(self._remaining_recommendations_table(guarded[3:5], start_index=4, contract=contract))
        if any(
            offer.source.startswith("mock")
            for rec in guarded[:5]
            for offer in rec.itinerary.offers
        ):
            chunks.append(self.mock_warning_panel())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def detail_view(
        self,
        recommendation: Recommendation,
        *,
        index: int,
        contract: TravelRequirementContract,
        exclusions: ExclusionRules,
    ) -> str:
        if not self._display_guard(recommendation, contract=contract, exclusions=exclusions):
            return self._panel("方案已失效", "这个方案已经不符合当前限制，我没有继续展示旧结果。", border_style="yellow")

        route = self.format_route(recommendation)
        label = self.recommendation_label(recommendation, index, contract)
        savings = self._format_savings(recommendation)
        header = "\n".join(
            [
                f"路线：{route}",
                f"总价：${recommendation.itinerary.total_price_usd:.0f}    预计节省：{savings}",
                f"风险：{self.risk_label(recommendation.risk.risk_level)}      数据：{self.data_quality_label(recommendation)}",
            ]
        )
        lines = [self._panel(f"方案 {index}：{label}", header, border_style="cyan"), "", "路线明细"]
        for leg_idx, offer in enumerate(recommendation.itinerary.offers, start=1):
            lines.extend(self._offer_detail_lines(leg_idx, offer))
            lines.append("")
        lines.extend(
            [
                "为什么推荐",
                f"- {self.short_reason(recommendation, contract)}",
                "- 主干段从更大枢纽出发，组合空间更大",
                "- 当前排序偏好下价格/风险权衡较好",
                "",
                "风险提醒 / 风险原因",
            ]
        )
        if recommendation.risk.warnings:
            lines.extend(f"- {warning}" for warning in recommendation.risk.warnings)
        else:
            lines.append("- 当前方案没有明显额外风险，但仍需预留足够转机时间")
        if recommendation.itinerary.route_type == "hub_split":
            lines.append("- 分段/多段方案可能需要重新托运行李或重新安检")
        special_notes = self.specials.interpret(contract.special_requirements).detail_view_notes
        if special_notes:
            lines.extend(["", "特殊需求提醒"])
            lines.extend(f"- {note}" for note in special_notes)
        lines.extend(
            [
                "",
                "数据说明",
                f"- {self.data_quality_note(recommendation)}",
            ]
        )
        return "\n".join(lines).strip()

    def no_route_message(self, *, contract: TravelRequirementContract, exclusions: ExclusionRules) -> str:
        constraints = [
            f"- 避开机场：{self._airport_list_label(exclusions.excluded_airports)}"
            if exclusions.excluded_airports
            else "- 当前没有额外排除机场"
        ]
        body = "\n".join(
            [
                "在当前限制下没有找到合适方案。",
                "没有找到满足当前限制的方案",
                "",
                "当前硬约束：",
                *constraints,
                "",
                "你可以尝试：",
                "- 允许杭州/南京作为出发枢纽",
                "- 放宽当前排除机场限制",
                "- 改成低风险优先但允许一次中转",
            ]
        )
        return self._panel("没有可用方案", body, border_style="yellow")

    def mock_warning_panel(self) -> str:
        body = "\n".join(
            [
                "以下是演示航班数据，不代表真实价格或可预订结果。",
                "mock/demo 航班不可下单、不可锁价；部分 fallback 数值仅用于功能演示。",
            ]
        )
        return self._panel("⚠ 数据说明", body, border_style="yellow")

    def export_success_panel(self, export_dir: Path) -> str:
        body = "\n".join(
            [
                "已导出本轮对话、schema 更新、推荐结果和 SFT 样本。",
                "已保存本轮对话、schema 更新、推荐结果和 SFT 样本",
                f"路径：{export_dir}/",
                "",
                "文件：",
                *[f"- {name}" for name in EXPORT_FILES],
            ]
        )
        return self._panel("导出完成", body, border_style="green")

    def followup_prompt(
        self,
        recommendations: list[Recommendation],
        *,
        contract: TravelRequirementContract | None = None,
        exclusions: ExclusionRules | None = None,
    ) -> str:
        if not recommendations:
            return ""
        excluded = set(exclusions.excluded_airports if exclusions else [])
        route_codes = {code for rec in recommendations[:5] for code in rec.itinerary.route}
        suggestions: list[str] = []
        if route_codes & {"JFK", "EWR", "LGA"} and not excluded & {"JFK", "EWR", "LGA"}:
            suggestions.append("不要纽约转")
        if route_codes & {"PVG", "SHA"} and not {"PVG", "SHA"}.issubset(excluded):
            if "SHA" not in excluded:
                suggestions.append("可以去上海浦东，但不去虹桥")
            suggestions.append("我不想去上海")
        if contract and contract.ranking.profile != "airline_priority":
            suggestions.append("主流航司优先")
        if contract:
            effects = self.specials.interpret(contract.special_requirements)
            suggestions.extend(effects.clarification_questions[:1])
            active_impacts = {
                area
                for item in contract.special_requirements
                if item.active
                for area in item.impact_areas
            }
            if active_impacts & {"risk", "connection_time", "self_transfer"}:
                suggestions.extend(["少折腾一点", "不要分开出票"])
        top = recommendations[0]
        if top.risk.risk_level in {"medium", "high"}:
            suggestions.extend(["风险低一点", "不要分开出票"])
        suggestions.extend(["解释第1个", "导出", "quit"])
        unique: list[str] = []
        for item in suggestions:
            if item not in unique:
                unique.append(item)
        body = "\n".join(f"• {item}" for item in unique[:6])
        return self._panel("你可以继续追问", body, border_style="cyan")

    def recommendations_summary(self, recommendations: list[Recommendation]) -> str:
        parts = []
        for rec in recommendations[:5]:
            parts.append(
                f"#{rec.rank} {self.format_route(rec)} ${rec.itinerary.total_price_usd:.0f}"
            )
        return "; ".join(parts)

    def recommendation_label(
        self,
        rec: Recommendation,
        idx: int,
        contract: TravelRequirementContract,
    ) -> str:
        if contract.ranking.profile == "cheapest" and idx == 1:
            return "最省钱"
        if contract.ranking.profile == "airline_priority":
            return "主流航司优先"
        if contract.ranking.profile == "low_risk":
            return "最低风险"
        if contract.ranking.profile == "fastest":
            return "时间最短"
        if rec.recommendation_type in TYPE_LABELS:
            return TYPE_LABELS[rec.recommendation_type]
        if idx == 1:
            return "综合最优"
        if rec.risk.risk_level == "low":
            return "最低风险"
        return "备选"

    def format_route(self, rec: Recommendation) -> str:
        segments = rec.itinerary.segments
        if not segments:
            return " → ".join(self.airport_label(code) for code in rec.itinerary.route)
        parts = [self.airport_label(segments[0].origin)]
        for segment in segments:
            arrow = "⇢" if segment.mode == "ground" else "→"
            parts.append(f"{arrow} {self.airport_label(segment.destination)}")
        return " ".join(parts)

    def format_airlines(self, rec: Recommendation) -> str:
        names = [self.airlines.display_name(code) for code in rec.itinerary.airlines]
        return " + ".join(names) if names else "地面接驳/估算"

    def data_quality_label(self, rec: Recommendation) -> str:
        sources = {offer.source for offer in rec.itinerary.offers}
        estimated = rec.itinerary.has_estimated_data
        real = any(not source.startswith("mock") for source in sources)
        fixed = any(source.startswith("mock_") and source != "mock_fallback" for source in sources)
        if real and (estimated or fixed):
            return "mixed"
        if real:
            return "真实API"
        if estimated:
            return "估算demo"
        return "demo固定"

    def data_quality_note(self, rec: Recommendation) -> str:
        label = self.data_quality_label(rec)
        if label == "估算demo":
            return "当前价格包含估算数据，不是真实市场报价。"
        if label == "demo固定":
            return "当前价格来自 demo 固定数据，不代表实时库存。"
        if label == "mixed":
            return "当前方案混合了真实/模拟数据，需要正式 provider 复核。"
        return "当前方案来自真实 API，但仍需下单前复核。"

    def short_reason(self, rec: Recommendation, contract: TravelRequirementContract) -> str:
        if contract.ranking.profile == "cheapest":
            return "价格最低，风险可接受"
        if contract.ranking.profile == "airline_priority":
            return "主流航司优先，同时满足当前硬约束"
        if contract.ranking.profile == "low_risk":
            return "少折腾，风险较低"
        if contract.ranking.profile == "fastest":
            return "总耗时更短"
        return rec.reason_zh

    def risk_label(self, level: str) -> str:
        mapping = {"low": "低", "medium": "中", "high": "高"}
        return mapping.get(level, level)

    def special_requirement_warning_panel(self, contract: TravelRequirementContract) -> str:
        warnings = self.specials.interpret(contract.special_requirements).warnings_to_display
        if not warnings:
            return ""
        return self._panel("特殊需求提醒", "\n".join(f"- {warning}" for warning in warnings), border_style="yellow")

    def airport_label(self, code: str | None) -> str:
        if not code:
            return "未知"
        code = code.upper()
        if code in SPECIAL_AIRPORT_LABELS:
            label = SPECIAL_AIRPORT_LABELS[code]
            return label if label.endswith(code) else f"{label} {code}"
        row = self.airports.get(code)
        if not row:
            return code
        city = row.get("city_cn") or row.get("city") or code
        return f"{city} {code}"

    def _recommendation_card(
        self,
        rec: Recommendation,
        idx: int,
        contract: TravelRequirementContract,
    ) -> str:
        label = self.recommendation_label(rec, idx, contract)
        body = "\n".join(
            [
                f"路线：{self.format_route(rec)}",
                f"航司：{self.format_airlines(rec)}",
                f"总价/价格：${rec.itinerary.total_price_usd:.0f}    节省：{self._format_savings(rec)}",
                f"风险：{self.risk_label(rec.risk.risk_level)}      数据：{self.data_quality_label(rec)}",
                f"推荐理由：{self.short_reason(rec, contract)}",
            ]
        )
        return self._panel(f"方案 {idx} · {label}", body, border_style="cyan")

    def _remaining_recommendations_table(
        self,
        recommendations: list[Recommendation],
        *,
        start_index: int,
        contract: TravelRequirementContract,
    ) -> str:
        try:
            from rich.console import Console
            from rich.table import Table

            sink = StringIO()
            console = Console(file=sink, width=110, color_system=None, force_terminal=False)
            table = Table(title="其他可选方案", show_header=True)
            for header in ["#", "类型", "路线", "总价", "风险", "数据"]:
                table.add_column(header, no_wrap=False)
            for offset, rec in enumerate(recommendations):
                idx = start_index + offset
                table.add_row(
                    str(idx),
                    self.recommendation_label(rec, idx, contract),
                    self.format_route(rec),
                    f"${rec.itinerary.total_price_usd:.0f}",
                    self.risk_label(rec.risk.risk_level),
                    self.data_quality_label(rec),
                )
            console.print(table)
            return sink.getvalue().rstrip()
        except Exception:
            lines = ["其他可选方案："]
            for offset, rec in enumerate(recommendations):
                idx = start_index + offset
                lines.append(
                    f"{idx}. {self.recommendation_label(rec, idx, contract)} · {self.format_route(rec)} · ${rec.itinerary.total_price_usd:.0f}"
                )
            return "\n".join(lines)

    def _format_savings(self, rec: Recommendation) -> str:
        if rec.savings_vs_baseline_usd <= 0:
            return "—"
        baseline = rec.itinerary.total_price_usd + rec.savings_vs_baseline_usd
        pct = rec.savings_vs_baseline_usd / baseline * 100 if baseline > 0 else 0
        return f"省 ${rec.savings_vs_baseline_usd:.0f} / {pct:.0f}%"

    def _airport_list_label(self, codes: list[str]) -> str:
        return " / ".join(self.airport_label(code) for code in codes)

    def _special_requirement_summary_lines(self, contract: TravelRequirementContract) -> list[str]:
        lines: list[str] = []
        for item in contract.special_requirements:
            if not item.active:
                continue
            label = SPECIAL_CATEGORY_LABELS.get(item.category, item.category or "特殊需求")
            description = item.description_zh or "已记录"
            suffix = ""
            if item.requires_clarification and item.clarification_question_zh:
                suffix = f"；待确认：{item.clarification_question_zh}"
            lines.append(f"- {label}：{description}{suffix}")
        return lines

    def _offer_detail_lines(self, leg_idx: int, offer: FlightOffer) -> list[str]:
        start = self.airport_label(offer.origin)
        end = self.airport_label(offer.destination)
        label = LEG_LABELS.get(offer.leg_type, offer.leg_type)
        arrow = "⇢" if offer.leg_type == "ground_cn" else "→"
        lines = [f"{leg_idx}. {label}：{start} {arrow} {end}"]
        if offer.leg_type == "ground_cn":
            lines.extend(
                [
                    "   - 方式：地面接驳估算",
                    f"   - 时间：约 {offer.estimated_time_hours or 3.0:.1f} 小时",
                    f"   - 成本：约 ${offer.total_price_usd:.0f}",
                ]
            )
            return lines
        airline_codes = [s.airline for s in offer.segments if s.airline and s.airline not in {"GROUND", "MOCK"}]
        airline = " + ".join(self.airlines.display_name(code) for code in dict.fromkeys(airline_codes)) or "估算航司"
        lines.extend(
            [
                f"   - 航司：{airline}",
                "   - 舱位：economy",
                f"   - 价格：${offer.total_price_usd:.0f}",
                f"   - 飞行时间：约 {offer.estimated_time_hours or 14.0:.0f} 小时",
                f"   - 数据：{offer.source} / {offer.confidence}",
            ]
        )
        return lines

    def _display_guard(
        self,
        rec: Recommendation,
        *,
        contract: TravelRequirementContract,
        exclusions: ExclusionRules,
    ) -> bool:
        route = rec.itinerary.route
        if not route:
            return False
        if route[0] != contract.trip.origin_airport:
            return False
        if route[-1] != contract.trip.destination_airport:
            return False
        excluded = set(exclusions.excluded_airports)
        if any(code in excluded for code in route):
            return False
        for segment in rec.itinerary.segments:
            if segment.origin in excluded or segment.destination in excluded:
                return False
        return True

    def _panel(self, title: str, body: str, *, border_style: str = "cyan") -> str:
        try:
            from rich.console import Console
            from rich.panel import Panel

            sink = StringIO()
            console = Console(file=sink, width=110, color_system=None, force_terminal=False)
            console.print(Panel(body, title=title, border_style=border_style, expand=False))
            return sink.getvalue().rstrip()
        except Exception:
            line = f"─ {title} "
            return f"{line}\n{body}"
