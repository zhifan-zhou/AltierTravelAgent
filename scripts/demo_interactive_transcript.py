#!/usr/bin/env python3
"""Simulated interactive chat transcript — LLM-first natural language demo."""

from __future__ import annotations

import asyncio

from travel_agent.core.orchestrator import TravelAgentOrchestrator
from travel_agent.core.logging import setup_logging
from travel_agent.agents.clarification_agent import ClarificationAgent
from travel_agent.agents.preference_agent import PreferenceAgent
from travel_agent.agents.intent_router_agent import IntentRouterAgent
from travel_agent.utils.money import format_usd


def A(msg: str) -> None:
    print(f"  AI: {msg}")

def U(msg: str) -> None:
    print(f"  你: {msg}")

def S(msg: str) -> None:
    print(f"     [{msg}]")


async def demo():
    setup_logging()
    orch = TravelAgentOrchestrator()
    clarification = ClarificationAgent()
    preference = PreferenceAgent()
    router = IntentRouterAgent()

    print("=" * 65)
    print("  Travel Agent MVP — LLM-First Natural Language Demo")
    print("=" * 65)
    A("我是你的 AI 出行管家。你可以直接用自然语言描述行程，\n"
      "比如：温州到匹兹堡，可以从上海走，越便宜越好。\n"
      "你想查哪段行程？")

    # ── Turn 1: Natural language query ────────────────────────────────
    query = "我想从温州去匹兹堡上学，可以去上海坐飞机，越便宜越好，但不要太冒险"
    U(query)

    A("正在分析你的需求...")
    intake = await orch.intake.execute(query)
    S(f"解析: {intake.origin_text} → {intake.destination_text}")

    plan = await clarification.execute(intake)
    if plan.should_ask:
        A("想确认几个问题：")
        for q in plan.questions[:2]:
            print(f"     - {q.question_zh}")

    U("6月初，能省钱但别太折腾")

    A("正在搜索航班...")
    result = await orch.run(query, debug=False)

    A(f"为你找到 {len(result.ranking.rankings) if result.ranking else 0} 个方案：")
    if result.ranking:
        for rec in result.ranking.rankings[:3]:
            it = rec.itinerary
            save = f"省${rec.savings_vs_baseline_usd:.0f}" if rec.savings_vs_baseline_usd > 0 else ""
            print(f"     #{rec.rank} ${it.total_price_usd:.0f} {save:>10s} 风险:{rec.risk_assessment.risk_level}")

    A("你可以继续追问，比如：我不想从纽约转 / 主流航司优先 / 解释第1个")

    # ── Turn 2: Natural language refinement ───────────────────────────
    U("我不想从纽约转，低风险一点")
    intent = await router.execute(("我不想从纽约转，低风险一点", {}))
    S(f"Intent: {intent.intent_type} avoid={intent.avoid_airports} profile={intent.profile}")
    A(f"好的，已避开纽约中转，改为低风险排序。")

    U("那如果主流航司优先呢？")
    intent = await router.execute(("那如果主流航司优先呢？", {}))
    S(f"Intent: rerank profile={intent.profile}")
    A("已切换为主流航司优先。更看重航司品质和可靠性。")

    # ── Turn 3: Explain ───────────────────────────────────────────────
    U("第一个为什么这么便宜？")
    intent = await router.execute(("第一个为什么这么便宜？", {}))
    S(f"Intent: {intent.intent_type} idx={intent.selected_option_index}")
    if result.ranking:
        rec = result.ranking.rankings[0]
        A(f"方案 #{rec.rank}: 这条路线的国际段从上海出发，价格更有竞争力，"
          f"但需要自行安排温州到上海的地面接驳。比直飞省约${rec.savings_vs_baseline_usd:.0f}。")

    # ── Turn 4: More natural language ─────────────────────────────────
    U("我爸妈也一起，别太折腾，航司靠谱一点")
    intent = await router.execute(("我爸妈也一起，别太折腾，航司靠谱一点", {}))
    S(f"Intent: profile={intent.profile} risk={intent.risk_preference}")
    A("理解！带父母出行，已切换为稳妥优先。会优先低风险、主流航司的方案。")

    U("导出")
    A("结果已保存至 runs/interactive/ 目录。")

    U("quit")
    A("再见！祝你旅途愉快！")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(demo())
