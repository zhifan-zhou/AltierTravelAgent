"""FollowUp Agent: interprets user follow-up messages in interactive chat."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.llm.schemas import FollowUpIntent
from travel_agent.llm.prompts import get_llm_client, is_llm_enabled


class FollowUpAgent(BaseAgent[tuple[str, dict], FollowUpIntent]):
    """Interpret follow-up messages and translate to constraint updates.

    Uses deterministic keyword matching first, LLM as optional enhancement.
    """

    name = "followup"

    # Deterministic keyword patterns — preference commands from unified metadata
    PREFERENCE_COMMANDS = {
        "主流航司优先": "airline_priority",
        "大牌航司": "airline_priority",
        "靠谱航司": "airline_priority",
        "不要廉航": "airline_priority",
        "不要廉价": "airline_priority",
        "不要低成本": "airline_priority",
        "少折腾": "low_risk",
        "稳妥一点": "low_risk",
        "不想折腾": "low_risk",
        "稳妥优先": "low_risk",
        "只要便宜": "cheapest",
        "最便宜优先": "cheapest",
        "越便宜越好": "cheapest",
        "价格优先": "cheapest",
        "时间短一点": "fastest",
        "快一点": "fastest",
        "时间优先": "fastest",
        "最快": "fastest",
        "综合均衡": "balanced",
        "均衡": "balanced",
        "平衡": "balanced",
        "综合优先": "balanced",
        "恢复默认": "balanced",
        "重置排序": "balanced",
        "取消偏好": "balanced",
        "undo": "balanced",
        "reset": "balanced",
        "默认排序": "balanced",
    }

    PATTERNS = [
        (["只看最便宜", "最便宜", "cheapest"], "refine_constraints", {"prefer_cheapest": True}),
        (["不要分开出票", "不要拆分", "no split"], "refine_constraints", {"accepts_split_ticket": False}),
        (["允许拆分", "可以拆分", "分开出票也行"], "refine_constraints", {"accepts_split_ticket": True}),
        (["商务舱", "business"], "refine_constraints", {"cabin": "business"}),
        (["经济舱", "economy"], "refine_constraints", {"cabin": "economy"}),
        (["不要纽约转", "不要纽瓦克"], "refine_constraints", {"avoid_hubs": ["JFK", "EWR"]}),
        (["不要华盛顿转"], "refine_constraints", {"avoid_hubs": ["IAD", "DCA"]}),
        (["不要芝加哥转"], "refine_constraints", {"avoid_hubs": ["ORD"]}),
        (["可以从上海走", "上海也行"], "refine_constraints", {"acceptable_origin_hubs": ["PVG", "SHA"]}),
        (["多给几个", "再看看"], "refine_constraints", {"more_options": True}),
        (["导出", "export", "保存"], "export", {}),
        (["重新开始", "reset", "重来"], "reset", {}),
        (["退出", "quit", "结束"], "quit", {}),
    ]

    async def execute(self, data: tuple[str, dict]) -> FollowUpIntent:
        message, state = data
        msg = message.strip()

        # 1. Preference commands (reranking, including reset/undo)
        for keyword, profile in self.PREFERENCE_COMMANDS.items():
            if keyword in msg:
                return FollowUpIntent(
                    intent_type="rerank",
                    constraint_updates={"scoring_profile": profile},
                    user_message_summary=message,
                    confidence=0.9 if len(msg) < 20 else 0.7,
                )

        # 2. Constraint refinements (non-preference)
        for keywords, intent_type, updates in self.PATTERNS:
            if any(kw in msg for kw in keywords):
                return FollowUpIntent(
                    intent_type=intent_type,
                    constraint_updates=updates,
                    user_message_summary=message,
                    confidence=0.9 if len(msg) < 20 else 0.7,
                )

        # 3. "解释第N个"
        import re
        m = re.search(r"解释?\s*第?\s*(\d+)", msg)
        if m:
            return FollowUpIntent(
                intent_type="explain_option",
                selected_option_index=int(m.group(1)),
                user_message_summary=f"解释第{m.group(1)}个方案",
                confidence=0.9,
            )

        # 3. LLM enhancement if enabled
        if is_llm_enabled("clarification"):
            llm = get_llm_client()
            result = await llm.interpret_followup(message, state)
            if result and result.confidence > 0.6:
                return result

        # 4. Fallback
        return FollowUpIntent(
            intent_type="unknown",
            user_message_summary=message,
            confidence=0.1,
        )
