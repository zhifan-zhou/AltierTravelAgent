"""Intent Router Agent: interprets natural language into structured ChatIntent.

Uses LLM when enabled, falls back to deterministic keyword matching otherwise.
Validates all outputs against known airports/profiles from the data layer.
"""

from __future__ import annotations

import re
import logging

from travel_agent.agents.base import BaseAgent
from travel_agent.llm.schemas import ChatIntent
from travel_agent.llm.prompts import get_llm_client, is_llm_enabled
from travel_agent.models.preference import SCORING_PROFILES, DEFAULT_PROFILE
from travel_agent.services.airport_service import AirportService

logger = logging.getLogger("travel_agent.agents.intent_router")


class IntentRouterAgent(BaseAgent[tuple[str, dict], ChatIntent]):
    """Routes any natural language chat message to a structured ChatIntent.

    LLM-first when enabled, deterministic keyword matching as fallback.
    """

    name = "intent_router"

    # Deterministic fallback patterns (used when LLM is off or fails)
    # IMPORTANT: specific patterns must come BEFORE generic ones
    DETERMINISTIC_PATTERNS: list[tuple[list[str], dict]] = [
        (["纽约转", "JFK转", "EWR转", "纽瓦克转"], {"avoid_airports": ["JFK", "EWR", "LGA"], "intent_type": "refine_search"}),
        (["华盛顿转", "IAD转", "DCA转"], {"avoid_airports": ["IAD", "DCA"], "intent_type": "refine_search"}),
        (["芝加哥转", "ORD转"], {"avoid_airports": ["ORD"], "intent_type": "refine_search"}),
        (["上海", "PVG", "SHA"], {}),  # These are origin hubs, not avoids
        (["杭州", "HGH"], {}),
        (["南京", "NKG"], {}),
        (["香港", "HKG"], {}),
        (["第一个", "第1个", "第 1"], {"selected_option_index": 1, "intent_type": "explain_option"}),
        (["第二个", "第2个", "第 2"], {"selected_option_index": 2, "intent_type": "explain_option"}),
        (["第三个", "第3个", "第 3"], {"selected_option_index": 3, "intent_type": "explain_option"}),
        (["导出", "保存结果"], {"intent_type": "export"}),
        (["退出", "quit", "结束"], {"intent_type": "quit"}),
    ]

    def __init__(self, airport_service: AirportService | None = None):
        super().__init__()
        self._airports = airport_service or AirportService()

    async def execute(self, data: tuple[str, dict]) -> ChatIntent:
        message, state = data
        msg = message.strip()

        # 1. Try LLM if enabled
        if is_llm_enabled("clarification"):
            llm = get_llm_client()
            try:
                result = await llm.interpret_followup(message, state)
                if result and result.confidence > 0.4:
                    return self._build_chat_intent_from_llm(result, msg)
            except Exception as e:
                logger.warning(f"LLM intent routing failed, falling back: {e}")

        # 2. Deterministic fallback
        return self._deterministic_intent(msg)

    def _build_chat_intent_from_llm(self, followup_intent, msg: str) -> ChatIntent:
        """Convert the LLM FollowUpIntent to a ChatIntent with airport validation."""
        intent_type = followup_intent.intent_type
        updates = followup_intent.constraint_updates

        intent = ChatIntent(
            intent_type=intent_type,
            confidence=followup_intent.confidence,
            natural_language_summary_zh=followup_intent.user_message_summary or msg,
            selected_option_index=followup_intent.selected_option_index,
            profile=updates.get("scoring_profile"),
            constraint_updates=updates,
        )

        # Validate airports against known data
        avoid_raw = updates.get("avoid_hubs", [])
        intent.avoid_airports = [a for a in avoid_raw if self._airports.get_airport(a)]

        hubs_raw = updates.get("acceptable_origin_hubs", [])
        intent.acceptable_origin_hubs = [a for a in hubs_raw if self._airports.get_airport(a)]

        # Detect cabin
        if "cabin" in updates and updates["cabin"] in ("business", "economy", "first"):
            intent.cabin = updates["cabin"]
            intent.needs_rerun_search = True

        # Detect rerank vs rerun
        if intent_type == "rerank":
            intent.needs_rerank_only = True
        elif intent_type == "refine_constraints":
            intent.needs_rerun_search = True

        if intent.profile and intent.profile not in SCORING_PROFILES:
            intent.profile = DEFAULT_PROFILE

        return intent

    def _deterministic_intent(self, msg: str) -> ChatIntent:
        """Keyword-based intent detection as fallback."""
        intent = ChatIntent()

        # Check followup command keywords
        from travel_agent.models.preference import PREFERENCE_COMMANDS
        for keyword, profile in PREFERENCE_COMMANDS.items():
            if keyword in msg:
                intent.intent_type = "rerank"
                intent.profile = profile
                intent.needs_rerank_only = True
                intent.confidence = 0.8
                intent.natural_language_summary_zh = f"切换排序偏好为{profile}"
                return intent

        # Check deterministic patterns
        for keywords, updates in self.DETERMINISTIC_PATTERNS:
            if any(kw in msg for kw in keywords):
                intent.confidence = 0.7
                if "intent_type" in updates:
                    intent.intent_type = updates["intent_type"]
                if "avoid_airports" in updates:
                    intent.avoid_airports = updates["avoid_airports"]
                if "selected_option_index" in updates:
                    intent.selected_option_index = updates["selected_option_index"]
                # Also check for profile/risk keywords in same message
                if "低风险" in msg or "稳妥" in msg or "折腾" in msg:
                    intent.profile = "low_risk"
                    intent.risk_preference = "low"
                if "主流航司" in msg or "大牌" in msg or "靠谱" in msg:
                    intent.profile = "airline_priority"
                if "便宜" in msg or "省钱" in msg:
                    intent.profile = "cheapest"
                intent.needs_rerun_search = True
                return intent

        # Generic constraint update detection
        if any(kw in msg for kw in ["不要", "别", "避开", "换一", "换一个", "改成", "少一点"]):
            intent.intent_type = "refine_search"
            intent.needs_rerun_search = True
            intent.confidence = 0.5

            # Detect hub avoidances
            for code in ["JFK", "EWR", "LGA", "IAD", "ORD", "LAX"]:
                airport = self._airports.get_airport(code)
                if airport:
                    for name in [airport.city_cn, airport.city, airport.code]:
                        if name and name in msg:
                            intent.avoid_airports.append(code)

            # Detect origin hub preferences
            if any(w in msg for w in ["上海", "PVG", "SHA"]):
                intent.acceptable_origin_hubs = ["PVG", "SHA"]
                intent.nearby_hub_policy = "prefer"
            elif any(w in msg for w in ["杭州", "HGH"]):
                intent.acceptable_origin_hubs = ["HGH"]
                intent.nearby_hub_policy = "prefer"

            intent.natural_language_summary_zh = msg
            return intent

        # Check for explain_option
        m = re.search(r"第\s*(\d+)\s*个", msg)
        if m:
            intent.intent_type = "explain_option"
            intent.selected_option_index = int(m.group(1))
            intent.confidence = 0.9
            return intent

        if any(kw in msg for kw in ["导出", "保存"]):
            intent.intent_type = "export"
            intent.confidence = 0.9
            return intent

        if any(kw in msg for kw in ["退出", "quit"]):
            intent.intent_type = "quit"
            intent.confidence = 0.9
            return intent

        intent.intent_type = "unknown"
        intent.confidence = 0.1
        return intent
