"""User Choice Agent: presents options and processes user selections."""

from __future__ import annotations

from pydantic import BaseModel

from travel_agent.agents.base import BaseAgent


class UserChoiceResult(BaseModel):
    """Parsed user choice from interactive input."""
    action: str = "unknown"  # cheapest, best_overall, lowest_risk, explain, rerun, export, reset, quit
    target_index: int | None = None
    constraint_updates: dict = {}
    message: str = ""


class UserChoiceAgent(BaseAgent[str, UserChoiceResult]):
    """Process user selections from interactive chat."""

    name = "user_choice"

    KEYWORD_MAP = {
        "只看最便宜": ("cheapest", {}),
        "最便宜": ("cheapest", {}),
        "cheapest": ("cheapest", {}),
        "综合最优": ("best_overall", {}),
        "best": ("best_overall", {}),
        "最低风险": ("lowest_risk", {}),
        "最安全": ("lowest_risk", {}),
        "不要分开出票": ("refine", {"accepts_split_ticket": False}),
        "不要拆分": ("refine", {"accepts_split_ticket": False}),
        "允许拆分": ("refine", {"accepts_split_ticket": True}),
        "商务舱": ("refine", {"cabin": "business"}),
        "经济舱": ("refine", {"cabin": "economy"}),
        "不要纽约转": ("refine", {"avoid_hubs": ["JFK", "EWR"]}),
        "不要上海走": ("refine", {"avoid_hubs": ["PVG", "SHA"]}),
        "可以从上海走": ("refine", {"acceptable_origin_hubs": ["PVG", "SHA"]}),
        "风险低一点": ("refine", {"risk_tolerance": "low"}),
        "多给几个选择": ("refine", {"more_options": True}),
        "只看最便宜": ("cheapest", {}),
        "综合推荐": ("best_overall", {}),
        "导出": ("export", {}),
        "导出JSON": ("export", {}),
        "帮助": ("help", {}),
        "help": ("help", {}),
        "重新开始": ("reset", {}),
        "reset": ("reset", {}),
        "退出": ("quit", {}),
        "quit": ("quit", {}),
        "q": ("quit", {}),
    }

    async def execute(self, data: str) -> UserChoiceResult:
        msg = data.strip().lower()

        # Check for "解释第N个" pattern
        import re
        m = re.search(r"解释\s*第?\s*(\d+)\s*个?", msg)
        if m:
            return UserChoiceResult(action="explain", target_index=int(m.group(1)))

        # "只看第N个"
        m = re.search(r"第?\s*(\d+)\s*个?", msg)
        if m:
            return UserChoiceResult(action="explain", target_index=int(m.group(1)))

        # Keyword matching
        for keyword, (action, updates) in self.KEYWORD_MAP.items():
            if keyword in msg:
                return UserChoiceResult(action=action, constraint_updates=updates)

        return UserChoiceResult(action="unknown", message=f"未识别的命令: {msg}")
