"""Clarification Agent: decides whether to ask follow-up questions.

Uses deterministic rules by default. LLM optional for question generation.
Max 3 questions per turn. Only asks high-impact questions.
"""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.llm.schemas import ClarificationPlan, ClarificationQuestion
from travel_agent.models.agent_outputs import IntakeOutput
from travel_agent.llm.prompts import get_llm_client, is_llm_enabled


class ClarificationAgent(BaseAgent[IntakeOutput, ClarificationPlan]):
    """Generate clarification questions for ambiguous/missing info."""

    name = "clarification"

    async def execute(self, data: IntakeOutput) -> ClarificationPlan:
        questions: list[ClarificationQuestion] = []
        missing: list[str] = []

        # Deterministic checks
        if not data.origin_text:
            missing.append("origin")
            questions.append(ClarificationQuestion(
                id="origin", question_zh="从哪个城市出发？",
                type="free_text", required=True, reason="出发城市缺失",
            ))
        if not data.destination_text:
            missing.append("destination")
            questions.append(ClarificationQuestion(
                id="dest", question_zh="目的地是哪个城市？",
                type="free_text", required=True, reason="目的城市缺失",
            ))
        if not data.departure_window.start_date:
            missing.append("date")
            questions.append(ClarificationQuestion(
                id="date", question_zh="大概什么时候出发？可以是一个日期，也可以是一个范围。",
                type="free_text", required=False, reason="出发日期不明确",
            ))
        if data.budget_usd is None and "cheap" in data.preferences:
            missing.append("budget")
        if data.accepts_nearby_hubs and data.accepts_split_ticket:
            if len(questions) < 3:
                questions.append(ClarificationQuestion(
                    id="risk", question_zh="更看重省钱还是更看重稳妥？省钱方案通常需要分开出票和转机。",
                    type="single_choice",
                    options=[
                        {"label": "越省钱越好", "value": "cheap_flexible"},
                        {"label": "平衡省钱和稳妥", "value": "balanced"},
                        {"label": "稳妥第一，少折腾", "value": "safe"},
                    ],
                    required=False, reason="权衡省钱和风险",
                ))

        # Cap at 3
        questions = questions[:3]

        # Try LLM enhancement if enabled
        if is_llm_enabled("clarification"):
            llm = get_llm_client()
            ctx = {
                "origin": data.origin_text,
                "destination": data.destination_text,
                "preferences": data.preferences,
                "accepts_nearby": data.accepts_nearby_hubs,
                "deterministic_questions": [q.model_dump() for q in questions],
            }
            llm_plan = await llm.generate_clarifying_questions(ctx)
            if llm_plan and llm_plan.questions:
                questions = llm_plan.questions[:3]

        return ClarificationPlan(
            should_ask=len(questions) > 0,
            questions=questions,
            missing_fields=missing,
            ambiguity_level="medium" if len(missing) > 1 else "low",
        )
