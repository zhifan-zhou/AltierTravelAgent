"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from travel_agent.llm.schemas import (
    LLMParsedRequest, LLMPreferenceInference, ClarificationPlan,
    LLMSearchStrategyAdvice, LLMExplanation, FollowUpIntent, LLMCallDiagnostic,
)


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers.

    Every method returns a Pydantic model or None on failure.
    All flight truth remains with the deterministic pipeline.
    """

    provider_name: str = "base"

    @abstractmethod
    async def parse_travel_request(
        self, query: str, context: dict | None = None
    ) -> LLMParsedRequest | None:
        """Extract structured travel request from natural language query."""

    @abstractmethod
    async def infer_preferences(
        self, query: str, current_request: dict | None = None
    ) -> LLMPreferenceInference | None:
        """Infer ranking weights from user preferences."""

    @abstractmethod
    async def generate_clarifying_questions(
        self, context: dict | None = None
    ) -> ClarificationPlan | None:
        """Generate clarifying questions for missing/ambiguous fields."""

    @abstractmethod
    async def reason_search_strategy(
        self, context: dict | None = None
    ) -> LLMSearchStrategyAdvice | None:
        """Provide advisory input on search strategy. Does not replace HubSplit."""

    @abstractmethod
    async def explain_recommendations(
        self, context: dict | None = None, language: str = "zh"
    ) -> LLMExplanation | None:
        """Generate human-readable explanation of recommendations."""

    @abstractmethod
    async def interpret_followup(
        self, message: str, state: dict | None = None
    ) -> FollowUpIntent | None:
        """Interpret a user follow-up message."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is reachable."""
