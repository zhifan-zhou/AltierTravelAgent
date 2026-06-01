"""Conversation State Agent: maintains interactive chat session state."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import TravelAgentResult, IntakeOutput


class ConversationState(BaseModel):
    """Full state for an interactive chat session."""
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    original_query: str = ""
    parsed_request: IntakeOutput | None = None
    clarification_answers: dict = Field(default_factory=dict)
    constraints: dict = Field(default_factory=dict)
    preferences: dict = Field(default_factory=dict)
    last_result: TravelAgentResult | None = None
    last_recommendations: list[dict] = Field(default_factory=list)
    selected_itinerary_id: str | None = None
    turn_count: int = 0
    output_dir: str = ""

    def save(self) -> Path:
        """Save session state to disk."""
        if not self.output_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"runs/interactive/{self.session_id}"
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        state_path = path / "state.json"
        state_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str)
        )
        return path


class ConversationStateAgent(BaseAgent[dict, ConversationState]):
    """Maintains interactive chat session state."""

    name = "conversation_state"

    def __init__(self):
        super().__init__()
        self._state = ConversationState()

    @property
    def state(self) -> ConversationState:
        return self._state

    async def execute(self, data: dict) -> ConversationState:
        """Update state with new data."""
        for key, value in data.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._state.turn_count += 1
        return self._state
