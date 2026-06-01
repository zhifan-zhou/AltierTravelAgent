"""Base agent class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base for all agents in the pipeline.

    Each agent accepts a typed input model and returns a typed output model.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"travel_agent.agents.{self.name}")

    @abstractmethod
    async def execute(self, data: InputT) -> OutputT:
        """Execute the agent's logic and return structured output."""
        ...
