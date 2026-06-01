"""Constraint models for travel search."""

from pydantic import BaseModel

from travel_agent.models.user_request import HardConstraints, SoftConstraints

# Re-export for convenience; primary definitions live in user_request.py
__all__ = ["HardConstraints", "SoftConstraints", "SearchConstraints"]

from travel_agent.models.user_request import SearchConstraints  # noqa: E402
