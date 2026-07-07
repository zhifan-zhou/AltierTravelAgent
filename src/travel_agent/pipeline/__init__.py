"""Deterministic search pipeline.

Import concrete services from their modules to avoid eager circular imports with
the planning layer.
"""

__all__ = ["LLMFirstChatSession", "SearchPipelineOrchestrator"]


def __getattr__(name: str):
    if name in __all__:
        from travel_agent.pipeline.orchestrator import (
            LLMFirstChatSession,
            SearchPipelineOrchestrator,
        )

        return {
            "LLMFirstChatSession": LLMFirstChatSession,
            "SearchPipelineOrchestrator": SearchPipelineOrchestrator,
        }[name]
    raise AttributeError(name)
