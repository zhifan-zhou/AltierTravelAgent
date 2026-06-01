"""Constraint Update Service — safely applies ChatIntent to session constraints."""

from __future__ import annotations

from travel_agent.llm.schemas import ChatIntent


class ConstraintUpdateService:
    """Apply a ChatIntent to current session state safely.

    Rules:
    - Never mutate primary origin/destination from hub/preference updates.
    - Alternative hubs (acceptable_origin_hubs) are additions, not replacements.
    - Avoid airports are cumulative unless user explicitly clears.
    - Profile changes may trigger rerank or rerun depending on what else changed.
    """

    def apply_intent(self, intent: ChatIntent, current_state: dict) -> dict:
        """Apply a ChatIntent to current constraints, returning a new state dict."""
        updated = dict(current_state)

        # Profile
        if intent.profile:
            updated["current_profile"] = intent.profile

        # Cabin (triggers rerun)
        if intent.cabin:
            updated["cabin"] = intent.cabin

        # Budget
        if intent.budget_usd:
            updated["budget_usd"] = intent.budget_usd

        # Risk preference
        if intent.risk_preference:
            updated["risk_preference"] = intent.risk_preference

        # Split ticket policy
        if intent.split_ticket_policy:
            updated["split_ticket_policy"] = intent.split_ticket_policy
            if intent.split_ticket_policy == "avoid":
                updated["accepts_split_ticket"] = False
            elif intent.split_ticket_policy == "prefer":
                updated["accepts_split_ticket"] = True

        # Nearby hub policy
        if intent.nearby_hub_policy:
            updated["nearby_hub_policy"] = intent.nearby_hub_policy

        # Acceptable origin hubs — append, don't replace
        if intent.acceptable_origin_hubs:
            existing = set(updated.get("acceptable_origin_hubs", []))
            for h in intent.acceptable_origin_hubs:
                existing.add(h)
            updated["acceptable_origin_hubs"] = list(existing)

        # Avoid airports — append
        if intent.avoid_airports:
            existing = set(updated.get("avoid_airports", []))
            for a in intent.avoid_airports:
                existing.add(a)
            updated["avoid_airports"] = list(existing)

        # Airline quality priority
        if intent.airline_quality_priority:
            updated["airline_quality_priority"] = intent.airline_quality_priority

        # Constraints dict updates
        for k, v in intent.constraint_updates.items():
            if k != "scoring_profile":
                updated[k] = v

        return updated

    def needs_rerun_search(self, intent: ChatIntent) -> bool:
        """Determine if this intent requires re-running the full search pipeline."""
        if intent.needs_rerun_search:
            return True
        if intent.intent_type == "new_search":
            return True
        if intent.intent_type == "refine_search":
            return True
        # Cabin change always needs rerun
        if intent.cabin:
            return True
        # Hub changes need rerun
        if intent.acceptable_origin_hubs or intent.avoid_airports:
            return True
        # Split ticket policy change typically needs rerun
        if intent.split_ticket_policy:
            return True
        return False

    def needs_rerank_only(self, intent: ChatIntent) -> bool:
        """Determine if this intent only needs reranking of existing candidates."""
        if intent.needs_rerank_only:
            return True
        if intent.intent_type == "rerank":
            return True
        # Profile-only change can rerank
        if intent.profile and not self.needs_rerun_search(intent):
            return True
        return False
