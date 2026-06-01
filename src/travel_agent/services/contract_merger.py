"""ContractMerger — safely applies ContractUpdate to TravelRequirementContract."""

from __future__ import annotations

from travel_agent.models.requirement_contract import (
    TravelRequirementContract, TravelRequirementContractUpdate,
    Assumption,
)


class ContractMerger:
    """Apply a ContractUpdate to a Contract with safety rules."""

    def apply(self, contract: TravelRequirementContract,
              update: TravelRequirementContractUpdate) -> TravelRequirementContract:
        # Apply field updates (top-level fields only)
        for key, value in update.field_updates.items():
            if hasattr(contract, key) and not key.startswith("trip"):
                setattr(contract, key, value)

        # Trip updates — never overwrite primary origin/destination unless create_new
        if update.update_type == "create_new":
            if "trip" in update.field_updates:
                trip_data = update.field_updates["trip"]
                for k, v in trip_data.items():
                    if hasattr(contract.trip, k):
                        setattr(contract.trip, k, v)

        # Add constraints
        for c in update.constraints_to_add:
            contract.constraints.hard_constraints.append(c)

        # Remove constraints by value (best-effort match)
        for v in update.constraints_to_remove:
            contract.constraints.hard_constraints = [
                c for c in contract.constraints.hard_constraints
                if c.value != v
            ]

        # Add preferences
        for p in update.preferences_to_add:
            contract.constraints.soft_preferences.append(p)

        # Remove preferences
        for v in update.preferences_to_remove:
            contract.constraints.soft_preferences = [
                p for p in contract.constraints.soft_preferences
                if p.value != v
            ]

        # Geography updates — acceptable hubs: append, avoid airports: append
        geo_updates = update.field_updates.get("geography", {})
        for key in ("acceptable_origin_hubs", "acceptable_destination_hubs"):
            for code in geo_updates.get(key, []):
                existing = getattr(contract.geography, key)
                if code not in existing:
                    existing.append(code)

        # Avoid airports: add and remove from acceptable
        for code in geo_updates.get("avoid_airports", []):
            if code not in contract.geography.avoid_airports:
                contract.geography.avoid_airports.append(code)
            for hub_key in ("acceptable_origin_hubs", "acceptable_destination_hubs",
                            "acceptable_transfer_hubs"):
                hubs = getattr(contract.geography, hub_key)
                if code in hubs:
                    hubs.remove(code)

        # Avoid cities
        if "avoid_cities" in geo_updates:
            for city in geo_updates["avoid_cities"]:
                if city not in contract.geography.avoid_cities:
                    contract.geography.avoid_cities.append(city)

        # Ranking
        rank_updates = update.field_updates.get("ranking_preferences", {})
        for k, v in rank_updates.items():
            if hasattr(contract.ranking_preferences, k):
                setattr(contract.ranking_preferences, k, v)

        # Risk
        risk_updates = update.field_updates.get("risk_preferences", {})
        for k, v in risk_updates.items():
            if hasattr(contract.risk_preferences, k):
                setattr(contract.risk_preferences, k, v)

        # Clarification questions
        contract.unresolved_questions = update.clarification_questions

        # Ready to search
        if update.should_search or update.should_rerun_search:
            contract.ready_to_search = contract.has_required_route()

        # Confidence
        if update.confidence > 0:
            contract.contract_confidence = update.confidence

        # Metadata
        contract.conversation_metadata.update_count += 1
        contract.conversation_metadata.last_update_summary_zh = update.reasoning_summary

        # Normalize to enforce schema consistency
        contract.normalize()

        return contract
