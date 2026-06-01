"""ConversationRequirementAgent — builds/updates TravelRequirementContract from chat.

Primary LLM-facing agent. Deterministic when LLM disabled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from travel_agent.agents.base import BaseAgent
from travel_agent.models.requirement_contract import TravelRequirementContract
from travel_agent.llm.prompts import get_llm_client, is_llm_enabled
from travel_agent.services.airport_service import AirportService
from travel_agent.services.location_normalization_service import LocationNormalizationService


class RequirementUpdateResult(BaseModel):
    updated_contract: TravelRequirementContract = Field(default_factory=TravelRequirementContract)
    action_type: str = "update_contract"
    changed_fields: list[str] = Field(default_factory=list)
    clarification_question_zh: str | None = None
    user_facing_summary_zh: str = ""
    confidence: float = 0.5


class ConversationRequirementAgent(BaseAgent[tuple[str, TravelRequirementContract | None, dict], RequirementUpdateResult]):
    """Build and update a TravelRequirementContract from conversation."""

    name = "conversation_requirement"

    def __init__(self, airport_service: AirportService | None = None):
        super().__init__()
        self._airports = airport_service or AirportService()
        self._locations = LocationNormalizationService()

    async def execute(
        self, data: tuple[str, TravelRequirementContract | None, dict]
    ) -> RequirementUpdateResult:
        message, current_contract, context = data
        msg = message.strip()

        # Start fresh contract if none exists
        contract = current_contract or TravelRequirementContract()
        contract.conversation_metadata.last_user_message = msg

        # Try LLM if enabled
        if is_llm_enabled("clarification"):
            llm = get_llm_client()
            try:
                result = await llm.interpret_followup(message, {
                    "current_contract": contract.model_dump(mode="json") if current_contract else None,
                    "context": context,
                })
                if result and result.confidence > 0.4:
                    return self._apply_llm_result(result, contract, msg)
            except Exception:
                pass

        # Deterministic update
        return self._deterministic_update(msg, contract)

    def _apply_llm_result(self, result, contract, msg) -> RequirementUpdateResult:
        updates = result.constraint_updates
        changed = []

        # Apply avoid hubs
        avoid = updates.get("avoid_hubs", [])
        if avoid:
            for a in avoid:
                if a not in contract.geography.avoid_airports:
                    contract.geography.avoid_airports.append(a)
            for hub_key in ("acceptable_origin_hubs", "acceptable_destination_hubs"):
                hubs = getattr(contract.geography, hub_key)
                for a in avoid:
                    if a in hubs:
                        hubs.remove(a)
            changed.append("avoid_airports")

        # Apply acceptable hubs
        hubs = updates.get("acceptable_origin_hubs", [])
        if hubs:
            for h in hubs:
                if h not in contract.geography.acceptable_origin_hubs:
                    contract.geography.acceptable_origin_hubs.append(h)
            changed.append("acceptable_origin_hubs")

        # Profile
        profile = updates.get("scoring_profile")
        if profile:
            contract.ranking_preferences.profile = profile
            changed.append("profile")

        # Ready to search
        if contract.has_required_route():
            contract.ready_to_search = True

        contract.normalize()

        return RequirementUpdateResult(
            updated_contract=contract,
            action_type="update_contract" if changed else "ready_to_search",
            changed_fields=changed,
            user_facing_summary_zh=result.user_message_summary or msg,
            confidence=result.confidence,
        )

    def _deterministic_update(self, msg: str, contract: TravelRequirementContract) -> RequirementUpdateResult:
        changed = []

        # Detect origin/destination
        if not contract.trip.primary_origin_text:
            intake = self._extract_route_from_msg(msg)
            if intake.get("origin"):
                contract.trip.primary_origin_text = intake["origin"]
                contract.trip.primary_origin_airport = self._airports.resolve_airport_code(intake["origin"])
                changed.append("origin")

        if not contract.trip.primary_destination_text:
            intake = self._extract_route_from_msg(msg)
            if intake.get("destination"):
                contract.trip.primary_destination_text = intake["destination"]
                contract.trip.primary_destination_airport = self._airports.resolve_airport_code(intake["destination"])
                changed.append("destination")

        # Location disambiguation: 浦东→PVG, 虹桥→SHA, 上海→both, etc.
        loc = self._locations.disambiguate(msg)

        # Apply exclusions (specific airports like SHA detected by disambiguate)
        for code in loc["exclude"]:
            if code not in contract.geography.avoid_airports:
                contract.geography.avoid_airports.append(code)
            # Remove from acceptable hubs
            for hub_key in ("acceptable_origin_hubs", "acceptable_destination_hubs", "acceptable_transfer_hubs"):
                hubs = getattr(contract.geography, hub_key)
                if code in hubs:
                    hubs.remove(code)
            changed.append("avoid_airports")

        # Apply allows (specific airports like PVG detected by disambiguate)
        for code in loc["allow"]:
            if code not in contract.geography.acceptable_origin_hubs:
                contract.geography.acceptable_origin_hubs.append(code)
            # If this code was previously excluded, remove from avoid (user is re-allowing it)
            if code in contract.geography.avoid_airports:
                contract.geography.avoid_airports.remove(code)
            changed.append("acceptable_origin_hubs")

        # Legacy "不想/不要X转" detection for city-level avoids
        import re
        avoid_city_match = re.findall(r"(?:不想|不要|别)\s*[从去]?\s*(\S{1,4})\s*(?:转|走|出发)?", msg)
        for city_name in avoid_city_match:
            if city_name in ("浦东", "虹桥", "PVG", "SHA", "pvg", "sha"):
                continue  # Already handled by disambiguate
            codes = self._airports.resolve_city_group(city_name)
            if codes and len(codes) <= 4:  # City-level group
                for c in codes:
                    if c not in contract.geography.avoid_airports and c not in loc["allow"]:
                        contract.geography.avoid_airports.append(c)
                changed.append("avoid_airports")

        # Legacy "可以从X走" -> add to acceptable origin hubs
        hub_match = re.findall(r"可以从\s*(\S{1,6})\s*(?:走|飞|出发|转)", msg)
        for city in hub_match:
            if city in ("浦东", "虹桥", "PVG", "SHA"):
                continue  # Already handled by disambiguate
            code = self._airports.resolve_airport_code(city)
            if code and code not in contract.geography.acceptable_origin_hubs:
                contract.geography.acceptable_origin_hubs.append(code)
                changed.append("acceptable_origin_hubs")

        # Profile keywords
        from travel_agent.models.preference import PREFERENCE_COMMANDS
        for kw, profile in PREFERENCE_COMMANDS.items():
            if kw in msg:
                contract.ranking_preferences.profile = profile
                changed.append("profile")
                break

        # "折腾" / "爸妈" -> low risk
        if any(w in msg for w in ["折腾", "爸妈", "父母", "老人"]):
            contract.risk_preferences.risk_tolerance = "low"
            contract.risk_preferences.family_friendly = True
            contract.ticketing_preferences.split_ticket_policy = "avoid"
            changed.append("risk_preferences")

        if contract.has_required_route():
            contract.ready_to_search = True

        contract.normalize()

        return RequirementUpdateResult(
            updated_contract=contract,
            action_type="ready_to_search" if contract.ready_to_search else "update_contract",
            changed_fields=changed,
            user_facing_summary_zh=msg,
            confidence=0.8 if changed else 0.3,
        )

    def _extract_route_from_msg(self, msg: str) -> dict:
        import re
        result = {}
        m = re.search(r"(\S{1,12})\s*(?:到|飞|去)\s*(\S{1,12})", msg)
        if m:
            result["origin"] = m.group(1).strip().rstrip("，。,")
            result["destination"] = m.group(2).strip().rstrip("，。,")
        return result
