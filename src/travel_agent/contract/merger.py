"""Merge validated LLM updates into the travel requirement contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ValidationError

from travel_agent.contract.models import (
    ConstraintItem,
    PreferenceItem,
    SpecialRequirement,
    TravelRequirementContract,
)
from travel_agent.contract.normalization import expand_location, normalize_airport_list
from travel_agent.contract.special_requirements import SpecialRequirementInterpreter
from travel_agent.llm.schemas import TravelRequirementContractUpdate


LIST_APPEND_PATHS = {
    ("geography", "acceptable_origin_hubs"),
    ("geography", "acceptable_transfer_hubs"),
    ("geography", "acceptable_destination_hubs"),
    ("geography", "preferred_airports"),
    ("geography", "avoid_airports"),
    ("geography", "avoid_cities"),
    ("geography", "unresolved_locations"),
    ("airline_preferences", "preferred_airlines"),
    ("airline_preferences", "avoid_airlines"),
    ("constraints", "hard_constraints"),
    ("constraints", "soft_preferences"),
    ("special_requirements",),
}

PROTECTED_MODIFY_PATHS = {
    ("trip", "origin_text"),
    ("trip", "origin_airport"),
    ("trip", "origin_city"),
    ("trip", "destination_text"),
    ("trip", "destination_airport"),
    ("trip", "destination_city"),
}


class ContractMerger:
    """Applies strict schema updates without letting old state leak across searches."""

    def __init__(self, special_interpreter: SpecialRequirementInterpreter | None = None):
        self.special_interpreter = special_interpreter or SpecialRequirementInterpreter()
        self.diagnostics: list[str] = []

    def apply(
        self,
        contract: TravelRequirementContract | None,
        update: TravelRequirementContractUpdate,
        user_message: str | None = None,
    ) -> TravelRequirementContract:
        update_type = update.update_type
        if update_type == "create_new":
            next_contract = TravelRequirementContract()
            if user_message:
                next_contract.metadata.original_user_goal = user_message
        else:
            next_contract = contract.model_copy(deep=True) if contract else TravelRequirementContract()

        next_contract.metadata.last_user_message = user_message
        next_contract.metadata.update_count += 1

        self.diagnostics = []
        update = self._coerce_update(update)
        updates = _expand_dotted_keys(deepcopy(update.field_updates))
        self._apply_nested_updates(next_contract, updates, update_type)

        next_contract.constraints.hard_constraints.extend(
            coerce_model_list(update.constraints_to_add, ConstraintItem, "constraints_to_add", self.diagnostics)
        )
        next_contract.constraints.soft_preferences.extend(
            coerce_model_list(update.preferences_to_add, PreferenceItem, "preferences_to_add", self.diagnostics)
        )
        next_contract.special_requirements.extend(
            coerce_model_list(
                update.special_requirements_to_add,
                SpecialRequirement,
                "special_requirements_to_add",
                self.diagnostics,
            )
        )

        self._remove_constraints(next_contract, update.constraints_to_remove)
        self._remove_preferences(next_contract, update.preferences_to_remove)
        self._remove_special_requirements(next_contract, update.special_requirements_to_remove)
        self._coerce_contract_lists(next_contract)
        self._sync_constraints_to_fields(next_contract)
        self._sync_preferences_to_fields(next_contract, update.preferences_to_add)
        self._apply_special_requirement_effects(next_contract)
        self._handle_explicit_reallows(next_contract, update)

        next_contract.confidence = update.confidence
        return next_contract.normalize()

    def _coerce_update(
        self,
        update: TravelRequirementContractUpdate,
    ) -> TravelRequirementContractUpdate:
        update.constraints_to_add = coerce_model_list(
            update.constraints_to_add, ConstraintItem, "constraints_to_add", self.diagnostics
        )
        update.preferences_to_add = coerce_model_list(
            update.preferences_to_add, PreferenceItem, "preferences_to_add", self.diagnostics
        )
        update.special_requirements_to_add = coerce_model_list(
            update.special_requirements_to_add,
            SpecialRequirement,
            "special_requirements_to_add",
            self.diagnostics,
        )
        return update

    def _apply_nested_updates(
        self,
        contract: TravelRequirementContract,
        updates: dict[str, Any],
        update_type: str,
        prefix: tuple[str, ...] = (),
    ) -> None:
        for key, value in updates.items():
            path = (*prefix, key)
            target = _get_attr_path(contract, prefix) if prefix else contract
            if isinstance(value, dict) and hasattr(target, key):
                self._apply_nested_updates(contract, value, update_type, path)
                continue
            if update_type != "create_new" and path in PROTECTED_MODIFY_PATHS:
                continue
            self._set_path_value(contract, path, value)

    def _set_path_value(self, contract: TravelRequirementContract, path: tuple[str, ...], value: Any) -> None:
        parent = _get_attr_path(contract, path[:-1])
        attr = path[-1]
        if not hasattr(parent, attr):
            return
        current = getattr(parent, attr)
        if path in LIST_APPEND_PATHS:
            incoming = value if isinstance(value, list) else [value]
            setattr(parent, attr, [*list(current or []), *incoming])
        else:
            setattr(parent, attr, value)

    def _remove_constraints(self, contract: TravelRequirementContract, removals: list[str]) -> None:
        if not removals:
            return
        removal_set = {str(item).lower() for item in removals}
        kept: list[ConstraintItem] = []
        for item in contract.constraints.hard_constraints:
            values = {str(item.value).lower(), item.type.lower(), item.reason.lower()}
            values.update(str(v).lower() for v in item.normalized_values)
            if values & removal_set:
                continue
            kept.append(item)
        contract.constraints.hard_constraints = kept

    def _remove_preferences(self, contract: TravelRequirementContract, removals: list[str]) -> None:
        if not removals:
            return
        removal_set = {str(item).lower() for item in removals}
        kept: list[PreferenceItem] = []
        for item in contract.constraints.soft_preferences:
            values = {str(item.value).lower(), item.type.lower()}
            values.update(str(v).lower() for v in item.normalized_values)
            if values & removal_set:
                continue
            kept.append(item)
        contract.constraints.soft_preferences = kept

    def _remove_special_requirements(
        self,
        contract: TravelRequirementContract,
        removals: list[str],
    ) -> None:
        if not removals:
            return
        removal_set = {str(item).lower() for item in removals}
        kept: list[SpecialRequirement] = []
        for item in contract.special_requirements:
            values = {
                item.category.lower(),
                item.description_zh.lower(),
                item.source_user_message.lower(),
            }
            values.update(str(v).lower() for v in item.structured_values.values())
            if values & removal_set:
                continue
            kept.append(item)
        contract.special_requirements = kept

    def _coerce_contract_lists(self, contract: TravelRequirementContract) -> None:
        contract.constraints.hard_constraints = coerce_model_list(
            contract.constraints.hard_constraints,
            ConstraintItem,
            "contract.constraints.hard_constraints",
            self.diagnostics,
        )
        contract.constraints.soft_preferences = coerce_model_list(
            contract.constraints.soft_preferences,
            PreferenceItem,
            "contract.constraints.soft_preferences",
            self.diagnostics,
        )
        contract.special_requirements = coerce_model_list(
            contract.special_requirements,
            SpecialRequirement,
            "contract.special_requirements",
            self.diagnostics,
        )

    def _sync_constraints_to_fields(self, contract: TravelRequirementContract) -> None:
        for item in contract.constraints.hard_constraints:
            if not item.active:
                continue
            if item.type == "avoid_airport":
                contract.geography.avoid_airports.extend(item.normalized_values or [str(item.value)])
            elif item.type == "avoid_city":
                contract.geography.avoid_cities.append(str(item.value))
                contract.geography.avoid_airports.extend(item.normalized_values or expand_location(str(item.value)))
            elif item.type == "avoid_airline":
                contract.airline_preferences.avoid_airlines.extend(
                    [str(v).upper() for v in (item.normalized_values or [item.value]) if v]
                )
            elif item.type == "no_split_ticket":
                contract.ticketing.split_ticket_policy = "avoid"
                contract.ticketing.allow_self_transfer = False

    def _sync_preferences_to_fields(
        self,
        contract: TravelRequirementContract,
        preferences: list[PreferenceItem],
    ) -> None:
        for item in preferences:
            if not item.active:
                continue
            if item.type == "prefer_hub":
                contract.geography.preferred_airports.extend(item.normalized_values or [str(item.value)])
            elif item.type == "prefer_airline":
                contract.airline_preferences.preferred_airlines.extend(
                    [str(v).upper() for v in (item.normalized_values or [item.value]) if v]
                )
            elif item.type == "prefer_low_price":
                contract.ranking.profile = "cheapest"
                contract.ranking.price_priority = item.weight_hint
                contract.hub_policy.nearby_hub_policy = "prefer"
            elif item.type == "prefer_low_risk":
                contract.ranking.profile = "low_risk"
                contract.ranking.risk_priority = item.weight_hint
            elif item.type == "prefer_short_time":
                contract.ranking.profile = "fastest"
                contract.ranking.time_priority = item.weight_hint

    def _apply_special_requirement_effects(self, contract: TravelRequirementContract) -> None:
        effects = self.special_interpreter.interpret(contract.special_requirements)
        if effects.avoid_self_transfer:
            contract.ticketing.split_ticket_policy = "avoid"
            contract.ticketing.allow_self_transfer = False
        if effects.avoid_complex_transfers:
            contract.hub_policy.avoid_complex_transfers = True
        if effects.risk_weight_adjustment > 0:
            contract.ranking.risk_priority = "high"
        if effects.prefer_full_service_airlines or effects.airline_quality_weight_adjustment > 0:
            contract.airline_preferences.prefer_major_airlines = True

    def _handle_explicit_reallows(
        self,
        contract: TravelRequirementContract,
        update: TravelRequirementContractUpdate,
    ) -> None:
        updates = _expand_dotted_keys(deepcopy(update.field_updates))
        geography = updates.get("geography", {}) if isinstance(updates.get("geography"), dict) else {}
        allowed = normalize_airport_list(
            [
                *geography.get("acceptable_origin_hubs", []),
                *geography.get("acceptable_transfer_hubs", []),
                *geography.get("acceptable_destination_hubs", []),
                *geography.get("preferred_airports", []),
            ]
        )
        allowed.extend(
            normalize_airport_list(
                [
                    *[v for p in update.preferences_to_add for v in (p.normalized_values or [p.value])],
                ]
            )
        )
        if not allowed:
            return
        allowed_set = set(allowed)
        existing_excluded = set(normalize_airport_list(contract.geography.avoid_airports))
        contract.geography.avoid_airports = [code for code in contract.geography.avoid_airports if code not in allowed_set]

        new_avoid_cities: list[str] = []
        for city in contract.geography.avoid_cities:
            expanded = set(expand_location(city))
            if expanded & allowed_set:
                remaining = sorted((expanded - allowed_set) & existing_excluded)
                contract.geography.avoid_airports.extend(remaining)
            else:
                new_avoid_cities.append(city)
        contract.geography.avoid_cities = new_avoid_cities

        kept: list[ConstraintItem] = []
        for item in contract.constraints.hard_constraints:
            item_values = set(normalize_airport_list(item.normalized_values or [str(item.value)]))
            if item.type in {"avoid_airport", "avoid_city"} and item_values & allowed_set:
                remaining = sorted(item_values - allowed_set)
                if remaining:
                    kept.append(
                        ConstraintItem(
                            type="avoid_airport",
                            value=",".join(remaining),
                            normalized_values=remaining,
                            reason=item.reason,
                            source_user_message=item.source_user_message,
                            active=item.active,
                        )
                    )
                continue
            kept.append(item)
        contract.constraints.hard_constraints = kept


def _get_attr_path(obj: Any, path: tuple[str, ...]) -> Any:
    target = obj
    for part in path:
        target = getattr(target, part)
    return target


def _expand_dotted_keys(updates: dict[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in (updates or {}).items():
        if "." not in key:
            if isinstance(value, dict):
                value = _expand_dotted_keys(value)
            expanded[key] = value
            continue
        cursor = expanded
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return expanded


def coerce_model_list(
    items: list[Any],
    model_cls: type[BaseModel],
    context_name: str,
    diagnostics: list[str] | None = None,
) -> list[Any]:
    result: list[Any] = []
    for idx, item in enumerate(items or []):
        if isinstance(item, model_cls):
            result.append(item)
            continue
        if isinstance(item, dict):
            try:
                result.append(model_cls.model_validate(item))
            except ValidationError as exc:
                if diagnostics is not None:
                    diagnostics.append(f"{context_name}[{idx}] skipped: {exc.errors()[0].get('msg', 'invalid')}")
            continue
        if diagnostics is not None:
            diagnostics.append(f"{context_name}[{idx}] skipped: expected dict or {model_cls.__name__}")
    return result
