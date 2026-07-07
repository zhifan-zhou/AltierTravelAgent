from __future__ import annotations

from travel_agent.contract.models import PetCompanion, TravelRequirementContract
from travel_agent.server.app import contract_summary


def test_contract_summary_is_sanitized_and_user_visible():
    contract = TravelRequirementContract()
    contract.trip.origin_text = "成都"
    contract.trip.origin_airport = "TFU"
    contract.trip.destination_text = "奥斯丁"
    contract.trip.destination_airport = "AUS"
    contract.time.departure_window_text = "六月初"
    contract.time.duration_days = 3
    contract.budget.priority = "low"
    contract.budget.currency = "USD"
    contract.companions.pets.append(PetCompanion(kind="dog", count=1, active=True))
    contract.preferences.avoid_red_eye = True
    contract.preferences.nonstop_preferred = True
    contract.pending.missing_fields = ["exact departure date"]

    summary = contract_summary(contract)
    text = str(summary)

    assert summary["route"] == {"origin": "成都 (TFU)", "destination": "奥斯丁 (AUS)"}
    assert summary["budget"]["level"] == "low"
    assert summary["companions"]["pets"][0]["kind"] == "dog"
    assert summary["preferences"]["avoid_red_eye"] is True
    assert "exact departure date" in summary["missing_fields"]
    assert "TravelRequirementContract(" not in text
    assert "schema_version" not in text
    assert "None" not in text
    assert "debug" not in text.casefold()
