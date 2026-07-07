"""Constraint checks that preserve live/mock/estimate boundaries."""

from __future__ import annotations

from datetime import datetime

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.pipeline.types import PipelineResult
from travel_agent.planning.models import ConstraintCheckResult, ConstraintFinding, CostEstimate
from travel_agent.tools.base import ToolResult


class ConstraintChecker:
    def check(
        self,
        contract: TravelRequirementContract,
        *,
        cost_estimate: CostEstimate | None = None,
        weather_result: ToolResult | None = None,
        pipeline_result: PipelineResult | None = None,
    ) -> ConstraintCheckResult:
        findings: list[ConstraintFinding] = []

        if _active_pet(contract):
            findings.append(
                ConstraintFinding(
                    category="pet",
                    level="warning",
                    message="宠物同行需分别确认航空公司、酒店、当地交通和活动场所的官方政策与名额。",
                )
            )

        if contract.preferences.avoid_red_eye:
            red_eye = _has_red_eye(pipeline_result)
            if red_eye is True:
                findings.append(
                    ConstraintFinding(
                        category="red_eye",
                        level="conflict",
                        message="demo/mock 航班中检测到深夜或凌晨出发时段，可能与避开红眼航班偏好冲突。",
                        evidence_type="mock_demo",
                    )
                )
            else:
                findings.append(
                    ConstraintFinding(
                        category="red_eye",
                        level="info",
                        message="已记录避开红眼航班偏好；没有真实时刻表时无法确认具体航班是否满足。",
                        evidence_type="mock_demo" if red_eye is False else "unknown",
                    )
                )

        if contract.preferences.nonstop_preferred:
            stops = _mock_stop_count(pipeline_result)
            if stops is not None and stops > 0:
                findings.append(
                    ConstraintFinding(
                        category="nonstop",
                        level="conflict",
                        message=f"当前 demo/mock 候选含 {stops} 次中转，可能不符合直飞偏好。",
                        evidence_type="mock_demo",
                    )
                )
            else:
                findings.append(
                    ConstraintFinding(
                        category="nonstop",
                        level="info",
                        message="已记录直飞优先；最终仍需用真实航班库存确认。",
                        evidence_type="mock_demo" if stops == 0 else "unknown",
                    )
                )

        budget_finding = _budget_finding(contract, cost_estimate)
        if budget_finding:
            findings.append(budget_finding)
        findings.extend(_weather_findings(weather_result))

        if pipeline_result and pipeline_result.recommendations:
            route = pipeline_result.recommendations[0].itinerary.route
            if len(route) > 2:
                findings.append(
                    ConstraintFinding(
                        category="airport_transfer",
                        level="warning",
                        message="demo/mock 路线包含中转；需为换航站楼、行李和重新安检预留时间。",
                        evidence_type="mock_demo",
                    )
                )

        active_categories = {item.category for item in contract.special_requirements if item.active}
        if "visa_constraint" in active_categories or _cross_border(contract):
            findings.append(
                ConstraintFinding(
                    category="documents",
                    level="warning",
                    message="跨境旅行请以政府、使领馆和承运方的官方签证、证件及入境政策为准。",
                )
            )
        if "accessibility" in active_categories:
            findings.append(
                ConstraintFinding(
                    category="accessibility",
                    level="warning",
                    message="请提前向交通、酒店和活动场所确认无障碍设施及协助服务。",
                )
            )
        return ConstraintCheckResult(findings=_dedupe_findings(findings))


def _active_pet(contract: TravelRequirementContract) -> bool:
    return any(pet.active for pet in contract.companions.pets) or any(
        item.active and item.category == "pet_travel" for item in contract.special_requirements
    )


def _has_red_eye(result: PipelineResult | None) -> bool | None:
    if not result or not result.recommendations:
        return None
    found_time = False
    for segment in result.recommendations[0].itinerary.segments:
        value = segment.departure_time
        if not value:
            continue
        found_time = True
        hour = value.hour if isinstance(value, datetime) else datetime.fromisoformat(str(value)).hour
        if hour < 6 or hour >= 23:
            return True
    return False if found_time else None


def _mock_stop_count(result: PipelineResult | None) -> int | None:
    if not result or not result.recommendations:
        return None
    route = result.recommendations[0].itinerary.route
    return max(0, len(route) - 2)


def _budget_finding(
    contract: TravelRequirementContract,
    estimate: CostEstimate | None,
) -> ConstraintFinding | None:
    if contract.budget.amount is None:
        return None
    if not estimate or estimate.total_max is None:
        return ConstraintFinding(
            category="budget",
            level="info",
            message="已记录预算，但当前信息不足，暂时无法判断是否超支。",
        )
    budget_currency = (contract.budget.currency or estimate.currency).upper()
    if budget_currency != estimate.currency.upper():
        return ConstraintFinding(
            category="budget",
            level="info",
            message="预算与估算币种不同，缺少可靠汇率时不做超支判断。",
        )
    if estimate.total_min > contract.budget.amount:
        return ConstraintFinding(
            category="budget",
            level="conflict",
            message="粗略估算下限已经高于当前预算，预算可能明显偏紧。",
            evidence_type="estimate",
        )
    if estimate.total_max > contract.budget.amount:
        return ConstraintFinding(
            category="budget",
            level="warning",
            message="粗略估算上限超过当前预算，建议保留缓冲或减少付费项目。",
            evidence_type="estimate",
        )
    return ConstraintFinding(
        category="budget",
        level="info",
        message="当前粗略区间未超过已记录预算，但实际价格仍需实时核实。",
        evidence_type="estimate",
    )


def _weather_findings(result: ToolResult | None) -> list[ConstraintFinding]:
    if not result or result.status != "ok" or not result.data:
        return []
    rows = result.data.get("daily")
    if not isinstance(rows, list):
        return []
    findings: list[ConstraintFinding] = []
    if any((row.get("precipitation_probability_max") or 0) >= 50 for row in rows):
        findings.append(
            ConstraintFinding(
                category="weather",
                level="warning",
                message="天气预报显示部分日期降水概率较高，建议准备室内替代活动。",
                evidence_type="live",
            )
        )
    if any((row.get("temperature_max") or -999) >= 35 for row in rows):
        findings.append(
            ConstraintFinding(
                category="weather",
                level="warning",
                message="天气预报显示可能高温，建议减少正午户外活动并注意补水。",
                evidence_type="live",
            )
        )
    if any((row.get("temperature_min") or 999) <= 0 for row in rows):
        findings.append(
            ConstraintFinding(
                category="weather",
                level="warning",
                message="天气预报显示可能低温或结冰，请准备保暖并关注交通。",
                evidence_type="live",
            )
        )
    return findings


def _cross_border(contract: TravelRequirementContract) -> bool:
    origin = (contract.trip.origin_airport or "").upper()
    destination = (contract.trip.destination_airport or "").upper()
    china = {"TFU", "CTU", "WNZ", "NGB", "PVG", "SHA", "HGH", "PEK"}
    us = {"AUS", "PIT", "MIA", "JFK", "EWR", "LGA", "DFW", "LAX", "SFO"}
    return bool((origin in china and destination in us) or (origin in us and destination in china))


def _dedupe_findings(items: list[ConstraintFinding]) -> list[ConstraintFinding]:
    seen: set[tuple[str, str]] = set()
    result: list[ConstraintFinding] = []
    for item in items:
        key = (item.category, item.message)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
