"""Deterministic, source-aware day-by-day itinerary drafts."""

from __future__ import annotations

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.planning.models import ItineraryDay, ItineraryPlan, SourceRef
from travel_agent.tools.base import ToolResult


class ItineraryBuilder:
    """Build practical drafts without inventing tickets, hours, or policies."""

    def build(
        self,
        contract: TravelRequirementContract,
        *,
        duration_days: int | None = None,
        weather_result: ToolResult | None = None,
        destination_brief_result: ToolResult | None = None,
    ) -> ItineraryPlan:
        destination = _destination_label(contract)
        days = duration_days or contract.time.duration_days or 3
        days = min(max(int(days), 1), 5)
        budget_level = _budget_level(contract)
        assumptions: list[str] = []
        if duration_days is None and contract.time.duration_days is None:
            assumptions.append("未指定行程长度，先按 3 天草案安排。")
        assumptions.append("活动顺序是规划建议，不包含实时门票、营业时间或预订。")

        weather_days = _weather_days(weather_result)
        sources: list[SourceRef] = []
        warnings: list[str] = []
        if weather_result and weather_result.status == "ok" and weather_result.source:
            sources.append(SourceRef(label="天气预报", source=weather_result.source, is_live=weather_result.is_live))
        elif weather_result and weather_result.status != "ok":
            warnings.append("实时天气暂时不可用，行程没有加入具体天气判断。")
        if destination_brief_result and destination_brief_result.status == "ok" and destination_brief_result.source:
            sources.append(
                SourceRef(
                    label="目的地简介",
                    source=destination_brief_result.source,
                    is_live=destination_brief_result.is_live,
                )
            )

        plan_days: list[ItineraryDay] = []
        for index in range(days):
            template = _day_template(index, days, budget_level)
            considerations = _weather_considerations(weather_days, index)
            notes = list(template["notes"])
            if _active_pet(contract):
                notes.append("宠物同行：出发前逐项确认住宿、交通和活动场所的官方宠物政策。")
            if _accessibility_needed(contract):
                notes.append("无障碍需求：提前向交通、住宿和活动场所确认可达性与协助方式。")
            plan_days.append(
                ItineraryDay(
                    day=index + 1,
                    title=template["title"],
                    morning=template["morning"],
                    afternoon=template["afternoon"],
                    evening=template["evening"],
                    notes=notes,
                    weather_considerations=considerations,
                    budget_level=budget_level,
                )
            )

        if _active_pet(contract):
            warnings.append("宠物政策以航空公司、酒店、当地交通和活动场所官方说明为准。")
        warnings.append("航班价格仍为 demo/mock，不代表真实报价、库存或可预订结果。")
        return ItineraryPlan(
            destination=destination,
            duration_days=days,
            days=plan_days,
            assumptions=assumptions,
            sources=sources,
            warnings=_dedupe(warnings),
        )


def _destination_label(contract: TravelRequirementContract) -> str:
    return (
        contract.trip.destination_text
        or contract.trip.destination_city
        or contract.trip.destination_airport
        or "未指定目的地"
    )


def _budget_level(contract: TravelRequirementContract) -> str:
    if contract.budget.preference == "lower" or contract.ranking.profile == "cheapest":
        return "low"
    if contract.budget.amount is not None:
        return "medium"
    return "unknown"


def _active_pet(contract: TravelRequirementContract) -> bool:
    return any(pet.active for pet in contract.companions.pets) or any(
        item.active and item.category == "pet_travel" for item in contract.special_requirements
    )


def _accessibility_needed(contract: TravelRequirementContract) -> bool:
    return any(item.active and item.category == "accessibility" for item in contract.special_requirements)


def _weather_days(result: ToolResult | None) -> list[dict]:
    if not result or result.status != "ok" or not result.data:
        return []
    daily = result.data.get("daily")
    return daily if isinstance(daily, list) else []


def _weather_considerations(rows: list[dict], index: int) -> list[str]:
    if index >= len(rows):
        return []
    row = rows[index]
    notes: list[str] = []
    summary = str(row.get("summary") or "").strip()
    precipitation = row.get("precipitation_probability_max")
    high = row.get("temperature_max")
    low = row.get("temperature_min")
    if summary:
        notes.append(f"预报摘要：{summary}。")
    if isinstance(precipitation, (int, float)) and precipitation >= 50:
        notes.append("降水概率较高，优先安排可替换的室内活动并携带雨具。")
    if isinstance(high, (int, float)) and high >= 35:
        notes.append("可能高温，减少正午户外停留并注意补水。")
    if isinstance(low, (int, float)) and low <= 0:
        notes.append("可能低温或结冰，准备保暖并留意交通状况。")
    return notes


def _day_template(index: int, duration: int, budget_level: str) -> dict[str, list[str] | str]:
    low_cost = "优先步行、公共交通和免费公共空间" if budget_level == "low" else "按体力选择步行或公共交通"
    templates = [
        {
            "title": "抵达与轻量适应",
            "morning": ["抵达后先处理入住或行李寄存，预留交通缓冲。"],
            "afternoon": [f"在住宿附近做轻量城市漫步；{low_cost}。"],
            "evening": ["选择离住宿较近的用餐区域，避免第一天排得过满。"],
            "notes": ["具体入住时间、交通班次和场所开放情况需向官方确认。"],
        },
        {
            "title": "城市核心体验",
            "morning": ["选择一个城市核心片区，集中安排步行可达的公共空间与文化体验。"],
            "afternoon": ["继续同一区域，保留室内活动作为天气或体力变化的替代方案。"],
            "evening": ["安排当地餐饮或社区文化体验，控制跨区交通。"],
            "notes": ["门票、预约和营业时间不在本草案中，出发前查看场所官网。"],
        },
        {
            "title": "社区与户外节奏",
            "morning": ["探索一个不同社区，以市场、街区或公共空间为主。"],
            "afternoon": ["天气合适时安排公园或河岸等户外活动；否则改为室内文化活动。"],
            "evening": ["留出自由时间，整理照片或补上前两天错过的体验。"],
            "notes": ["避免把跨城或相距较远的活动塞进同一时段。"],
        },
        {
            "title": "深度体验与弹性日",
            "morning": ["从前几天最感兴趣的主题中选一个做深度体验。"],
            "afternoon": ["保留半天机动，用于天气调整、休息或低成本自由探索。"],
            "evening": ["选择交通简单的晚间活动，不安排无法取消的紧凑衔接。"],
            "notes": ["需要预约的活动应以官方确认结果为准。"],
        },
        {
            "title": "收尾与返程缓冲",
            "morning": ["安排轻量活动并完成行李整理。"],
            "afternoon": ["根据离境时间保留充足机场或车站交通缓冲。"],
            "evening": ["如仍在目的地，选择住宿附近的简单活动。"],
            "notes": ["跨境返程请再次核对证件、行李和承运方要求。"],
        },
    ]
    template = dict(templates[min(index, len(templates) - 1)])
    if duration == 1:
        template["title"] = "一日城市概览"
        template["morning"] = ["先在城市核心片区安排一组步行可达的活动。"]
        template["afternoon"] = ["选择一项室内或户外核心体验，并准备天气替代方案。"]
        template["evening"] = ["就近用餐并预留返程交通时间。"]
    return template


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
