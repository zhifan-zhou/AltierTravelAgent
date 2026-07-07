"""Honest rough trip-cost estimator."""

from __future__ import annotations

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.pipeline.types import PipelineResult
from travel_agent.planning.models import CostEstimate, CostItem, SourceRef
from travel_agent.tools.base import ToolResult


class CostEstimator:
    def estimate(
        self,
        contract: TravelRequirementContract,
        *,
        duration_days: int | None = None,
        pipeline_result: PipelineResult | None = None,
        conversion_result: ToolResult | None = None,
    ) -> CostEstimate:
        days = min(max(int(duration_days or contract.time.duration_days or 3), 1), 30)
        low_budget = contract.budget.preference == "lower" or contract.ranking.profile == "cheapest"
        items: list[CostItem] = []
        warnings = ["以下住宿、餐饮、当地交通和活动费用均为粗略估算，不是实时报价。"]
        sources: list[SourceRef] = []

        flight_price = _mock_flight_price(pipeline_result)
        if flight_price is None:
            items.append(
                CostItem(
                    category="航班",
                    currency="USD",
                    source_type="unknown",
                    note="没有可用航班金额；未将机票计入总计。",
                )
            )
        else:
            items.append(
                CostItem(
                    category="航班",
                    amount_min=flight_price,
                    amount_max=flight_price,
                    currency="USD",
                    confidence="low",
                    source_type="mock_demo",
                    note="demo/mock only，不代表真实价格、库存或可预订结果。",
                )
            )
            warnings.append("航班金额来自 demo/mock 数据，不能用于购买决策。")

        lodging = (60, 120) if low_budget else (100, 220)
        food = (25, 45) if low_budget else (45, 90)
        transport = (8, 20) if low_budget else (20, 50)
        activities = (0, 25) if low_budget else (20, 70)
        for category, per_day, note in [
            ("住宿", lodging, "按每晚粗略区间估算，税费和旺季波动未单独计算。"),
            ("餐饮", food, "按每日粗略区间估算。"),
            ("当地交通", transport, "按步行/公共交通或少量短途出行的粗略区间估算。"),
            ("活动", activities, "不含已核实的实时门票；付费项目需查官网。"),
        ]:
            items.append(
                CostItem(
                    category=category,
                    amount_min=per_day[0] * days,
                    amount_max=per_day[1] * days,
                    currency="USD",
                    confidence="low",
                    source_type="estimate",
                    note=note,
                )
            )

        total_min = sum(item.amount_min or 0 for item in items)
        total_max = sum(item.amount_max or 0 for item in items)
        currency = "USD"
        target = (contract.budget.currency or "USD").upper()
        if target != "USD" and conversion_result and conversion_result.status == "ok" and conversion_result.data:
            rate = conversion_result.data.get("rate")
            if isinstance(rate, (int, float)) and rate > 0:
                for item in items:
                    if item.amount_min is not None:
                        item.amount_min = round(item.amount_min * rate, 2)
                    if item.amount_max is not None:
                        item.amount_max = round(item.amount_max * rate, 2)
                    item.currency = target
                total_min = round(total_min * rate, 2)
                total_max = round(total_max * rate, 2)
                currency = target
                sources.append(SourceRef(label="汇率换算", source=conversion_result.source or "currency", is_live=True))
        elif target != "USD":
            warnings.append(f"无法取得可靠的 USD→{target} 汇率，本次保留 USD 估算。")

        assumptions = [f"按 {days} 天估算。", "住宿按每晚一间基础房型的宽区间估算。"]
        return CostEstimate(
            items=items,
            total_min=round(total_min, 2),
            total_max=round(total_max, 2),
            currency=currency,
            assumptions=assumptions,
            warnings=list(dict.fromkeys(warnings)),
            sources=sources,
        )


def _mock_flight_price(result: PipelineResult | None) -> float | None:
    if not result:
        return None
    if result.recommendations:
        return float(result.recommendations[0].itinerary.total_price_usd)
    prices = [offer.total_price_usd for offer in result.offers if offer.total_price_usd > 0]
    return min(prices) if prices else None
