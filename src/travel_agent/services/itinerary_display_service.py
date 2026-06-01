"""Itinerary display formatting — converts structured itineraries to human-readable text."""

from __future__ import annotations

from travel_agent.models.itinerary import Itinerary
from travel_agent.models.flight import FlightOffer
from travel_agent.models.ranking import RankedRecommendation
from travel_agent.models.risk import RiskAssessment
from travel_agent.services.airport_service import AirportService
from travel_agent.services.airline_service import AirlineService
from travel_agent.utils.money import format_usd


class ItineraryDisplayService:
    """Formats itinerary data for user-facing display."""

    def __init__(self, airport_service: AirportService | None = None,
                 airline_service: AirlineService | None = None):
        self._airports = airport_service or AirportService()
        self._airlines = airline_service or AirlineService()

    # Arrow symbols: → for flight, ⇢ for ground/access transfer
    FLIGHT_ARROW = " → "
    GROUND_ARROW = " ⇢ "

    def format_route_codes(self, itinerary: Itinerary) -> str:
        """Format as IATA codes: WNZ ⇢ PVG + PVG → JFK → PIT"""
        codes = []
        has_ground = (itinerary.total_access_cost_usd > 0 and itinerary.origin_airport
                      and itinerary.origin_airport not in {s.origin for s in itinerary.segments})

        if has_ground:
            codes.append(itinerary.origin_airport)
        for seg in itinerary.segments:
            if not codes or codes[-1] != seg.origin:
                codes.append(seg.origin)
            codes.append(seg.destination)
        if not codes:
            return itinerary.origin_airport + " → " + itinerary.destination_airport

        # Build with mixed arrows
        parts = []
        for i in range(len(codes) - 1):
            arrow = self.GROUND_ARROW if (has_ground and i == 0) else self.FLIGHT_ARROW
            parts.append(codes[i] + arrow)
        parts.append(codes[-1])
        return "".join(parts)

    def format_route_human_zh(self, itinerary: Itinerary) -> str:
        """Format as Chinese names with mixed arrows for ground vs flight."""
        names = []
        origin_code = itinerary.origin_airport
        dest_code = itinerary.destination_airport

        has_ground = (itinerary.total_access_cost_usd > 0 and origin_code
                      and origin_code not in {s.origin for s in itinerary.segments})

        if has_ground:
            names.append(self._airport_name(origin_code))

        for seg in itinerary.segments:
            o_name = self._airport_name(seg.origin)
            if not names or names[-1] != o_name:
                names.append(o_name)
            names.append(self._airport_name(seg.destination))

        if not names:
            return f"{self._airport_name(origin_code)} → {self._airport_name(dest_code)}"

        parts = []
        for i in range(len(names) - 1):
            arrow = self.GROUND_ARROW if (has_ground and i == 0) else self.FLIGHT_ARROW
            parts.append(names[i] + arrow)
        parts.append(names[-1])
        return "".join(parts)

    def format_leg_details_zh(self, itinerary: Itinerary) -> list[str]:
        """Format leg-by-leg details for a detailed view."""
        lines = []
        leg_num = 1

        # Origin access
        if itinerary.total_access_cost_usd > 0:
            origin_code = itinerary.origin_airport
            seen = {s.origin for s in itinerary.segments}
            if origin_code not in seen:
                lines.append(
                    f"{leg_num}. 接驳：{self._airport_name(origin_code)} ⇢ "
                    f"{self._airport_name(itinerary.segments[0].origin) if itinerary.segments else ''}"
                    f"\n   方式：地面/高铁/车"
                    f"\n   预计成本：{format_usd(itinerary.total_access_cost_usd)}"
                    f"\n   提醒：此段需自行安排，建议预留充足时间。"
                )
                leg_num += 1

        for seg in itinerary.segments:
            airline_name = self._airlines.get_display_name(seg.airline) if seg.airline else "未知航司"
            source_label = ""
            for offer in itinerary.offers:
                if seg in offer.segments:
                    if offer.source == "mock_fallback":
                        source_label = " [估算demo]"
                    elif offer.source == "mock_exact":
                        source_label = " [demo]"
                    elif offer.is_real:
                        source_label = " [实时]"
                    break

            segment_price = ""
            for offer in itinerary.offers:
                if seg in offer.segments:
                    segment_price = f"\n   价格：{format_usd(offer.total_price_usd)}"
                    break

            airlines = self._airlines.get_airline(seg.airline) if seg.airline else None
            quality_note = ""
            if airlines:
                tier = airlines.get("quality_tier")
                if tier and tier in ("premium", "major", "standard", "budget", "unknown"):
                    quality_note = f" | 品质：{self._airlines.tier_label_zh(tier)}"
                else:
                    # Fallback to score threshold
                    score = self._airlines.score_airline_for_route(seg.airline, is_long_haul=True)
                    label = self._airlines.tier_label_for_score(score)
                    quality_note = f" | 品质：{label}"

            lines.append(
                f"{leg_num}. {self._segment_type_label(seg)}："
                f"{self._airport_name(seg.origin)} {seg.origin} → "
                f"{self._airport_name(seg.destination)} {seg.destination}"
                f"\n   航司：{airline_name}{quality_note}"
                f"\n   舱位：{seg.cabin.value if hasattr(seg.cabin, 'value') else seg.cabin}"
                f"\n   航班号：{seg.flight_number or 'N/A'}"
                f"{segment_price}"
                f"\n   数据来源：{source_label}"
            )
            leg_num += 1

        return lines

    def format_airline_summary(self, itinerary: Itinerary) -> str:
        """Format airline summary for display."""
        return self._airlines.summarize_airlines(itinerary)["airline_summary"]

    def format_transfer_summary(self, itinerary: Itinerary) -> str:
        """Format transfer/stopover summary."""
        stops = []
        for i, seg in enumerate(itinerary.segments[:-1]):
            city = self._airport_name(seg.destination)
            if city not in stops:
                stops.append(city)
        if stops:
            return "经" + "、".join(stops) + "转机"
        return "直飞"

    def recommendation_label(self, rec_type: str) -> str:
        """Translate recommendation type to Chinese."""
        return {
            "cheapest_reasonable": "最省钱",
            "lowest_risk": "最低风险",
            "best_overall": "综合最优",
            "": "—",
        }.get(rec_type, rec_type)

    def recommendation_reason(self, rec: RankedRecommendation) -> str:
        """Human-readable recommendation reason with trade-off explanation."""
        it = rec.itinerary
        saving = rec.savings_vs_baseline_usd
        airline_info = self._airlines.summarize_airlines(it)
        risk_level = rec.risk_assessment.risk_level

        if rec.recommendation_type == "cheapest_reasonable":
            if saving > 100:
                return (
                    f"这是当前最低价方案，比直飞/OTA便宜约${saving:.0f}，"
                    f"但需要{'自行安排地面接驳' if it.total_access_cost_usd > 0 else '接受拆分出票'}，"
                    f"适合能接受一定行程复杂度的人。"
                )
            return f"价格最低，省约${saving:.0f}，风险{risk_level}，需注意拆分出票。"

        if rec.recommendation_type == "lowest_risk":
            return (
                f"这个方案不一定最便宜，但拆分少、接驳风险低，"
                f"更适合不想折腾或对延误敏感的行程。"
            )

        if rec.recommendation_type == "best_overall":
            airline_tier = self._airlines.tier_label_for_score(rec.airline_quality_score)
            if saving > 50:
                return (
                    f"价格、风险和舒适度综合最均衡，"
                    f"比直飞省约${saving:.0f}，航司品质{airline_tier}。"
                )
            return f"综合最均衡，航司品质{airline_tier}，风险{risk_level}。"

        # Generic fallback with trade-off info
        parts = []
        if saving > 50:
            parts.append(f"比直飞省约${saving:.0f}（{rec.savings_percentage:.0f}%）")
        elif saving > 0:
            parts.append(f"略省${saving:.0f}")

        if risk_level == "low":
            parts.append("风险低")
        elif risk_level == "medium":
            parts.append("风险中等，可接受")

        airline_tier = self._airlines.tier_label_for_score(rec.airline_quality_score)
        parts.append(f"航司品质：{airline_tier}")

        if it.total_access_cost_usd > 0:
            parts.append("含地面接驳")

        return "，".join(parts) if parts else "—"

    def data_quality_label(self, itinerary: Itinerary) -> str:
        """Data quality label for display."""
        for offer in itinerary.offers:
            if offer.is_real:
                return "真实API"
            if offer.source == "mock_exact":
                return "demo"
            if offer.source == "mock_fallback":
                return "估算"
        return "未知"

    def _airport_name(self, code: str) -> str:
        """Get human-readable airport/city name from code."""
        airport = self._airports.get_airport(code)
        if airport:
            name = airport.city_cn or airport.city
            if len(name) <= 4:
                return name
            return name[:4]
        return code

    def _segment_type_label(self, seg) -> str:
        """Label segment type in Chinese."""
        countries = set()
        countries.add(self._airports.get_country(seg.origin))
        countries.add(self._airports.get_country(seg.destination))
        if len(countries) >= 2 or ("" in countries and len(countries) > 1):
            return "国际段"
        return "国内段"
