"""Route Composer Agent: combines flight offers into complete itineraries.

Supports three HubSplit modes:
  - origin_side: access cost + international flight (no domestic needed)
  - destination_side: international flight + domestic connection
  - both_side: access cost + international + domestic connection
"""

from __future__ import annotations

import uuid

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import (
    FlightRetrievalOutput,
    HubSplitOutput,
    RouteComposerResult,
)
from travel_agent.models.airport import NearbyHub
from travel_agent.models.flight import FlightOffer, FlightSegment
from travel_agent.models.itinerary import Itinerary, RouteComposerOutput
from travel_agent.services.constraint_compiler import ExclusionRules

TOP_K = 3  # Take top-k offers per pair instead of [:1]


class RouteComposerAgent(BaseAgent[tuple[FlightRetrievalOutput, HubSplitOutput, ExclusionRules | None], RouteComposerResult]):
    """Compose raw flight offers into complete itineraries — with exclusion pre-filter."""

    name = "route_composer"

    async def execute(self, data: tuple[FlightRetrievalOutput, HubSplitOutput, ExclusionRules | None]) -> RouteComposerResult:
        flight_data, hub_split, exclusions = data
        itineraries: list[Itinerary] = []
        plan = hub_split.plan

        # 1. Direct/OTA itineraries
        for offer in flight_data.direct_offers:
            it = self._make_itinerary(it_type="direct", offers=[offer])
            itineraries.append(it)

        # 2. Hub-split itineraries — handle all three modes
        for pair in plan.candidate_hub_pairs:
            mode = pair.split_mode
            oh_code = pair.origin_hub_code
            dh_code = pair.destination_hub_code

            # Find matching international offers
            intl_offers = self._find_matching_offers(
                flight_data.hub_split_offers, oh_code, dh_code,
            )

            # Find matching domestic offers (only needed for dest_side and both_side)
            dom_offers: list[FlightOffer] = []
            if mode in ("destination_side", "both_side"):
                dom_offers = self._find_matching_offers(
                    flight_data.domestic_offers, dh_code, plan.destination_airport_code,
                )

            access_cost = pair.estimated_access_cost_usd
            access_time = pair.estimated_access_time_hours

            for intl in intl_offers[:TOP_K]:
                if mode == "origin_side":
                    # origin -> China hub -> destination (no domestic needed)
                    segments = list(intl.segments)
                    total_time = self._calc_total_time(segments) + access_time
                    total_price = intl.total_price_usd + access_cost
                    it = self._build_hubsplit(
                        plan, pair, segments, [intl],
                        total_price, total_time, access_cost,
                    )
                    itineraries.append(it)

                elif mode == "destination_side":
                    for dom in dom_offers[:TOP_K]:
                        segments = list(intl.segments) + list(dom.segments)
                        total_time = self._calc_total_time(segments) + access_time
                        total_price = intl.total_price_usd + dom.total_price_usd + access_cost
                        it = self._build_hubsplit(
                            plan, pair, segments, [intl, dom],
                            total_price, total_time, access_cost,
                        )
                        itineraries.append(it)

                elif mode == "both_side":
                    for dom in dom_offers[:TOP_K]:
                        segments = list(intl.segments) + list(dom.segments)
                        total_time = self._calc_total_time(segments) + access_time
                        total_price = intl.total_price_usd + dom.total_price_usd + access_cost
                        it = self._build_hubsplit(
                            plan, pair, segments, [intl, dom],
                            total_price, total_time, access_cost,
                        )
                        itineraries.append(it)

                else:
                    # Unknown mode — try with domestic if available
                    targets = dom_offers if dom_offers else [None]  # type: ignore
                    for dom in targets[:TOP_K]:
                        segs = list(intl.segments)
                        offers = [intl]
                        if dom is not None:
                            segs += list(dom.segments)
                            offers.append(dom)
                        total_time = self._calc_total_time(segs) + access_time
                        total_price = intl.total_price_usd + (dom.total_price_usd if dom else 0) + access_cost
                        it = self._build_hubsplit(
                            plan, pair, segs, offers,
                            total_price, total_time, access_cost,
                        )
                        itineraries.append(it)

        # Apply exclusion pre-filter
        filtered = []
        before = len(itineraries)
        if exclusions:
            for it in itineraries:
                route_codes = [s.origin for s in it.segments] + [it.destination_airport]
                if exclusions.route_contains_exclusion(route_codes):
                    filtered.append({
                        "route": "→".join(route_codes[:4]),
                        "excluded": [c for c in route_codes if exclusions.is_airport_excluded(c)],
                    })
                    continue
                itineraries_copy = list(itineraries)  # will reassign after loop
        # Actually filter:
        if exclusions:
            valid = []
            for it in itineraries:
                route_codes = [s.origin for s in it.segments] + [s.segments[-1].destination if it.segments else it.destination_airport]
                if not exclusions.route_contains_exclusion(route_codes):
                    valid.append(it)
            itineraries = valid

        self._last_composer_diagnostics = {
            "before_filter": before,
            "after_filter": len(itineraries),
            "filtered_by_exclusion": before - len(itineraries),
        }

        # Baseline
        baseline_id = None
        for it in itineraries:
            if it.type == "direct":
                baseline_id = it.id
                break

        self.logger.info(
            f"Composed {len(itineraries)} itineraries "
            f"({sum(1 for i in itineraries if i.type == 'direct')} direct, "
            f"{sum(1 for i in itineraries if i.type == 'hub_split')} hub_split)"
        )

        return RouteComposerResult(
            output=RouteComposerOutput(
                itineraries=itineraries,
                baseline_itinerary_id=baseline_id,
            )
        )

    def _build_hubsplit(
        self, plan, pair, segments, offers,
        total_price: float, total_time: float, access_cost: float,
    ) -> Itinerary:
        return Itinerary(
            id=f"hubsplit-{uuid.uuid4().hex[:8]}",
            type="hub_split",
            segments=segments,
            offers=offers,
            total_price_usd=round(total_price, 2),
            total_access_cost_usd=round(access_cost, 2),
            total_estimated_time_hours=round(total_time, 1),
            number_of_segments=len(segments),
            split_ticket_count=len(offers),
            origin_airport=plan.origin_airport_code,
            destination_airport=plan.destination_airport_code,
            main_international_leg=f"{pair.origin_hub_code}->{pair.destination_hub_code}",
            warnings=["Split ticket: 多段航程为分开出票，前段延误将影响后段行程。建议预留充足转机时间。"],
        )

    def _make_itinerary(self, it_type: str, offers: list[FlightOffer]) -> Itinerary:
        all_segments: list[FlightSegment] = []
        total_price = 0.0
        for offer in offers:
            all_segments.extend(offer.segments)
            total_price += offer.total_price_usd

        return Itinerary(
            id=f"{it_type}-{uuid.uuid4().hex[:8]}",
            type=it_type,
            segments=all_segments,
            offers=offers,
            total_price_usd=round(total_price, 2),
            total_access_cost_usd=0.0,
            total_estimated_time_hours=self._calc_total_time(all_segments),
            number_of_segments=len(all_segments),
            split_ticket_count=1 if len(offers) > 1 else 0,
            origin_airport=all_segments[0].origin if all_segments else "",
            destination_airport=all_segments[-1].destination if all_segments else "",
        )

    def _find_matching_offers(
        self, offers: list[FlightOffer], origin: str, destination: str,
    ) -> list[FlightOffer]:
        return [
            o for o in offers
            if o.segments
            and o.segments[0].origin == origin
            and o.segments[-1].destination == destination
        ]

    def _calc_total_time(self, segments: list[FlightSegment]) -> float:
        if not segments:
            return 0.0
        start = segments[0].departure_time
        end = segments[-1].arrival_time
        return (end - start).total_seconds() / 3600
