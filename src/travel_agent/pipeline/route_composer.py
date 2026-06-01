"""Compose complete itineraries from provider offers."""

from __future__ import annotations

from itertools import product

from travel_agent.contract.compiler import ExclusionRules
from travel_agent.pipeline.types import FlightOffer, HubCandidatePair, Itinerary, SearchTask


class RouteComposer:
    def compose(
        self,
        *,
        hub_pairs: list[HubCandidatePair],
        tasks: list[SearchTask],
        offers: list[FlightOffer],
        exclusions: ExclusionRules,
    ) -> list[Itinerary]:
        by_task: dict[str, list[FlightOffer]] = {}
        task_by_id = {task.task_id: task for task in tasks}
        for offer in offers:
            if offer.task_id:
                by_task.setdefault(offer.task_id, []).append(offer)
                continue
            for task in tasks:
                if (
                    task.origin == offer.origin
                    and task.destination == offer.destination
                    and task.leg_type == offer.leg_type
                ):
                    by_task.setdefault(task.task_id, []).append(offer)

        itineraries: list[Itinerary] = []
        for task in tasks:
            if task.leg_type == "direct":
                for offer in by_task.get(task.task_id, [])[:3]:
                    itinerary = self._build_itinerary(
                        itinerary_id=f"itin-{offer.id}",
                        route_type="baseline",
                        offers=[offer],
                    )
                    if self._allowed(itinerary, exclusions):
                        itineraries.append(itinerary)

        for pair in hub_pairs:
            leg_task_ids: list[str] = []
            if pair.origin_airport != pair.origin_hub:
                leg_task_ids.append(f"{pair.pair_id}:ground_cn")
            leg_task_ids.append(f"{pair.pair_id}:international")
            if pair.destination_hub != pair.destination_airport:
                leg_task_ids.append(f"{pair.pair_id}:domestic_us")
            if any(task_id not in task_by_id for task_id in leg_task_ids):
                continue
            leg_offers = [by_task.get(task_id, [])[:2] for task_id in leg_task_ids]
            if any(not bucket for bucket in leg_offers):
                continue
            for combo in product(*leg_offers):
                itinerary = self._build_itinerary(
                    itinerary_id="itin-" + "-".join(offer.id for offer in combo),
                    route_type="hub_split",
                    offers=list(combo),
                )
                if self._allowed(itinerary, exclusions):
                    itineraries.append(itinerary)
        return itineraries[:80]

    def _build_itinerary(self, itinerary_id: str, route_type: str, offers: list[FlightOffer]) -> Itinerary:
        segments = [segment for offer in offers for segment in offer.segments]
        route: list[str] = []
        for segment in segments:
            if not route:
                route.append(segment.origin)
            if route[-1] != segment.destination:
                route.append(segment.destination)
        price = sum(offer.total_price_usd for offer in offers)
        estimated = any(offer.has_estimated_data for offer in offers)
        return Itinerary(
            id=itinerary_id,
            route_type=route_type,
            route=route,
            offers=offers,
            segments=segments,
            total_price_usd=price,
            total_estimated_time_hours=sum(offer.estimated_time_hours or 0 for offer in offers) or _estimate_hours(segments),
            source="mock",
            confidence="estimated" if estimated else "known",
        )

    def _allowed(self, itinerary: Itinerary, exclusions: ExclusionRules) -> bool:
        for code in itinerary.route:
            if exclusions.airport_is_excluded(code):
                return False
        for segment in itinerary.segments:
            if exclusions.airline_is_excluded(segment.airline):
                return False
        return True


def _estimate_hours(segments) -> float:
    total = 0.0
    for segment in segments:
        if segment.mode == "ground":
            total += 3.0
        elif segment.origin in {"PVG", "SHA", "HGH", "NGB", "WNZ"} and segment.destination in {
            "JFK",
            "EWR",
            "IAD",
            "ORD",
            "ATL",
            "DFW",
            "LAX",
            "SFO",
            "SEA",
            "BOS",
            "PHL",
            "MIA",
        }:
            total += 14.0
        else:
            total += 2.0
    if len(segments) > 1:
        total += (len(segments) - 1) * 2.0
    return total
