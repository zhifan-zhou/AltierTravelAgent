"""Flight Retrieval Agent: executes search tasks against providers."""

from __future__ import annotations

from travel_agent.agents.base import BaseAgent
from travel_agent.models.agent_outputs import FlightRetrievalOutput, SearchStrategyOutput
from travel_agent.models.flight import FlightOffer
from travel_agent.services.airport_service import AirportService
from travel_agent.services.constraint_compiler import ExclusionRules


class FlightRetrievalAgent(BaseAgent[SearchStrategyOutput, FlightRetrievalOutput]):
    """Execute flight searches using the provider router.

    Flights come ONLY from providers -- never from LLM generation.
    """

    name = "flight_retrieval"

    def __init__(self, router=None, airport_service: AirportService | None = None):
        super().__init__()
        self._router = router
        self._airport_service = airport_service or AirportService()
        self._last_skip_diagnostics: dict = {}

    async def execute(self, data: tuple[SearchStrategyOutput, ExclusionRules | None]) -> FlightRetrievalOutput:
        search_output, exclusions = data
        self._last_skip_diagnostics = {}
        skipped: list[dict] = []
        all_offers: list[FlightOffer] = []
        direct_offers: list[FlightOffer] = []
        hub_split_offers: list[FlightOffer] = []
        domestic_offers: list[FlightOffer] = []

        for task in search_output.search_tasks:
            # Skip excluded airports
            if exclusions and (exclusions.is_airport_excluded(task.origin) or
                               exclusions.is_airport_excluded(task.destination)):
                skipped.append({
                    "route": f"{task.origin}->{task.destination}",
                    "reason": f"Excluded by constraint"
                })
                continue
            try:
                if self._router is not None:
                    offers = await self._router.search_flights(task)
                else:
                    offers = []
                all_offers.extend(offers)
                self.logger.info(f"Searched {task.origin}->{task.destination}: {len(offers)} offers")
            except Exception as e:
                self.logger.warning(f"Search failed for {task.origin}->{task.destination}: {e}")

        self._last_skip_diagnostics = {"skipped_tasks": len(skipped), "details": skipped}

        # Categorize offers using airport data, not hardcoded sets
        for offer in all_offers:
            if self._is_direct_route(offer, search_output):
                direct_offers.append(offer)
            elif self._is_international(offer):
                hub_split_offers.append(offer)
            else:
                domestic_offers.append(offer)

        self.logger.info(
            f"Retrieved: {len(direct_offers)} direct, "
            f"{len(hub_split_offers)} international, "
            f"{len(domestic_offers)} domestic"
        )

        return FlightRetrievalOutput(
            direct_offers=direct_offers,
            hub_split_offers=hub_split_offers,
            domestic_offers=domestic_offers,
            all_offers=all_offers,
        )

    def _is_direct_route(self, offer: FlightOffer, data: SearchStrategyOutput) -> bool:
        if not data.direct_task:
            return False
        if not offer.segments:
            return False
        first = offer.segments[0].origin
        last = offer.segments[-1].destination
        return first == data.direct_task.origin and last == data.direct_task.destination

    def _is_international(self, offer: FlightOffer) -> bool:
        """Check if an offer crosses country boundaries using airport data."""
        if not offer.segments:
            return False
        countries: set[str] = set()
        for seg in offer.segments:
            origin_country = self._airport_service.get_country(seg.origin)
            dest_country = self._airport_service.get_country(seg.destination)
            if origin_country:
                countries.add(origin_country)
            if dest_country:
                countries.add(dest_country)
        return len(countries) >= 2
