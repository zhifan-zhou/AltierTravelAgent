"""Search task planning from hub candidates."""

from __future__ import annotations

from travel_agent.contract.compiler import SearchConstraints
from travel_agent.pipeline.types import HubCandidatePair, SearchTask


class SearchTaskPlanner:
    def plan(
        self,
        *,
        constraints: SearchConstraints,
        hub_pairs: list[HubCandidatePair],
    ) -> list[SearchTask]:
        tasks: list[SearchTask] = []
        excluded = set(constraints.exclusions.excluded_airports)

        def add(task: SearchTask) -> None:
            if task.origin in excluded or task.destination in excluded:
                return
            if task.origin == task.destination:
                return
            if (task.leg_type, task.origin, task.destination, task.pair_id) in seen:
                return
            seen.add((task.leg_type, task.origin, task.destination, task.pair_id))
            tasks.append(task)

        seen: set[tuple[str, str, str, str | None]] = set()
        add(
            SearchTask(
                task_id=f"direct:{constraints.origin_airport}->{constraints.destination_airport}",
                leg_type="direct",
                origin=constraints.origin_airport,
                destination=constraints.destination_airport,
                cabin=constraints.cabin,
            )
        )
        for pair in hub_pairs:
            if pair.origin_airport != pair.origin_hub:
                add(
                    SearchTask(
                        task_id=f"{pair.pair_id}:ground_cn",
                        pair_id=pair.pair_id,
                        leg_type="ground_cn",
                        origin=pair.origin_airport,
                        destination=pair.origin_hub,
                        cabin=constraints.cabin,
                    )
                )
            add(
                SearchTask(
                    task_id=f"{pair.pair_id}:international",
                    pair_id=pair.pair_id,
                    leg_type="international",
                    origin=pair.origin_hub,
                    destination=pair.destination_hub,
                    cabin=constraints.cabin,
                )
            )
            if pair.destination_hub != pair.destination_airport:
                add(
                    SearchTask(
                        task_id=f"{pair.pair_id}:domestic_us",
                        pair_id=pair.pair_id,
                        leg_type="domestic_us",
                        origin=pair.destination_hub,
                        destination=pair.destination_airport,
                        cabin=constraints.cabin,
                    )
                )
        return tasks
