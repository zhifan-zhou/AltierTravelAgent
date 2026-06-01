#!/usr/bin/env python3
"""Profile evaluation runner. Runs the pipeline on profile-specific queries
and measures recommendation quality per profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from travel_agent.core.logging import setup_logging
from travel_agent.core.orchestrator import TravelAgentOrchestrator
from travel_agent.services.itinerary_display_service import ItineraryDisplayService
from travel_agent.models.preference import get_profile_label, SCORING_PROFILES

_display = ItineraryDisplayService()


def load_queries(path: str, limit: int | None = None) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    if limit:
        queries = queries[:limit]
    return queries


async def run_profile_evals(queries: list[dict], output_dir: str | None = None) -> dict:
    setup_logging()
    orchestrator = TravelAgentOrchestrator()

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"runs/evals/profile/{ts}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    errors = 0
    route_ok = 0
    airline_ok = 0
    dq_warn_ok = 0
    profile_parse_ok = 0
    no_id_visible = 0

    metrics_per_profile = {p: {"queries": 0, "top_is_cheapest": 0, "top_is_lowest_risk": 0,
                                "top_has_major_airline": 0, "top_is_fastest": 0}
                           for p in SCORING_PROFILES}

    start = time.monotonic()

    for i, q in enumerate(queries):
        qid = q.get("id", f"q{i:03d}")
        query_text = q["query"]
        expected_profile = q.get("expected_profile", "balanced")
        print(f"[{i+1}/{len(queries)}] {qid}: {query_text[:65]}...", end=" ", flush=True)

        try:
            result = await orchestrator.run(query_text, debug=False)
            if result.error:
                errors += 1
                print(f"ERROR: {result.error[:80]}")
                continue

            itins = result.route_composer.output.itineraries if result.route_composer else []
            if not itins:
                print("SKIP: no itineraries")
                continue

            # Run all 5 profiles on the same candidates
            for profile in SCORING_PROFILES:
                risk = result.risk
                constraints = result.constraints
                comp = result.route_composer
                if not (comp and risk and constraints):
                    continue
                ranking_out = await orchestrator.ranking.execute(
                    (comp, risk, constraints), profile=profile,
                )
                rankings = ranking_out.rankings
                if not rankings:
                    continue
                top = rankings[0]
                mp = metrics_per_profile[profile]
                mp["queries"] += 1

                # Cheapest check
                all_prices = [r.itinerary.total_price_usd for r in rankings
                              if r.risk_assessment.risk_level in ("low", "medium")]
                if all_prices and top.itinerary.total_price_usd <= min(all_prices) + 1:
                    mp["top_is_cheapest"] += 1

                # Lowest risk check
                all_risks = [r.risk_assessment.risk_score for r in rankings]
                if all_risks and top.risk_assessment.risk_score <= min(all_risks) + 0.01:
                    mp["top_is_lowest_risk"] += 1

                # Major airline check
                from travel_agent.services.airline_service import AirlineService
                asvc = AirlineService()
                for seg in top.itinerary.segments:
                    a = asvc.get_airline(seg.airline) if seg.airline else None
                    if a and a.get("quality_tier") in ("premium", "major"):
                        mp["top_has_major_airline"] += 1
                        break

                # Fastest check
                all_times = [r.itinerary.total_estimated_time_hours for r in rankings]
                if all_times and top.itinerary.total_estimated_time_hours <= min(all_times) + 0.5:
                    mp["top_is_fastest"] += 1

            # Display quality checks on balanced results
            top = result.ranking.rankings[0] if result.ranking else None
            if top:
                route_str = _display.format_route_codes(top.itinerary)
                if "→" in route_str or "⇢" in route_str:
                    route_ok += 1
                if top.itinerary.id not in route_str or len(route_str) > 10:
                    no_id_visible += 1
                airline_str = _display.format_airline_summary(top.itinerary)
                if airline_str and airline_str != "未知航司":
                    airline_ok += 1

            # Data quality check
            if result.flight_retrieval:
                has_fallback = any(o.source == "mock_fallback" for o in result.flight_retrieval.all_offers)
                has_real = any(o.is_real for o in result.flight_retrieval.all_offers)
                if has_fallback or has_real:
                    dq_warn_ok += 1

            results.append({"id": qid, "query": query_text, "error": None})
            print("OK")

        except Exception as e:
            errors += 1
            print(f"EXCEPTION: {e!r}")
            results.append({"id": qid, "query": query_text, "error": str(e)})

    elapsed = time.monotonic() - start
    total = len(queries)
    err_free = total - errors

    metrics = {
        "total_queries": total, "elapsed_seconds": round(elapsed, 2),
        "route_display_success_rate": round(route_ok / max(err_free, 1), 3),
        "airline_display_success_rate": round(airline_ok / max(err_free, 1), 3),
        "data_quality_warning_success_rate": round(dq_warn_ok / max(err_free, 1), 3),
        "no_itinerary_id_visible_rate": round(no_id_visible / max(err_free, 1), 3),
        "error_count": errors,
    }

    # Per-profile metrics
    for p, mp in metrics_per_profile.items():
        n = mp["queries"]
        if n > 0:
            metrics[f"{p}_top_is_cheapest_rate"] = round(mp["top_is_cheapest"] / n, 3)
            metrics[f"{p}_top_is_lowest_risk_rate"] = round(mp["top_is_lowest_risk"] / n, 3)
            metrics[f"{p}_top_has_major_airline_rate"] = round(mp["top_has_major_airline"] / n, 3)
            metrics[f"{p}_top_is_fastest_rate"] = round(mp["top_is_fastest"] / n, 3)

    summary = {"metrics": metrics, "results": results, "generated_at": datetime.now().isoformat()}
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_path / "details.json").write_text(json.dumps({"per_profile": metrics_per_profile, "results": results}, indent=2, ensure_ascii=False, default=str))

    print("\n" + "=" * 60)
    print("PROFILE EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float) and "rate" in k:
            print(f"  {k:<40} {v:.1%}")
        else:
            print(f"  {k:<40} {v}")
    print(f"  Output: {out_path}")
    print("=" * 60)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Travel Agent Profile Eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    queries_path = Path(__file__).resolve().parent.parent / "evals" / "profile_eval_queries.jsonl"
    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found")
        raise SystemExit(1)

    queries = load_queries(str(queries_path), limit=args.limit)
    print(f"Loaded {len(queries)} profile eval queries\n")
    asyncio.run(run_profile_evals(queries, output_dir=args.output))


if __name__ == "__main__":
    main()
