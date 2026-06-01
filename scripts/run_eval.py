#!/usr/bin/env python3
"""Evaluation runner for Travel Agent MVP — Phase 4.

Metrics per query and aggregated across all queries.
Saves per-query diagnostics to diagnostics.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from travel_agent.core.logging import setup_logging
from travel_agent.core.orchestrator import TravelAgentOrchestrator


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


async def run_evals(queries: list[dict], output_dir: str | None = None) -> dict:
    setup_logging()
    orchestrator = TravelAgentOrchestrator()

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"runs/evals/{ts}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    errors = 0
    parse_ok = 0
    hubsplit_ok = 0
    tasks_ok = 0
    offers_ok = 0
    itin_ok = 0
    split_itin_ok = 0
    total_hub_pairs = 0
    total_tasks = 0
    total_offers = 0
    total_itineraries = 0
    total_split_itineraries = 0
    total_recommendations = 0
    total_direct_price = 0.0
    direct_price_count = 0

    # Track failure reasons
    failure_reasons: dict[str, int] = {}

    start = time.monotonic()

    for i, q in enumerate(queries):
        qid = q.get("id", f"q{i:03d}")
        query_text = q["query"]
        print(f"[{i+1}/{len(queries)}] {qid}: {query_text[:65]}...", end=" ", flush=True)

        try:
            result = await orchestrator.run(query_text, debug=False)

            # Count per-query
            n_pairs = len(result.hub_split.plan.candidate_hub_pairs) if result.hub_split else 0
            n_tasks = len(result.search_strategy.search_tasks) if result.search_strategy else 0
            n_offers = len(result.flight_retrieval.all_offers) if result.flight_retrieval else 0
            itins = result.route_composer.output.itineraries if result.route_composer else []
            n_itins = len(itins)
            n_split = sum(1 for it in itins if it.type == "hub_split")
            n_recs = len(result.ranking.rankings) if result.ranking else 0

            # Query-level booleans
            if result.intake and result.intake.origin_text and result.intake.destination_text:
                parse_ok += 1
            if n_pairs > 0:
                hubsplit_ok += 1
            if n_tasks > 0:
                tasks_ok += 1
            if n_offers > 0:
                offers_ok += 1
            if n_itins > 0:
                itin_ok += 1
            if n_split > 0:
                split_itin_ok += 1

            # Totals
            total_hub_pairs += n_pairs
            total_tasks += n_tasks
            total_offers += n_offers
            total_itineraries += n_itins
            total_split_itineraries += n_split
            total_recommendations += n_recs

            # Direct price baseline
            for it in itins:
                if it.type == "direct":
                    total_direct_price += it.total_price_usd
                    direct_price_count += 1

            # Per-query diagnostics
            diag = {
                "id": qid,
                "query": query_text,
                "error": result.error,
                "parse_ok": bool(result.intake and result.intake.origin_text and result.intake.destination_text),
                "accepts_nearby": result.intake.accepts_nearby_hubs if result.intake else False,
                "hub_pairs": n_pairs,
                "origin_side_pairs": sum(1 for p in (result.hub_split.plan.candidate_hub_pairs if result.hub_split else []) if p.split_mode == "origin_side"),
                "dest_side_pairs": sum(1 for p in (result.hub_split.plan.candidate_hub_pairs if result.hub_split else []) if p.split_mode == "destination_side"),
                "both_side_pairs": sum(1 for p in (result.hub_split.plan.candidate_hub_pairs if result.hub_split else []) if p.split_mode == "both_side"),
                "search_tasks": n_tasks,
                "flight_offers": n_offers,
                "offers_exact": sum(1 for o in (result.flight_retrieval.all_offers if result.flight_retrieval else []) if o.source == "mock_exact"),
                "offers_fallback": sum(1 for o in (result.flight_retrieval.all_offers if result.flight_retrieval else []) if o.source == "mock_fallback"),
                "itineraries": n_itins,
                "split_itineraries": n_split,
                "direct_itineraries": sum(1 for it in itins if it.type == "direct"),
                "recommendations": n_recs,
            }

            # Composition failure diagnostics
            if result.hub_split and result.hub_split.plan.candidate_hub_pairs:
                missing_intl = 0
                missing_dom = 0
                composed = 0
                for pair in result.hub_split.plan.candidate_hub_pairs:
                    oh, dh = pair.origin_hub_code, pair.destination_hub_code
                    has_intl = any(
                        o.segments and o.segments[0].origin == oh and o.segments[-1].destination == dh
                        for o in (result.flight_retrieval.hub_split_offers if result.flight_retrieval else [])
                    )
                    needs_dom = pair.split_mode in ("destination_side", "both_side")
                    has_dom = True
                    if needs_dom:
                        has_dom = any(
                            o.segments and o.segments[0].origin == dh and
                            o.segments[-1].destination == result.hub_split.plan.destination_airport_code
                            for o in (result.flight_retrieval.domestic_offers if result.flight_retrieval else [])
                        )
                    has_composed = any(
                        it.type == "hub_split" and it.main_international_leg == f"{oh}->{dh}"
                        for it in itins
                    )
                    if has_composed:
                        composed += 1
                    else:
                        if not has_intl:
                            missing_intl += 1
                            failure_reasons["missing_international_flight"] = \
                                failure_reasons.get("missing_international_flight", 0) + 1
                        if not has_dom and needs_dom:
                            missing_dom += 1
                            failure_reasons["missing_domestic_flight"] = \
                                failure_reasons.get("missing_domestic_flight", 0) + 1

                diag["composition"] = {
                    "pairs_total": len(result.hub_split.plan.candidate_hub_pairs),
                    "pairs_composed": composed,
                    "pairs_failed_missing_intl": missing_intl,
                    "pairs_failed_missing_dom": missing_dom,
                    "composition_rate": round(composed / max(len(result.hub_split.plan.candidate_hub_pairs), 1), 3),
                }

            if result.error:
                errors += 1
                print(f"ERROR: {result.error[:80]}")
            else:
                print("OK")

            results.append(diag)

        except Exception as e:
            errors += 1
            print(f"EXCEPTION: {e!r}")
            results.append({"id": qid, "query": query_text, "error": str(e)})

    elapsed = time.monotonic() - start
    total = len(queries)

    avg_direct = round(total_direct_price / direct_price_count, 0) if direct_price_count else 0

    metrics = {
        "total_queries": total,
        "elapsed_seconds": round(elapsed, 2),
        "parse_success_rate": round(parse_ok / total, 3),
        "hubsplit_plan_generated_rate": round(hubsplit_ok / total, 3),
        "search_task_generated_rate": round(tasks_ok / total, 3),
        "flight_offer_return_rate": round(offers_ok / total, 3),
        "itinerary_generated_rate": round(itin_ok / total, 3),
        "split_itinerary_generated_rate": round(split_itin_ok / total, 3),
        "avg_hub_pairs_per_query": round(total_hub_pairs / total, 1),
        "avg_search_tasks_per_query": round(total_tasks / total, 1),
        "avg_offers_per_query": round(total_offers / total, 1),
        "avg_itineraries_per_query": round(total_itineraries / total, 1),
        "avg_split_itineraries_per_query": round(total_split_itineraries / total, 1),
        "avg_recommendations": round(total_recommendations / total, 1),
        "avg_direct_price_baseline_usd": avg_direct,
        "error_count": errors,
        "error_rate": round(errors / total, 3),
        "top_failure_reasons": dict(
            sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]
        ),
    }

    summary = {
        "metrics": metrics,
        "results": results,
        "generated_at": datetime.now().isoformat(),
    }
    summary_path = out_path / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Per-query diagnostics
    diag_path = out_path / "diagnostics.json"
    diag_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Phase 4)")
    print("=" * 60)
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        if isinstance(val, float) and key.endswith("rate"):
            print(f"  {label:<35} {val:.1%}")
        elif isinstance(val, float):
            print(f"  {label:<35} {val:.1f}")
        elif isinstance(val, dict):
            print(f"  {label}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {label:<35} {val}")
    print(f"  {'Output':<35} {summary_path}")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Travel Agent MVP Eval Runner")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--queries-file", type=str, default=None)
    args = parser.parse_args()

    if args.queries_file:
        queries_path = args.queries_file
    else:
        queries_path = Path(__file__).resolve().parent.parent / "evals" / "eval_queries.jsonl"

    if not Path(queries_path).exists():
        print(f"ERROR: Queries file not found: {queries_path}")
        raise SystemExit(1)

    queries = load_queries(str(queries_path), limit=args.limit)
    print(f"Loaded {len(queries)} eval queries from {queries_path}\n")
    asyncio.run(run_evals(queries, output_dir=args.output))


if __name__ == "__main__":
    main()
