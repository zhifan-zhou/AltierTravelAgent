# Evaluation Framework

Runs the full Travel Agent pipeline on a set of test queries and measures success metrics.

## Query Format

`evals/eval_queries.jsonl` — one JSON object per line:

```json
{"id": "q001", "query": "宁波到匹兹堡", "expected_origin": "宁波", "expected_dest": "匹兹堡", "tags": ["cheap", "nearby"]}
```

## Running

```bash
# Run all eval queries
PYTHONPATH=src python scripts/run_eval.py

# Run with limit (first N queries)
PYTHONPATH=src python scripts/run_eval.py --limit 5

# Run with specific output dir
PYTHONPATH=src python scripts/run_eval.py --output runs/evals/custom
```

## Metrics

- **parse_success_rate**: % of queries where origin AND destination were extracted
- **hubsplit_success_rate**: % of queries where HubSplit found nearby hubs
- **split_route_generated_rate**: % where split route itineraries were created
- **recommendation_count**: avg recommendations per query
- **error_count**: pipeline errors

## Adding Queries

Add new lines to `eval_queries.jsonl`. Each line should have:

| Field | Required | Description |
|-------|----------|-------------|
| id | yes | Unique query identifier |
| query | yes | Natural language query (Chinese preferred) |
| expected_origin | no | Expected origin city (Chinese name) |
| expected_dest | no | Expected destination city (Chinese name) |
| tags | no | List of tag strings for filtering |
