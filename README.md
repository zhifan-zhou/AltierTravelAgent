# AltierTravelAgent

LLM-first Travel Agent MVP for conversational travel planning. v0.3 adds a planning-grade itinerary, cost-estimation, constraint-checking, response-only, and streaming layer while preserving the contract-based pipeline.

This is not a training project. It does not use a GPU and does not book, pay, ticket, hold, or lock prices.

## Pipeline

```text
User input
→ LLM schema extraction
→ semantic validation
→ contract normalization
→ contract merge
→ action decision
→ tool routing / search / clarification
→ response planning
→ streaming response rendering
```

## v0.3 capabilities

- Day-by-day 1-day, 3-day, and 5-day itinerary drafts
- Rough trip-cost estimates with explicit live/mock/estimate labels
- Constraint checks for pets, red-eye avoidance, nonstop preference, budget, weather, transfers, documents, and accessibility
- Open-Meteo geocoding/weather, Frankfurter currency, Python `zoneinfo` local time, local airport lookup, and optional Wikivoyage briefs
- Unified `ToolRequest` / `ToolResult` metadata with timeout, retry, TTL cache, and lightweight rate limiting
- Response-only ordinary renderer: normal chat receives only a `UserResponse`
- Streaming final responses enabled by default, with a deterministic chunked fallback
- Debug information isolated to opt-in debug output
- Multi-turn contract and active/inactive constraint history

## User-facing output boundary

Normal chat prints only actionable travel responses, concise tool summaries, source labels, necessary clarifying questions, and honest unavailability notices. It never prints schema updates, raw contracts, tool JSON, prompts, decision traces, stack traces, retry/cache details, or other internal diagnostics.

Debug mode is opt-in. Diagnostics are separated from the user response and written to stderr; the final response remains clean.

## Streaming

Final user responses stream in moderate chunks by default. Schema extraction and tool routing are never streamed. Deterministic itinerary, cost, tool, and flight-demo responses use the same `ResponseStreamer`.

Use `--no-stream` for one-shot rendering in tests or terminals that do not benefit from streaming.

## Live, demo, and estimated data

| Capability | Source | Classification |
|---|---|---|
| Geocoding | Open-Meteo | live public API |
| Weather | Open-Meteo | live forecast API |
| Currency | Frankfurter | latest available reference rate |
| Local time | Python `zoneinfo` with geocoded timezone | live local calculation |
| Airport lookup | bundled airport file | local static data |
| Destination brief | Wikivoyage/Wikimedia | attributed public content |
| Flight search/prices | bundled mock provider | mock/demo only |
| Lodging, food, local transport, activities | planning estimator | rough estimate |

Mock flights are never real prices, live inventory, or bookable results. Rough local costs are wide planning estimates, not quotes or financial advice. External API failures return an honest unavailable response and never fabricated fallback facts.

## Install

```bash
git clone https://github.com/zhifan-zhou/AltierTravelAgent.git
cd AltierTravelAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a real `DEEPSEEK_API_KEY` to local `.env` for interactive LLM chat. `.env`, runs, logs, and caches are ignored. Open-Meteo and Frankfurter need no key.

## CLI

```bash
# Streaming final responses (default)
PYTHONPATH=src python -m travel_agent.cli chat

# Render each final response at once
PYTHONPATH=src python -m travel_agent.cli chat --no-stream

# Diagnostics on stderr; user response remains separate
PYTHONPATH=src python -m travel_agent.cli chat --debug
```

Other commands:

```bash
PYTHONPATH=src python -m travel_agent.cli search "温州到匹兹堡，六月初，越便宜越好"
PYTHONPATH=src python -m travel_agent.cli llm-check --ping
```

Example conversation:

```text
你 > 我想从成都飞奥斯丁
你 > 六月初，越便宜越好
你 > 帮我安排三天行程
你 > 估算一下预算
你 > 目的地天气怎么样？
```

The itinerary is a practical draft, not a reservation. Specific tickets, opening hours, visa rules, airline policies, and accessibility arrangements must be confirmed with official sources.

## Tests and smoke

Pytest uses injected fake transports and does not require live network access:

```bash
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python scripts/final_codex_acceptance.py
```

Optional real-network smoke test:

```bash
PYTHONPATH=src python scripts/smoke_real_tools.py
```

## Safety and limitations

- Flight prices and availability remain mock/demo.
- No booking, payment, ticketing, price lock, airline login, or airline-site scraping.
- No production web UI.
- Public APIs may fail, change, or rate-limit requests.
- Cost output is a planning estimate, not financial advice.
- Visa, border, airline, pet, baggage, and accessibility policies require official confirmation.

## License

MIT License. See [LICENSE](LICENSE).
