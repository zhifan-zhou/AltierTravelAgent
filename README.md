# AltierTravelAgent

LLM-first Travel Agent MVP for conversational trip planning. v0.2 keeps the existing contract pipeline and adds real/free public-data adapters, a robust unified tool layer, richer constraints, debug diagnostics, and dialogue evals.

This is not a model-training project. It does not use a GPU and does not book, pay, ticket, hold, or lock prices.

## Pipeline

```text
User input
→ LLM schema extraction
→ semantic validation
→ contract normalization
→ contract merge
→ action decision
→ tool routing / search / clarification
→ CLI response
```

## v0.2 capabilities

- DeepSeek-backed structured contract updates with deterministic validation and normalization
- Open-Meteo geocoding and live weather forecasts (no API key)
- Frankfurter currency conversion/reference rates (no API key)
- Local time resolution using Open-Meteo timezone metadata plus Python `zoneinfo`
- Local airport lookup and optional Wikivoyage destination briefs
- Unified `ToolRequest` / `ToolResult` metadata: status, source, fetched time, live/mock flags, and sanitized error code
- External-call timeout, at most two retries, exponential backoff, in-memory TTL cache, and light rate limiting
- General companions/pets, budget, time, and flight preferences with active/inactive constraint history
- Multi-turn pending-question state and structured eval dialogues
- Developer-only `chat --debug` diagnostics

## Live data and demo data

| Capability | Source | Classification |
|---|---|---|
| Geocoding | Open-Meteo | live public API |
| Weather | Open-Meteo | live/forecast public API |
| Currency | Frankfurter | latest available reference rate |
| Local time | Python `zoneinfo` using geocoded timezone | live local calculation |
| Airport lookup | bundled airport file | local static data |
| Destination brief | Wikivoyage/Wikimedia | attributed public content |
| Flight search/prices | bundled mock provider | mock/demo only |

Flight results are always labeled as demo data. They are not real prices, availability, or bookable inventory.

If an external API times out, rate-limits, returns invalid data, or is unavailable, its tool returns `unavailable`/`error` with no fabricated fallback facts. Missing optional keys never crash the default free adapters.

## Install

```bash
git clone https://github.com/zhifan-zhou/AltierTravelAgent.git
cd AltierTravelAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a real `DEEPSEEK_API_KEY` to the local `.env` for interactive LLM chat. `.env`, runs, logs, and caches are ignored and must not be committed. Open-Meteo and Frankfurter need no key.

## CLI

```bash
PYTHONPATH=src python -m travel_agent.cli chat
PYTHONPATH=src python -m travel_agent.cli search "温州到匹兹堡，六月初，越便宜越好"
PYTHONPATH=src python -m travel_agent.cli llm-check --ping
```

Developer diagnostics are opt-in and do not appear in normal output:

```bash
PYTHONPATH=src python -m travel_agent.cli chat --debug
```

Debug mode shows intent, contract diff, missing fields, next action, tool request, and result metadata. Debug artifacts are written below ignored `runs/` paths.

## Tests and manual smoke

The test suite injects fake HTTP transports and never requires live network access:

```bash
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python scripts/final_codex_acceptance.py
```

Optional real-network smoke test:

```bash
PYTHONPATH=src python scripts/smoke_real_tools.py
```

Network failure is reported clearly and is never converted into fake success.

## Safety and limitations

- Flight price search remains mock/demo unless a real provider is explicitly added in the future.
- No booking, payment, ticketing, price lock, passenger document submission, or airline-site scraping.
- No production web UI.
- Public APIs may fail, change, or rate-limit requests.
- Currency output is conversion data, not financial advice; all travel results are not travel/legal advice.
- Airline, visa, border, baggage, pet, and accessibility policies require official-source confirmation.

## License

MIT License. See [LICENSE](LICENSE).
