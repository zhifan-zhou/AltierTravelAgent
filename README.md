# AltierTravelAgent

LLM-first travel planning prototype. v0.4 adds a FastAPI backend, lightweight web UI, session persistence, contract sidebar, product cards, streaming web chat, and GitHub Actions CI while preserving the v0.3 CLI planning agent.

This is not a training, GPU, fine-tuning, booking, payment, ticketing, price-lock, or airline-login project.

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

v0.4 adds:

```text
FastAPI backend
→ session store
→ streaming API endpoint
→ web UI chat
→ contract sidebar
→ itinerary/cost/constraint/source cards
→ CI quality gate
```

## v0.4 capabilities

- FastAPI backend with health, session, contract summary, non-stream chat, and streaming chat endpoints
- Lightweight vanilla HTML/CSS/JS web UI served by FastAPI
- Streaming assistant response in CLI and web
- Local JSON session persistence under `.local/sessions/`
- Sanitized TravelContract sidebar for route, dates, budget, companions, preferences, missing fields, and warnings
- Itinerary, cost estimate, constraint, source, safety, and mock flight cards
- Response-only user renderer boundary for ordinary CLI and web output
- Debug remains opt-in; ordinary UI/API does not expose debug/internal data
- GitHub Actions CI for pytest, final acceptance, and `.env` tracking safety
- Provider disclosure model for future flight providers while keeping current flight data mock/demo only

## Install

```bash
git clone https://github.com/zhifan-zhou/AltierTravelAgent.git
cd AltierTravelAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Interactive real-LLM CLI chat needs a local `DEEPSEEK_API_KEY`. The web prototype can run without a key and falls back to the deterministic fake LLM used by tests. `.env`, `.local/`, `runs/`, logs, and caches are ignored.

## Web prototype

Start the backend:

```bash
PYTHONPATH=src uvicorn travel_agent.server.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Try:

```text
我想从成都飞奥斯丁
六月初，越便宜越好
我想带狗一起去，不想坐红眼航班，最好不要转机
帮我安排三天行程
估算一下预算
目的地天气怎么样？
奥斯丁现在几点？
```

Expected behavior:

- Chat response streams into the assistant bubble.
- Contract sidebar updates after each turn.
- Itinerary, cost, constraint, source, and safety cards update.
- Flight cards are clearly labeled demo/mock and not bookable.
- No booking/payment UI appears.
- User-friendly errors are shown without traceback.

## API summary

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Healthcheck with service/version |
| `POST /api/sessions` | Create a session |
| `GET /api/sessions` | List sessions |
| `GET /api/sessions/{session_id}` | Resume/read a session |
| `DELETE /api/sessions/{session_id}` | Clear a session |
| `GET /api/sessions/{session_id}/contract` | Read sanitized contract summary |
| `POST /api/chat` | Non-stream chat response for tests/fallback |
| `POST /api/chat/stream` | SSE streaming chat response |

SSE emits only user-visible text chunks:

```text
event: token
data: {"text": "..."}

event: final
data: {"contract_summary": {...}, "cards": [...], "sources": [...], "warnings": [...]}
```

Raw tool requests/results, raw contracts, prompts, stack traces, route validator diagnostics, retry/cache details, and debug logs are not streamed.

## CLI

The v0.3 CLI remains supported:

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

## Session persistence

Sessions are stored as JSON files under:

```text
.local/sessions/
```

The store keeps user-visible messages, sanitized contract summary, cards, sources, warnings, and the minimum contract JSON needed to resume the planning state. It does not save API keys, debug traces, prompts, raw `ToolRequest`, raw `ToolResult`, or credentials.

Use the web Reset button or:

```bash
curl -X DELETE http://127.0.0.1:8000/api/sessions/{session_id}
```

## Live, demo, and estimated data

| Capability | Source | Classification |
|---|---|---|
| Geocoding | Open-Meteo | live public API |
| Weather | Open-Meteo | live forecast API |
| Currency | Frankfurter | latest available reference rate |
| Local time | Python `zoneinfo` with geocoded timezone | live local calculation |
| Airport lookup | bundled airport file | local static data |
| Destination brief | Wikivoyage/Wikimedia | attributed public content |
| Flight search/prices | bundled mock provider | mock/demo only, not bookable |
| Lodging, food, local transport, activities | planning estimator | rough estimate |

Mock flights are never real prices, live inventory, or bookable results. Rough local costs are wide planning estimates, not quotes or financial advice. External API failures return honest unavailable responses and never fabricated fallback facts.

## CI

GitHub Actions runs on push and pull request:

```bash
pip install -r requirements.txt
test -z "$(git ls-files .env)"
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python scripts/final_codex_acceptance.py
```

CI does not run the live smoke script and does not require `DEEPSEEK_API_KEY`.

## Tests and smoke

Local deterministic checks:

```bash
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python scripts/final_codex_acceptance.py
```

Optional real-network smoke test:

```bash
PYTHONPATH=src python scripts/smoke_real_tools.py
```

The smoke script calls public APIs and may fail honestly if the network or upstream service is unavailable.

## Safety and limitations

- Not production-ready.
- No booking, payment, ticketing, price lock, airline login, CAPTCHA bypass, airline-site scraping, or passenger document collection.
- Flight prices and availability remain mock/demo unless a future provider explicitly says otherwise.
- Public APIs may fail, change, rate-limit, or return incomplete data.
- Cost output is a planning estimate, not financial advice.
- Visa, border, airline, pet, baggage, and accessibility policies require official confirmation.

## License

MIT License. See [LICENSE](LICENSE).
