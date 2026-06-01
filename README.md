# AltierTravelAgent

LLM-first travel agent MVP for conversational trip planning.

## Status

AltierTravelAgent is an MVP / research prototype / demo project. It focuses on a contract-based travel-planning loop rather than model training, booking, payment, or production flight inventory.

## Core Pipeline

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

## Features

- Conversational CLI for trip-planning turns
- DeepSeek-backed LLM schema extraction
- Route understanding with `RouteSemanticValidator`
- Local airport lookup through `AirportService`
- Contract-based state management and merge logic
- `ToolRouter` with an MCP-ready local tool layer for non-flight actions
- Weather, time, and currency tools implemented as safe local stubs
- Mock/demo flight search for deterministic development and tests
- Focused pytest suite and final acceptance script

## Installation

```bash
git clone https://github.com/zhifan-zhou/AltierTravelAgent.git
cd AltierTravelAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

```bash
cp .env.example .env
```

Then fill in:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

The default provider is mock/demo search. Keep `.env` local and do not commit real API keys.

## Run The CLI

```bash
PYTHONPATH=src python -m travel_agent.cli chat
```

For a deterministic mock demo without a real DeepSeek key:

```bash
PYTHONPATH=src python -m travel_agent.cli search "温州到匹兹堡，六月初，可以从上海走，越便宜越好"
```

## LLM Check

```bash
PYTHONPATH=src python -m travel_agent.cli llm-check --ping
```

If no real DeepSeek key is configured, the check will report that the key is missing or the ping failed.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python scripts/final_codex_acceptance.py
```

## Limitations

- Weather, time, and currency tools are currently safe stubs, not real-time integrations.
- Flight prices and availability are mock/demo data, not real market quotes.
- There is no booking, ticketing, payment, or passenger-data vault support.
- There is no production-grade web UI yet.
- Visa, border-entry, disruption, and fare-rule handling are not production-grade.

## License

MIT License. See [LICENSE](LICENSE).
