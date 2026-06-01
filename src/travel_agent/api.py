"""FastAPI web API for the Travel Agent MVP."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from travel_agent.core.config import get_settings
from travel_agent.core.logging import setup_logging
from travel_agent.core.orchestrator import TravelAgentOrchestrator


class SearchRequest(BaseModel):
    query: str = Field(description="Natural language travel query")
    debug: bool = Field(default=False)


# Orchestrator created per-request via lazy init
_orchestrator: TravelAgentOrchestrator | None = None


def _get_orchestrator() -> TravelAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TravelAgentOrchestrator()
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="Travel Agent MVP",
    description="AI Travel Agent for route optimization via hub splitting",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "service": "travel-agent-mvp",
        "provider": settings.travel_agent_provider,
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Search for travel itineraries using the full agent pipeline."""
    orchestrator = _get_orchestrator()
    result = await orchestrator.run(query=request.query, debug=request.debug)

    status_code = 200 if result.error is None else 500

    return JSONResponse(
        status_code=status_code,
        content={
            "query": request.query,
            "error": result.error,
            "result": result.model_dump(mode="json"),
        },
    )


def main():
    """Entry point for uvicorn."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "travel_agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
