"""DeepSeek client and requirement agent."""

from __future__ import annotations

import json
import time
from typing import Protocol

import httpx
from pydantic import ValidationError

from travel_agent.config import Settings, load_settings
from travel_agent.contract.models import TravelRequirementContract
from travel_agent.contract.normalization import airport_alias_map
from travel_agent.contract.route_semantics import RouteSemanticValidator
from travel_agent.llm.prompts import SYSTEM_PROMPT, build_requirement_prompt
from travel_agent.llm.schemas import TravelRequirementContractUpdate


class InvalidRequirementUpdate(ValueError):
    """Raised when DeepSeek does not return a valid executable update."""


class RequirementLLMClient(Protocol):
    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek chat client."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.last_meta: dict = {}

    @property
    def is_configured(self) -> bool:
        return self.settings.deepseek_configured

    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured:
            raise InvalidRequirementUpdate(
                "DeepSeek 未配置，无法启动 LLM-first chat。请在 .env 中设置 DEEPSEEK_API_KEY。"
            )
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.settings.deepseek_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.deepseek_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            self.last_meta = {
                "model": self.settings.deepseek_model,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "token_usage": data.get("usage") or {},
            }
            return data["choices"][0]["message"]["content"]

    async def ping(self) -> tuple[bool, str]:
        if not self.is_configured:
            return False, "not configured"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.settings.deepseek_base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                )
            return response.status_code == 200, f"HTTP {response.status_code}"
        except Exception as exc:  # pragma: no cover - network dependent
            return False, exc.__class__.__name__


class DeepSeekRequirementAgent:
    """LLM-first schema updater. It never searches flights."""

    def __init__(self, client: RequirementLLMClient | None = None):
        self.client = client or DeepSeekClient()
        self.last_meta: dict = {}
        self.last_validation_diagnostics: list[str] = []
        self.route_validator = RouteSemanticValidator()

    async def update(
        self,
        *,
        contract: TravelRequirementContract | None,
        user_message: str,
        history_summary: str = "",
        displayed_recommendations_summary: str = "",
    ) -> TravelRequirementContractUpdate:
        prompt = build_requirement_prompt(
            contract=contract,
            user_message=user_message,
            history_summary=history_summary,
            airport_alias_map=airport_alias_map(),
            displayed_recommendations_summary=displayed_recommendations_summary,
        )
        try:
            raw = await self.client.complete_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
        except httpx.HTTPError as exc:
            raise InvalidRequirementUpdate(str(exc)) from exc
        self.last_meta = dict(getattr(self.client, "last_meta", {}) or {})
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise InvalidRequirementUpdate("DeepSeek JSON root must be an object")
            update = TravelRequirementContractUpdate.model_validate(parsed)
            route_result = self.route_validator.validate_update(
                user_message=user_message,
                update=update,
                current_contract=contract,
            )
            self.last_validation_diagnostics = route_result.diagnostics
            return route_result.update
        except (json.JSONDecodeError, ValidationError, InvalidRequirementUpdate) as exc:
            raise InvalidRequirementUpdate(str(exc)) from exc
