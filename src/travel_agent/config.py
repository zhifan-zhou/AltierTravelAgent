"""Project configuration for the LLM-first travel agent demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking up from CWD."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".env").exists():
            return candidate
    return current


@dataclass(frozen=True)
class Settings:
    project_root: Path
    env_path: Path
    env_found: bool
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    travel_agent_provider: str
    enable_tools: bool
    enabled_tools: tuple[str, ...]
    data_dir: Path
    runs_dir: Path

    @property
    def deepseek_configured(self) -> bool:
        return self.llm_provider.lower() == "deepseek" and bool(self.deepseek_api_key)

    @property
    def masked_deepseek_key(self) -> str:
        key = self.deepseek_api_key
        if not key:
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:4]}...{key[-4:]}"


def load_settings() -> Settings:
    """Explicitly load .env from the project root and return settings."""
    root = find_project_root()
    env_path = root / ".env"
    env_found = env_path.exists()
    if env_found:
        load_dotenv(dotenv_path=env_path, override=False)

    model = os.getenv("DEEPSEEK_MODEL") or os.getenv("DEEPSEEK_MODEL_FAST") or "deepseek-chat"

    return Settings(
        project_root=root,
        env_path=env_path,
        env_found=env_found,
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip() or "none",
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/"),
        deepseek_model=model.strip(),
        travel_agent_provider=os.getenv("TRAVEL_AGENT_PROVIDER", "mock").strip() or "mock",
        enable_tools=os.getenv("ENABLE_TOOLS", "true").strip().lower() not in {"0", "false", "no", "off"},
        enabled_tools=tuple(
            item.strip()
            for item in os.getenv(
                "ENABLED_TOOLS", "weather,airport_lookup,time,currency,destination_brief"
            ).split(",")
            if item.strip()
        ),
        data_dir=Path(os.getenv("TRAVEL_AGENT_DATA_DIR", root / "src" / "travel_agent" / "data")),
        runs_dir=Path(os.getenv("RUNS_DIR", root / "runs")),
    )
