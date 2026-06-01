"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    log_level: str = "INFO"
    debug: bool = False

    # Data directory — resolved to absolute at init time
    _data_dir_override: Optional[str] = None

    # Flight API provider selection: "mock" (default), "duffel", or "hybrid"
    travel_agent_provider: str = "mock"

    # Hybrid mode settings
    hybrid_enable_mock_fallback: bool = True
    real_provider_priority: str = "duffel,serpapi_google_flights,searchapi_google_flights,skyscanner,kiwi,amadeus"

    # Mock provider fallback pricing (deterministic, based on hub scores)
    mock_provider_enable_fallback: bool = True

    # Route composer limits
    max_itineraries_per_query: int = 30
    max_offers_per_leg: int = 3

    # ── Provider API credentials ──────────────────────────────────────
    # Duffel (primary real provider)
    duffel_api_token: str = ""
    duffel_api_key: str = ""  # legacy alias
    duffel_base_url: str = "https://api.duffel.com"
    duffel_api_version: str = "v2"  # Deprecated but kept for compat
    duffel_timeout_seconds: int = 20
    duffel_max_retries: int = 2

    # SerpApi Google Flights
    serpapi_api_key: str = ""

    # SearchApi Google Flights
    searchapi_api_key: str = ""

    # Skyscanner
    skyscanner_api_key: str = ""

    # Kiwi.com
    kiwi_api_key: str = ""

    # Amadeus (legacy/fallback only)
    amadeus_api_key: str = ""
    amadeus_api_secret: str = ""

    # ── LLM Provider ────────────────────────────────────────────────
    llm_provider: str = "none"  # none | deepseek

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_fast: str = "deepseek-chat"
    deepseek_model_reasoning: str = "deepseek-reasoner"
    deepseek_timeout_seconds: int = 30
    deepseek_max_retries: int = 2

    # LLM feature toggles
    llm_enable_intake: bool = False
    llm_enable_clarification: bool = False
    llm_enable_preference_inference: bool = False
    llm_enable_explanation: bool = False
    llm_enable_strategy_reasoning: bool = False

    # Search budget
    max_search_tasks_mock: int = 80
    max_search_tasks_real: int = 20
    max_hub_pairs_real: int = 8

    # HubSplit config
    max_origin_hubs: int = 4
    max_dest_hubs: int = 6
    max_candidate_pairs: int = 30

    # Scoring config
    default_baseline_price_usd: float = 2000.0

    # Date search defaults
    default_departure_days_from_now: int = 14
    default_date_window_days: int = 14

    # Output
    runs_dir: str = "runs"

    @property
    def data_dir(self) -> Path:
        """Resolve data directory to absolute path.

        Tries in order:
        1. Explicit override (_data_dir_override)
        2. TRAVEL_AGENT_DATA_DIR env var
        3. Package-relative: <package>/data/
        4. CWD-relative: src/travel_agent/data
        """
        import os
        from pathlib import Path as _Path

        if self._data_dir_override:
            return _Path(self._data_dir_override)

        env_dir = os.environ.get("TRAVEL_AGENT_DATA_DIR")
        if env_dir:
            return _Path(env_dir)

        # Package-relative: resolve from this config file's location
        package_data = _Path(__file__).resolve().parent.parent / "data"
        if package_data.is_dir():
            return package_data

        # Fallback: CWD-relative
        return _Path("src/travel_agent/data")

    @property
    def data_path(self) -> Path:
        return self.data_dir


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
