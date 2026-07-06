"""Frankfurter currency adapter."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from travel_agent.tools.http_client import HttpClient, HttpClientError


CURRENCY_ALIASES = {
    "美元": "USD",
    "美金": "USD",
    "人民币": "CNY",
    "元": "CNY",
    "欧元": "EUR",
    "英镑": "GBP",
    "日元": "JPY",
    "新币": "SGD",
    "新加坡元": "SGD",
}


class CurrencyConversion(BaseModel):
    amount: float
    from_currency: str
    to_currency: str
    converted_amount: float
    rate: float
    date: str
    source: str = "frankfurter"
    fetched_at: datetime
    is_live: bool = True


class ExchangeRates(BaseModel):
    base: str
    rates: dict[str, float] = Field(default_factory=dict)
    date: str
    source: str = "frankfurter"
    fetched_at: datetime
    is_live: bool = True


def normalize_currency(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    code = CURRENCY_ALIASES.get(raw, raw.upper())
    return code if len(code) == 3 and code.isalpha() else None


class FrankfurterCurrencyClient:
    endpoint = "https://api.frankfurter.dev/v1/latest"

    def __init__(self, http: HttpClient | None = None):
        self.http = http or HttpClient()

    def convert(self, amount: float, from_currency: str, to_currency: str) -> CurrencyConversion:
        source = normalize_currency(from_currency)
        target = normalize_currency(to_currency)
        if not source or not target:
            raise HttpClientError("invalid_currency", "Currency code is missing or invalid")
        payload = self.http.get_json(
            self.endpoint,
            params={"amount": amount, "from": source, "to": target},
            cache_ttl=60 * 60,
        )
        converted = (payload.get("rates") or {}).get(target)
        if converted is None:
            raise HttpClientError("invalid_currency", "Currency is unsupported or missing from response")
        converted_amount = float(converted)
        return CurrencyConversion(
            amount=float(amount),
            from_currency=source,
            to_currency=target,
            converted_amount=converted_amount,
            rate=converted_amount / float(amount) if amount else 0.0,
            date=str(payload.get("date") or ""),
            fetched_at=datetime.now(UTC),
        )

    def latest(self, base: str, symbols: list[str]) -> ExchangeRates:
        normalized_base = normalize_currency(base)
        normalized_symbols = [code for value in symbols if (code := normalize_currency(value))]
        if not normalized_base or not normalized_symbols:
            raise HttpClientError("invalid_currency", "Currency code is missing or invalid")
        payload = self.http.get_json(
            self.endpoint,
            params={"from": normalized_base, "to": ",".join(normalized_symbols)},
            cache_ttl=60 * 60,
        )
        return ExchangeRates(
            base=normalized_base,
            rates={str(key): float(value) for key, value in (payload.get("rates") or {}).items()},
            date=str(payload.get("date") or ""),
            fetched_at=datetime.now(UTC),
        )
