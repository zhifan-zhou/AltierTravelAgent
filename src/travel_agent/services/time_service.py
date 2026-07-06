"""Local time resolver backed by IANA timezone data and zoneinfo."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel


class LocalTimeResult(BaseModel):
    location_name: str
    timezone: str
    local_time: str
    utc_offset: str
    source: str = "python_zoneinfo"
    geocoding_source: str | None = None
    fetched_at: datetime
    is_live: bool = True


class LocalTimeService:
    def resolve(
        self,
        *,
        location_name: str,
        timezone: str,
        geocoding_source: str | None = None,
        now_utc: datetime | None = None,
    ) -> LocalTimeResult:
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone unavailable") from exc
        fetched_at = now_utc or datetime.now(UTC)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        local = fetched_at.astimezone(zone)
        return LocalTimeResult(
            location_name=location_name,
            timezone=timezone,
            local_time=local.isoformat(timespec="minutes"),
            utc_offset=local.strftime("%z")[:3] + ":" + local.strftime("%z")[3:],
            geocoding_source=geocoding_source,
            fetched_at=fetched_at,
        )
