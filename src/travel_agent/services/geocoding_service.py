"""Open-Meteo geocoding adapter."""

from __future__ import annotations

from pydantic import BaseModel

from travel_agent.tools.http_client import HttpClient


class GeoLocation(BaseModel):
    name: str
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None
    latitude: float
    longitude: float
    timezone: str | None = None
    source: str = "open_meteo_geocoding"


class OpenMeteoGeocodingClient:
    endpoint = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(self, http: HttpClient | None = None):
        self.http = http or HttpClient()

    def search_location(self, query: str, country_code: str | None = None) -> list[GeoLocation]:
        params: dict[str, object] = {"name": query, "count": 10, "language": "zh", "format": "json"}
        if country_code:
            params["countryCode"] = country_code.upper()
        payload = self.http.get_json(self.endpoint, params=params, cache_ttl=24 * 60 * 60)
        locations: list[GeoLocation] = []
        for row in payload.get("results") or []:
            try:
                locations.append(
                    GeoLocation(
                        name=str(row["name"]),
                        country=row.get("country"),
                        country_code=row.get("country_code"),
                        admin1=row.get("admin1"),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        timezone=row.get("timezone"),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return locations


def select_unambiguous_location(
    query: str,
    locations: list[GeoLocation],
    *,
    country_code: str | None = None,
) -> GeoLocation | None:
    if len(locations) == 1:
        return locations[0]
    normalized = query.strip().casefold()
    exact = [item for item in locations if item.name.casefold() == normalized]
    if len(exact) == 1:
        return exact[0]
    if exact and len({(item.country_code, item.admin1) for item in exact}) == 1:
        return exact[0]
    if country_code:
        in_country = [item for item in locations if item.country_code == country_code.upper()]
        if in_country:
            # The country hint comes from a resolved local airport/contract role.
            return in_country[0]
    return None
