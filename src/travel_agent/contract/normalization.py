"""Location and airport normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable


CITY_AIRPORTS: dict[str, list[str]] = {
    "上海": ["PVG", "SHA"],
    "shanghai": ["PVG", "SHA"],
    "纽约": ["JFK", "EWR", "LGA"],
    "new york": ["JFK", "EWR", "LGA"],
    "nyc": ["JFK", "EWR", "LGA"],
    "华盛顿": ["IAD", "DCA", "BWI"],
    "washington": ["IAD", "DCA", "BWI"],
    "成都": ["TFU", "CTU"],
    "chengdu": ["TFU", "CTU"],
}

AIRPORT_ALIASES: dict[str, list[str]] = {
    "PVG": ["上海浦东", "浦东", "pvg", "PVG", "shanghai pudong"],
    "SHA": ["上海虹桥", "虹桥", "sha", "SHA", "shanghai hongqiao"],
    "JFK": ["jfk", "JFK"],
    "EWR": ["ewr", "EWR", "newark", "Newark", "纽瓦克"],
    "LGA": ["lga", "LGA", "laguardia", "LaGuardia", "拉瓜迪亚"],
    "WNZ": ["温州", "wenzhou", "WNZ", "wnz"],
    "TFU": ["成都天府", "天府", "tfu", "TFU", "chengdu tianfu"],
    "CTU": ["成都双流", "双流", "ctu", "CTU", "chengdu shuangliu"],
    "AUS": ["奥斯丁", "奥斯汀", "austin", "Austin", "AUS", "aus", "Austin-Bergstrom", "austin bergstrom"],
    "NGB": ["宁波", "ningbo", "NGB", "ngb"],
    "HGH": ["杭州", "hangzhou", "HGH", "hgh"],
    "PIT": ["匹兹堡", "pittsburgh", "PIT", "pit"],
    "MIA": ["迈阿密", "miami", "MIA", "mia"],
    "IAD": ["iad", "IAD", "杜勒斯", "dulles"],
    "DCA": ["dca", "DCA", "里根", "reagan"],
    "BWI": ["bwi", "BWI", "巴尔的摩"],
    "ORD": ["ord", "ORD", "芝加哥", "chicago"],
    "ATL": ["atl", "ATL", "亚特兰大", "atlanta"],
    "DFW": ["dfw", "DFW", "达拉斯", "dallas"],
    "LAX": ["lax", "LAX", "洛杉矶", "los angeles"],
    "SFO": ["sfo", "SFO", "旧金山", "san francisco"],
    "SEA": ["sea", "SEA", "西雅图", "seattle"],
    "BOS": ["bos", "BOS", "波士顿", "boston"],
    "PHL": ["phl", "PHL", "费城", "philadelphia"],
}

ALIAS_TO_AIRPORTS: dict[str, list[str]] = {}
for code, aliases in AIRPORT_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_AIRPORTS[alias.strip().lower()] = [code]
for city, codes in CITY_AIRPORTS.items():
    ALIAS_TO_AIRPORTS[city.strip().lower()] = codes


def normalize_code(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if len(value) == 3 and value.isascii():
        return value.upper()
    matches = expand_location(value)
    if len(matches) == 1:
        return matches[0]
    return None


def expand_location(value: str | None) -> list[str]:
    """Resolve a city or airport expression to airport codes."""
    if value is None:
        return []
    raw = str(value).strip()
    if not raw:
        return []
    if len(raw) == 3 and raw.isascii():
        return [raw.upper()]
    lowered = raw.lower()
    if lowered in ALIAS_TO_AIRPORTS:
        return list(ALIAS_TO_AIRPORTS[lowered])
    for alias, codes in ALIAS_TO_AIRPORTS.items():
        if alias and (alias in lowered or lowered in alias):
            return list(codes)
    return []


def normalize_airport_list(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        codes = expand_location(value) or ([str(value).upper()] if str(value).strip() else [])
        for code in codes:
            code = code.upper()
            if code not in seen:
                seen.add(code)
                result.append(code)
    return result


def dedupe(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def airport_alias_map() -> dict[str, list[str]]:
    return {key: list(value) for key, value in sorted(ALIAS_TO_AIRPORTS.items())}
