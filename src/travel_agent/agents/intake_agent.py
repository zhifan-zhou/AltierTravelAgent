"""Intake Agent: converts natural language query to structured travel request.

Supports:
- Chinese natural language, airport codes (NGB, PVG, JFK), English city names
- Mixed queries like "NGB飞PIT", "HK to JFK", "Shanghai to Pittsburgh", "杭州 to BOS"
- Defaults: accepts_nearby_hubs=true unless explicitly rejected
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from travel_agent.agents.base import BaseAgent
from travel_agent.core.config import get_settings
from travel_agent.models.agent_outputs import IntakeOutput
from travel_agent.models.user_request import CabinClass, DateWindow
from travel_agent.services.airport_service import AirportService


class IntakeAgent(BaseAgent[str, IntakeOutput]):
    """Parse natural language into structured UserTravelRequest.

    Uses AirportService for city/airport matching (data-driven, alias-based).
    Rule-based extraction for dates, cabin, budget, preferences.
    """

    name = "intake"

    CABIN_KEYWORDS: dict[CabinClass, list[str]] = {
        CabinClass.BUSINESS: ["商务", "商务舱", "business"],
        CabinClass.FIRST: ["头等", "头等舱", "first class", "first"],
        CabinClass.PREMIUM_ECONOMY: ["超经", "超级经济", "premium economy", "premium"],
    }

    PREF_KEYWORDS: dict[str, list[str]] = {
        "cheap": ["便宜", "省钱", "便宜点", "最低价", "特价", "cheap", "lowest", "经济实惠"],
        "comfort": ["舒适", "舒服", "comfort", "宽敞"],
        "fast": ["快", "最快", "时间短", "fast", "quick"],
        "safe": ["安全", "靠谱", "稳定", "可靠"],
        "family_friendly": ["父母", "家人", "老人", "孩子", "行李多"],
    }

    # Patterns that REJECT nearby hubs
    REJECT_NEARBY_PATTERNS: list[str] = [
        r"不要.?折腾", r"不要.?换.*城市", r"只搜.*这个.*机场", r"只要直飞",
        r"只从.*出发", r"固定.*出发", r"只飞.*直飞", r"不换.*机场",
        r"no.?transfer", r"only.?direct", r"exact.?airport",
    ]

    # Patterns that indicate split ticket acceptance
    SPLIT_KEYWORDS: list[str] = ["分开", "拆开", "接驳", "转机", "中转", "transfer"]

    # IATA code pattern: 3 uppercase letters
    IATA_PATTERN = re.compile(r'\b([A-Za-z]{3})\b')

    def __init__(self, airport_service: AirportService | None = None):
        super().__init__()
        self._airport_service = airport_service or AirportService()
        self._settings = get_settings()

    async def execute(self, query: str) -> IntakeOutput:
        q = query.strip()

        origin_text = self._extract_origin(q)
        dest_text = self._extract_destination(q)
        departure_window = self._extract_dates(q)
        cabin = self._extract_cabin(q)
        budget = self._extract_budget(q)
        preferences = self._extract_preferences(q)

        # Default: accept nearby hubs unless explicitly rejected
        rejects_nearby = self._rejects_nearby(q)
        accepts_nearby = not rejects_nearby

        # Default: accept split if nearby is accepted (unless family/safe preferences dominate)
        accepts_split = self._check_split_acceptance(q, accepts_nearby, preferences)

        self.logger.info(
            f"Intake: {origin_text} -> {dest_text}, "
            f"nearby={accepts_nearby}, split={accepts_split}, pref={preferences}"
        )

        return IntakeOutput(
            origin_text=origin_text,
            destination_text=dest_text,
            departure_window=departure_window,
            return_window=None,
            cabin=cabin,
            passengers=1,
            budget_usd=budget,
            accepts_nearby_hubs=accepts_nearby,
            accepts_split_ticket=accepts_split,
            preferences=preferences,
            raw_query=query,
        )

    # ── origin extraction ──────────────────────────────────────────────

    def _extract_origin(self, query: str) -> str:
        # 1. Airport codes: "NGB飞PIT", "NGB to PIT", "PVG-JFK"
        result = self._try_code_pattern(query, position="first")
        if result:
            return result

        # 2. Primary travel phrase: "X 飞/到 Y" — origin is X (before "从X走")
        m = re.search(r"(\S{1,20}?)\s*(?:飞|到|to)\s*(\S{1,20})", query, re.IGNORECASE)
        if m:
            origin_cand = m.group(1).strip()
            dest_cand = m.group(2).strip()
            dest_airport = self._resolve(dest_cand)
            if dest_airport:
                origin_airport = self._resolve(origin_cand)
                if origin_airport:
                    return origin_airport

        # 3. "从 X 飞/到/出发/走/去" — alternative, check AFTER primary phrase
        # Only use if we haven't already found a primary origin
        m = re.search(r"从\s*(\S{1,20}?)\s*(?:飞|到|出发|走|去)", query)
        if m:
            airport = self._resolve(m.group(1))
            if airport:
                # Make sure this isn't an alternative departure point
                # ("从X走" could mean "can also depart from X")
                # Only use as origin if no primary origin found
                return airport

        # 4. Airport codes near the beginning
        all_codes = self._find_all_iata_codes(query)
        if len(all_codes) >= 2:
            airport = self._airport_service.find_airport_by_text(all_codes[0])
            if airport:
                return airport.city_cn or airport.city

        # 5. "X to Y" (English)
        m = re.search(r"(\S{1,20}?)\s+to\s+(\S{1,20})", query, re.IGNORECASE)
        if m:
            origin_airport = self._resolve(m.group(1))
            if origin_airport:
                return origin_airport

        # 6. Scan for any known city/alias near the start
        return self._scan_city_in_text(query, prefer_first=True) or ""

    def _extract_destination(self, query: str) -> str:
        # 1. Airport codes: second code in "NGB飞PIT"
        result = self._try_code_pattern(query, position="last")
        if result:
            return result

        # 2. "飞/到/去 X"
        m = re.search(r"(?:飞往|飞到|飞|到|去)\s*(\S{1,20}?)\s*(?:，|。|$|便宜|可以|也|转|不要|越)", query)
        if m:
            airport = self._resolve(m.group(1))
            if airport:
                return airport

        # 3. "to X" (English)
        m = re.search(r"\bto\s+(\S{1,20})", query, re.IGNORECASE)
        if m:
            airport = self._resolve(m.group(1))
            if airport:
                return airport

        # 4. Airport codes — second/last one
        all_codes = self._find_all_iata_codes(query)
        if len(all_codes) >= 2:
            airport = self._airport_service.find_airport_by_text(all_codes[-1])
            if airport:
                return airport.city_cn or airport.city

        # 5. Scan for any known city/alias near the end
        return self._scan_city_in_text(query, prefer_first=False) or ""

    def _try_code_pattern(self, query: str, position: str) -> str | None:
        """Try to extract origin/dest from code-based patterns like NGB飞PIT, PVG-JFK."""
        # Pattern: XXX飞YYY, XXX到YYY, XXX-YYY, XXX to YYY
        m = re.search(
            r'\b([A-Za-z]{3})\s*(?:飞|到|->|→|-|to)\s*([A-Za-z]{3})\b',
            query, re.IGNORECASE,
        )
        if m:
            code1, code2 = m.group(1).upper(), m.group(2).upper()
            a1 = self._airport_service.get_airport(code1)
            a2 = self._airport_service.get_airport(code2)
            if a1 and a2:
                if position == "first":
                    return a1.city_cn or a1.city
                return a2.city_cn or a2.city

        # Single code + Chinese verb: "NGB飞", "飞PIT"
        if position == "first":
            m = re.search(r'\b([A-Za-z]{3})\s*(?:飞|到|出发|走)', query)
            if m:
                a = self._airport_service.get_airport(m.group(1).upper())
                if a:
                    return a.city_cn or a.city
        else:
            m = re.search(r'(?:飞|到|去)\s*([A-Za-z]{3})\b', query)
            if m:
                a = self._airport_service.get_airport(m.group(1).upper())
                if a:
                    return a.city_cn or a.city
            # "to XXX"
            m = re.search(r'\bto\s+([A-Za-z]{3})\b', query, re.IGNORECASE)
            if m:
                a = self._airport_service.get_airport(m.group(1).upper())
                if a:
                    return a.city_cn or a.city

        return None

    def _find_all_iata_codes(self, query: str) -> list[str]:
        """Find all valid 3-letter IATA codes in the query."""
        codes = []
        seen = set()
        for m in self.IATA_PATTERN.finditer(query):
            code = m.group(1).upper()
            if code in self._airport_service.all_airports and code not in seen:
                codes.append(code)
                seen.add(code)
        return codes

    def _resolve(self, text: str) -> str | None:
        """Resolve text to an airport display name (city_cn preferred)."""
        text = text.strip().rstrip("，。,.")
        if not text:
            return None
        airport = self._airport_service.find_airport_by_text(text)
        if airport:
            return airport.city_cn or airport.city
        return None

    def _scan_city_in_text(self, query: str, prefer_first: bool) -> str | None:
        """Scan entire query for known city names/aliases. Returns display name."""
        matches: list[tuple[int, str]] = []
        all_airports = self._airport_service.all_airports
        alias_map = self._airport_service.alias_map

        # Check all aliases
        for alias, code in alias_map.items():
            if len(alias) < 2:
                continue
            pos = query.lower().find(alias.lower())
            if pos >= 0:
                airport = all_airports.get(code)
                if airport:
                    name = airport.city_cn or airport.city
                    matches.append((pos, name))

        if not matches:
            return None

        matches.sort(key=lambda x: x[0])
        if prefer_first:
            return matches[0][1]
        return matches[-1][1]

    # ── dates, cabin, budget, preferences ──────────────────────────────

    def _extract_dates(self, query: str) -> DateWindow:
        today = date.today()
        settings = self._settings

        m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", query)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            result = date(today.year, month, day)
            if result < today:
                result = date(today.year + 1, month, day)
            return DateWindow(start_date=result, end_date=result + timedelta(days=3), flexible=True)

        m = re.search(r"(\d{1,2})/(\d{1,2})", query)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            result = date(today.year, month, day)
            if result < today:
                result = date(today.year + 1, month, day)
            return DateWindow(start_date=result, end_date=result + timedelta(days=3), flexible=True)

        if "下周" in query:
            d = today + timedelta(days=7)
            return DateWindow(start_date=d, end_date=d + timedelta(days=3), flexible=True)

        default_start = today + timedelta(days=settings.default_departure_days_from_now)
        return DateWindow(
            start_date=default_start,
            end_date=default_start + timedelta(days=settings.default_date_window_days),
            flexible=True,
        )

    def _extract_cabin(self, query: str) -> CabinClass:
        for cabin, keywords in self.CABIN_KEYWORDS.items():
            if any(w in query.lower() for w in keywords):
                return cabin
        return CabinClass.ECONOMY

    def _extract_budget(self, query: str) -> float | None:
        m = re.search(r"(?:预算|不超过|以内|budget)\s*\$?\s*(\d{3,6})", query, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"\$\s*(\d{3,6})", query)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d{3,6})\s*(?:美元|刀|美金|usd)", query, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    def _extract_preferences(self, query: str) -> list[str]:
        prefs = []
        for pref_name, keywords in self.PREF_KEYWORDS.items():
            if any(w in query for w in keywords):
                prefs.append(pref_name)
        return prefs

    # ── nearby / split acceptance ──────────────────────────────────────

    def _rejects_nearby(self, query: str) -> bool:
        """Check if user explicitly rejects nearby hub expansion."""
        for pat in self.REJECT_NEARBY_PATTERNS:
            if re.search(pat, query, re.IGNORECASE):
                return True
        return False

    def _check_split_acceptance(self, query: str, accepts_nearby: bool, preferences: list[str]) -> bool:
        """Determine if user accepts split ticketing."""
        # Explicit mentions
        if any(w in query for w in self.SPLIT_KEYWORDS):
            return True
        # "不要折腾" explicitly rejects complexity
        if re.search(r"不要.?折腾", query):
            return False
        # Family-friendly: cautious about split
        if "family_friendly" in preferences and "cheap" not in preferences:
            return False
        # If nearby is accepted and user wants cheap, default to allowing splits
        if accepts_nearby and "cheap" in preferences:
            return True
        # Default: follow nearby acceptance
        return accepts_nearby
