"""Location normalization — disambiguates city groups into specific airports."""

# Airport distinctions for city groups
CITY_GROUP: dict[str, list[str]] = {
    "上海": ["PVG", "SHA"], "shanghai": ["PVG", "SHA"],
    "纽约": ["JFK", "EWR", "LGA"], "new york": ["JFK", "EWR", "LGA"], "nyc": ["JFK", "EWR", "LGA"],
    "华盛顿": ["IAD", "DCA", "BWI"], "washington": ["IAD", "DCA", "BWI"], "dc": ["IAD", "DCA", "BWI"],
    "北京": ["PEK", "PKX"], "beijing": ["PEK", "PKX"],
    "芝加哥": ["ORD", "MDW"], "chicago": ["ORD", "MDW"],
}

# Specific airport name → IATA code
SPECIFIC_AIRPORT: dict[str, str] = {
    "浦东": "PVG", "pudong": "PVG", "上海浦东": "PVG", "shanghai pudong": "PVG",
    "虹桥": "SHA", "hongqiao": "SHA", "上海虹桥": "SHA", "shanghai hongqiao": "SHA",
    "肯尼迪": "JFK", "jfk": "JFK",
    "纽瓦克": "EWR", "newark": "EWR",
    "拉瓜迪亚": "LGA", "laguardia": "LGA",
    "杜勒斯": "IAD", "dulles": "IAD",
    "里根": "DCA", "reagan": "DCA",
    "巴尔的摩": "BWI", "baltimore": "BWI",
    "首都": "PEK", "capital": "PEK",
    "大兴": "PKX", "daxing": "PKX",
    "奥黑尔": "ORD", "ohare": "ORD",
}


class LocationNormalizationService:
    def resolve_city_group(self, text: str) -> list[str]:
        """Get all airports in a city group. '上海' → ['PVG', 'SHA']."""
        key = text.strip().lower()
        return CITY_GROUP.get(key, [text.upper()] if len(text) == 3 else [])

    def resolve_specific_airport(self, text: str) -> str | None:
        """Resolve specific airport name to IATA code. '浦东' → 'PVG'."""
        key = text.strip().lower()
        return SPECIFIC_AIRPORT.get(key)

    def disambiguate(self, text: str) -> dict:
        """Parse text for airport/city mentions with disambiguation.

        Returns dict with:
        - allow: list of airport codes to allow
        - exclude: list of airport codes to exclude
        - unresolved: list of texts that couldn't be resolved
        """
        result = {"allow": [], "exclude": [], "unresolved": []}

        # Check for specific airport mentions: 浦东, 虹桥, JFK, SHA, etc.
        text_lower = text.lower()

        # "但不去X" / "不去X" → exclude (stop at punctuation)
        import re
        exclude_patterns = [
            r"(?:但|但是|可是|不过|就是)\s*不[要去能]\s*([^\s，。,、不要别]{1,8})",
            r"不[要去能]\s*([^\s，。,、不要别]{1,8})",
        ]
        for pat in exclude_patterns:
            for m in re.findall(pat, text):
                code = self.resolve_specific_airport(m)
                if code:
                    result["exclude"].append(code)
                else:
                    codes = self.resolve_city_group(m)
                    if codes:
                        result["exclude"].extend(codes)
                    else:
                        result["unresolved"].append(m)

        # "去X" / "可以从X" / "X可以" → allow (stop at punctuation)
        allow_patterns = [
            r"(?:可以|能|要)[去从]+\s*([^\s，。,、但不要别]{1,8})",
        ]
        for pat in allow_patterns:
            for m in re.findall(pat, text):
                code = self.resolve_specific_airport(m)
                if code:
                    result["allow"].append(code)
                else:
                    codes = self.resolve_city_group(m)
                    if codes:
                        # If a specific airport was excluded from the same city,
                        # only allow the non-excluded ones
                        result["allow"].extend([c for c in codes if c not in result["exclude"]])

        return result
