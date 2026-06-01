"""Utility functions for date parsing and geo calculations."""

import math
import re
from datetime import date, datetime, timedelta
from typing import Optional


def parse_flexible_date(text: str) -> Optional[date]:
    """Parse common Chinese date expressions into a date object.

    Supports: '6月15号', '6/15', '2026-06-15', '下周', '下个月'
    Returns None if no date found.
    """
    if not text:
        return None

    # Try ISO format first
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue

    # Chinese format: 6月15号, 6月15日
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        today = date.today()
        result = date(today.year, month, day)
        if result < today:
            result = date(today.year + 1, month, day)
        return result

    # MM/DD format
    m = re.search(r"(\d{1,2})/(\d{1,2})", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        today = date.today()
        result = date(today.year, month, day)
        if result < today:
            result = date(today.year + 1, month, day)
        return result

    # Relative dates
    if "下周" in text:
        return date.today() + timedelta(days=7)
    if "下个月" in text or "下月" in text:
        return date.today() + timedelta(days=30)
    if "明天" in text:
        return date.today() + timedelta(days=1)
    if "后天" in text:
        return date.today() + timedelta(days=2)

    return None


def parse_city_from_text(text: str, airports: dict[str, dict]) -> Optional[str]:
    """Match user text to an airport code using city_cn, city, or code."""
    text_lower = text.strip().lower()
    for code, info in airports.items():
        if text_lower == code.lower():
            return code
        if text_lower in info.get("city_cn", "").lower() or info.get("city_cn", "").lower() in text_lower:
            return code
        if text_lower in info.get("city", "").lower():
            return code
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points using Haversine."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
