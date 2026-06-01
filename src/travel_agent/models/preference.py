"""Unified preference profile metadata — single source of truth.

Used by:
- ScoringService: reads weights per profile
- ItineraryDisplayService / CLI: reads label_zh, description_zh
- FollowUpAgent: command-to-profile mapping

Add or modify profiles here; all consumers update automatically.
"""

from __future__ import annotations

# ── Profile definitions ──────────────────────────────────────────────

SCORING_PROFILES: dict[str, dict] = {
    "balanced": {
        "label_zh": "综合均衡",
        "description_zh": "综合考虑价格、风险、时间、舒适度和航司品质。",
        "weights": {
            "savings": 0.32, "comfort": 0.12, "time": 0.12,
            "risk": 0.22, "airline": 0.07, "preference": 0.15,
        },
    },
    "cheapest": {
        "label_zh": "最便宜优先",
        "description_zh": "优先低价，但会避免明显高风险方案。",
        "weights": {
            "savings": 0.45, "comfort": 0.08, "time": 0.09,
            "risk": 0.20, "airline": 0.03, "preference": 0.15,
        },
    },
    "airline_priority": {
        "label_zh": "主流航司优先",
        "description_zh": "更看重主流航司、可靠性和较低风险，价格权重略降低。",
        "weights": {
            "savings": 0.18, "comfort": 0.12, "time": 0.09,
            "risk": 0.24, "airline": 0.22, "preference": 0.15,
        },
    },
    "low_risk": {
        "label_zh": "少折腾 / 稳妥优先",
        "description_zh": "更看重少折腾、低风险、少拆分和接驳稳定性。",
        "weights": {
            "savings": 0.15, "comfort": 0.16, "time": 0.14,
            "risk": 0.32, "airline": 0.08, "preference": 0.15,
        },
    },
    "fastest": {
        "label_zh": "时间优先",
        "description_zh": "更看重总耗时和中转效率。",
        "weights": {
            "savings": 0.15, "comfort": 0.10, "time": 0.40,
            "risk": 0.18, "airline": 0.05, "preference": 0.12,
        },
    },
}

DEFAULT_PROFILE = "balanced"

# ── Helper accessors ─────────────────────────────────────────────────

def get_profile_weights(profile: str) -> dict[str, float]:
    """Get the weight dict for a profile. Falls back to balanced."""
    entry = SCORING_PROFILES.get(profile, SCORING_PROFILES[DEFAULT_PROFILE])
    return dict(entry["weights"])

def get_profile_label(profile: str) -> str:
    """Get the Chinese label for a profile."""
    entry = SCORING_PROFILES.get(profile, SCORING_PROFILES[DEFAULT_PROFILE])
    return entry.get("label_zh", profile)

def get_profile_description(profile: str) -> str:
    """Get the Chinese description for a profile."""
    entry = SCORING_PROFILES.get(profile, SCORING_PROFILES[DEFAULT_PROFILE])
    return entry.get("description_zh", "")

def get_profile_header_text(profile: str) -> str:
    """Get the full header text for display above the recommendation table."""
    label = get_profile_label(profile)
    desc = get_profile_description(profile)
    return f"当前排序偏好：{label}\n排序逻辑：{desc}"

# ── Command-to-profile mapping (used by FollowUpAgent) ───────────────

PREFERENCE_COMMANDS: dict[str, str] = {
    "主流航司优先": "airline_priority",
    "不要廉航": "airline_priority",
    "少折腾": "low_risk",
    "稳妥一点": "low_risk",
    "只要便宜": "cheapest",
    "最便宜优先": "cheapest",
    "时间短一点": "fastest",
    "时间优先": "fastest",
    "综合均衡": "balanced",
    "恢复默认": "balanced",
    "重置排序": "balanced",
    "取消偏好": "balanced",
    "undo": "balanced",
    "reset": "balanced",
    "默认排序": "balanced",
    "均衡": "balanced",
    "平衡": "balanced",
}
