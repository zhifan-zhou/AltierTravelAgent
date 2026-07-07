from __future__ import annotations

import pytest

from travel_agent.planning.models import UserResponse
from travel_agent.rendering.user_renderer import UserRenderer


def test_user_renderer_accepts_only_user_response():
    with pytest.raises(TypeError):
        UserRenderer().render({"text": "hello"})


def test_user_renderer_returns_clean_response_text():
    response = UserResponse(text="这是最终旅行建议。", response_type="general_answer")
    assert UserRenderer().render(response) == response.text


@pytest.mark.parametrize("marker", ["[debug]", "ToolRequest", "ToolResult", "Contract(", "raw JSON", "Traceback"])
def test_user_renderer_safely_hides_internal_markers(marker: str):
    rendered = UserRenderer().render(UserResponse(text=f"oops {marker} details", response_type="error"))
    assert marker.casefold() not in rendered.casefold()
    assert "已安全隐藏" in rendered
