from __future__ import annotations

import pytest

from tests.helpers_v03 import make_v03_session


FORBIDDEN = [
    "[debug]",
    "ToolRequest",
    "ToolResult",
    "Contract(",
    "schema update",
    "chain of thought",
    "internal",
    "traceback",
    "route_semantics",
    "LLM prompt",
    "raw JSON",
]


def assert_clean(text: str) -> None:
    lowered = text.casefold()
    for marker in FORBIDDEN:
        assert marker.casefold() not in lowered


@pytest.mark.asyncio
async def test_normal_route_tool_itinerary_cost_and_failure_outputs_never_leak_internals():
    chat = make_v03_session()
    results = [
        await chat.handle_user_message("我想从成都飞奥斯丁"),
        await chat.handle_user_message("目的地天气怎么样？"),
        await chat.handle_user_message("100美元是多少人民币？"),
        await chat.handle_user_message("帮我安排三天行程"),
        await chat.handle_user_message("估算一下预算"),
    ]
    failed = make_v03_session(fail=True)
    await failed.handle_user_message("我想从成都飞奥斯丁")
    results.append(await failed.handle_user_message("目的地天气怎么样？"))
    for result in results:
        assert_clean(result.message)
        assert result.debug_summary == ""
        assert result.user_response is not None


@pytest.mark.asyncio
async def test_debug_is_separate_from_clean_user_response():
    chat = make_v03_session(debug=True)
    await chat.handle_user_message("我想从成都飞奥斯丁")
    result = await chat.handle_user_message("目的地天气怎么样？")
    assert_clean(result.message)
    assert "[debug]" in result.debug_summary
    assert "tool_result:" in result.debug_summary
