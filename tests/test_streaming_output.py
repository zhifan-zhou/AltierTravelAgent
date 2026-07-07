from __future__ import annotations

from io import StringIO

from travel_agent.cli import emit_user_response
from travel_agent.planning.models import SourceRef, UserResponse
from travel_agent.rendering.response_streamer import ResponseStreamer


def test_stream_response_returns_multiple_chunks_that_join_exactly():
    response = UserResponse(text="这是一个用于验证中文流式输出完整性的较长旅行建议。" * 4)
    chunks = list(ResponseStreamer(chunk_size=12).stream_response(response))
    assert len(chunks) > 1
    assert "".join(chunks) == response.text


def test_streaming_never_adds_debug_or_raw_metadata():
    response = UserResponse(
        text="天气暂时不可用，请稍后重试。",
        response_type="error",
        sources=[SourceRef(label="天气", source="open_meteo")],
        warnings=["network unavailable"],
    )
    rendered = "".join(ResponseStreamer().stream_response(response))
    assert rendered == response.text
    assert "[debug]" not in rendered
    assert "SourceRef" not in rendered


def test_no_stream_cli_path_prints_full_text_once():
    output = StringIO()
    response = UserResponse(text="一次性输出完整响应。")
    emit_user_response(response, stream=False, output=output)
    assert output.getvalue() == response.text + "\n"


def test_stream_cli_path_preserves_api_failure_response():
    output = StringIO()
    response = UserResponse(text="无法获取实时天气，请稍后重试。", response_type="error")
    emit_user_response(
        response,
        stream=True,
        output=output,
        streamer=ResponseStreamer(chunk_size=8),
    )
    assert output.getvalue() == response.text + "\n"
