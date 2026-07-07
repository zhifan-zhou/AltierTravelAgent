"""CLI for the LLM-first travel agent demo."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from travel_agent.config import load_settings
from travel_agent.llm.deepseek_client import DeepSeekClient, DeepSeekRequirementAgent
from travel_agent.llm.fake_client import FakeRequirementLLM
from travel_agent.pipeline.orchestrator import LLMFirstChatSession
from travel_agent.planning.models import UserResponse
from travel_agent.rendering.debug_renderer import DebugRenderer
from travel_agent.rendering.response_streamer import ResponseStreamer
from travel_agent.rendering.user_renderer import UserRenderer
from travel_agent.services.display_service import DisplayService
from travel_agent.services.sft_logger import SFTLogger


NO_DEEPSEEK_MESSAGE = (
    "DeepSeek 未配置，无法启动 LLM-first chat。请在 .env 中设置 DEEPSEEK_API_KEY。"
    "你也可以使用 search 命令查看静态 mock demo。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TravelAgent LLM-first demo")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Start DeepSeek LLM-first chat")
    chat.add_argument("--debug", action="store_true")
    chat.add_argument("--no-stream", action="store_true", help="Print each final response at once")
    chat.add_argument("--show-reasoning", action="store_true", help="Include decision trace in --debug output")

    search = sub.add_parser("search", help="Run one deterministic one-shot search with FakeLLM")
    search.add_argument("query")

    check = sub.add_parser("llm-check", help="Check DeepSeek configuration")
    check.add_argument("--ping", action="store_true")

    args = parser.parse_args()
    if args.command == "chat":
        asyncio.run(
            run_chat(
                debug=args.debug,
                show_reasoning=args.show_reasoning,
                stream=not args.no_stream,
            )
        )
    elif args.command == "search":
        asyncio.run(run_search(args.query))
    elif args.command == "llm-check":
        asyncio.run(run_llm_check(ping=args.ping))
    else:
        parser.print_help()


async def run_chat(
    *,
    debug: bool = False,
    show_reasoning: bool = False,
    stream: bool = True,
) -> None:
    settings = load_settings()
    if not settings.deepseek_configured:
        _print_missing_deepseek()
        return

    # Suppress INFO logs in normal mode; show in debug mode
    if not debug:
        logging.getLogger("travel_agent").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    debug_dir = settings.runs_dir / "chat" / datetime.now().strftime("%Y%m%d_%H%M%S") if debug else None
    session = LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(DeepSeekClient(settings)),
        logger=SFTLogger(),
        debug=debug,
        show_reasoning=show_reasoning,
        debug_dir=debug_dir,
    )
    _print_opening(debug_dir=debug_dir)

    while True:
        try:
            message = input("你 > ").strip()
        except EOFError:
            break
        if not message:
            continue
        result = await session.handle_user_message(message)
        if debug and result.debug_summary:
            print(DebugRenderer().render(result.debug_summary), file=sys.stderr, flush=True)
        emit_user_response(result.user_response, stream=stream)
        if result.update and result.update.update_type == "quit":
            break


async def run_search(query: str) -> None:
    session = LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(FakeRequirementLLM()),
        logger=SFTLogger(),
    )
    result = await session.handle_user_message(query)
    emit_user_response(result.user_response, stream=False)


async def run_llm_check(*, ping: bool = False) -> None:
    settings = load_settings()
    print(f".env found: {'yes' if settings.env_found else 'no'}")
    print(f"LLM_PROVIDER: {settings.llm_provider}")
    print(f"base URL: {settings.deepseek_base_url}")
    print(f"model: {settings.deepseek_model}")
    print(f"key loaded: {'yes' if bool(settings.deepseek_api_key) else 'no'}")
    print(f"masked key: {settings.masked_deepseek_key or '(none)'}")
    if ping:
        ok, detail = await DeepSeekClient(settings).ping()
        print(f"ping status: {'ok' if ok else 'failed'} ({detail})")


def _print_opening(*, debug_dir: Path | None = None) -> None:
    display = DisplayService()
    lines = display.opening_screen_text().splitlines()
    title_line = lines[0]
    subtitle_line = lines[1]
    body = "\n".join([subtitle_line, "", *lines[3:], "", display.opening_tips()])
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        panel = Panel(
            body,
            title=f"[bold white]{title_line}[/bold white]",
            border_style="cyan",
            expand=False,
        )
        console.print(panel)
        if debug_dir:
            Console(stderr=True).print(f"[dim]Debug logs: {debug_dir / 'logs.txt'}[/dim]")
    except Exception:
        print("╭──────────────────────────────────────────────╮")
        print("│  AI 出行管家 · HubSplit Travel Demo           │")
        print("│  用自然语言描述行程，我会先理解需求，再搜索组合路线  │")
        print("╰──────────────────────────────────────────────╯")
        print()
        print(body)
        if debug_dir:
            print(f"Debug logs: {debug_dir / 'logs.txt'}", file=sys.stderr)


def _print_missing_deepseek() -> None:
    body = "\n".join(
        [
            "DeepSeek 未配置，无法启动 LLM-first chat。",
            "请在 .env 中设置 DEEPSEEK_API_KEY。",
            "",
            "你也可以运行：",
            'PYTHONPATH=src python -m travel_agent.cli search "温州到匹兹堡"',
        ]
    )
    try:
        from rich.console import Console
        from rich.panel import Panel

        Console().print(Panel(body, title="DeepSeek 未配置", border_style="yellow", expand=False))
    except Exception:
        print(body)


def emit_user_response(
    response: UserResponse | None,
    *,
    stream: bool,
    output=None,
    streamer: ResponseStreamer | None = None,
) -> None:
    if response is None:
        response = UserResponse(text="本轮没有可显示的结果，请重试。", response_type="error")
    target = output or sys.stdout
    if not stream:
        print(UserRenderer().render(response), file=target, flush=True)
        return
    active_streamer = streamer or ResponseStreamer()
    try:
        for chunk in active_streamer.stream_response(response):
            print(chunk, end="", file=target, flush=True)
        print(file=target, flush=True)
    except (BrokenPipeError, KeyboardInterrupt):
        print(file=target, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(130)
