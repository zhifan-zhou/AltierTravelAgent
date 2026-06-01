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
    chat.add_argument("--show-reasoning", action="store_true", help="Show validated schema decision trace")

    search = sub.add_parser("search", help="Run one deterministic one-shot search with FakeLLM")
    search.add_argument("query")

    check = sub.add_parser("llm-check", help="Check DeepSeek configuration")
    check.add_argument("--ping", action="store_true")

    args = parser.parse_args()
    if args.command == "chat":
        asyncio.run(run_chat(debug=args.debug, show_reasoning=args.show_reasoning))
    elif args.command == "search":
        asyncio.run(run_search(args.query))
    elif args.command == "llm-check":
        asyncio.run(run_llm_check(ping=args.ping))
    else:
        parser.print_help()


async def run_chat(*, debug: bool = False, show_reasoning: bool = False) -> None:
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
        result = await _handle_with_status(session, message)
        _print_message(result.message)
        if result.update and result.update.update_type == "quit":
            break


async def run_search(query: str) -> None:
    session = LLMFirstChatSession(
        requirement_agent=DeepSeekRequirementAgent(FakeRequirementLLM()),
        logger=SFTLogger(),
    )
    result = await session.handle_user_message(query)
    print(result.message)


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
            console.print(f"[dim]Debug logs: {debug_dir / 'logs.txt'}[/dim]")
    except Exception:
        print("╭──────────────────────────────────────────────╮")
        print("│  AI 出行管家 · HubSplit Travel Demo           │")
        print("│  用自然语言描述行程，我会先理解需求，再搜索组合路线  │")
        print("╰──────────────────────────────────────────────╯")
        print()
        print(body)
        if debug_dir:
            print(f"Debug logs: {debug_dir / 'logs.txt'}")


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


async def _handle_with_status(session: LLMFirstChatSession, message: str):
    try:
        from rich.console import Console

        console = Console()
        with console.status("[cyan]正在理解你的需求...[/cyan]", spinner="dots"):
            return await session.handle_user_message(message)
    except Exception:
        return await session.handle_user_message(message)


def _print_message(message: str) -> None:
    try:
        from rich.console import Console

        Console().print(message, markup=False)
    except Exception:
        print(message)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(130)
