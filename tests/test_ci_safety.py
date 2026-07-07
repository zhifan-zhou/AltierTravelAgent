from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_runtime_paths_are_gitignored():
    for path in [".env", ".local", "runs", "logs"]:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{path} should be ignored"


def test_env_is_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_no_obvious_real_secret_in_tracked_files():
    files = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    allowed_literals = {
        "sk-1234567890abcdef",
        "sk-test12345678",
        "Bearer secret",
        "your_deepseek_api_key_here",
    }
    suspicious = []
    secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{12,}"),
        re.compile(r"Bearer\s+[A-Za-z0-9._-]{8,}"),
    ]
    for rel in files:
        path = ROOT / rel
        if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in secret_patterns:
            for match in pattern.findall(text):
                if match not in allowed_literals:
                    suspicious.append((rel, match))
    assert suspicious == []
