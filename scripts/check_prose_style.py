#!/usr/bin/env python3
"""Fail CI when public prose breaks the project's writing style guide.

Scans the site HTML and the README for the typography and vocabulary rules the
project writes to: plain ASCII punctuation (no em or en dashes as literal
characters or HTML entities), straight quotes, no filler vocabulary from the
list below, and no relative dates in place of a real one. Code regions (script,
style, pre, code) are stripped first so a dash inside a shell snippet or a class
name never trips the check.

Run locally or in CI:

    python scripts/check_prose_style.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "docs" / "site" / "index.html",
    ROOT / "docs" / "site" / "methodology.html",
    ROOT / "docs" / "site" / "demo.html",
    ROOT / "README.md",
]

BANNED_WORDS = [
    "leverage", "seamless", "delve", "underscore", "pivotal",
    "testament", "tapestry", "realm", "harness", "crucial",
]
RELATIVE_DATES = ["today", "currently"]

CODE_BLOCK = re.compile(r"<(script|style|pre|code)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]*`")
HTML_TAG = re.compile(r"<[^>]+>")


def visible_prose(path: Path) -> str:
    """Return the human-readable prose with code regions and markup removed."""
    text = path.read_text(encoding="utf-8")
    text = CODE_BLOCK.sub(" ", text)
    text = FENCED_CODE.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    return text


def check_file(path: Path) -> list[str]:
    prose = visible_prose(path)
    problems: list[str] = []

    def flag(label: str, pattern: str, *, flags: int = 0) -> None:
        for m in re.finditer(pattern, prose, flags):
            start = max(0, m.start() - 30)
            snippet = " ".join(prose[start:m.end() + 30].split())
            problems.append(f"{path.name}: {label}: ...{snippet}...")

    flag("em dash", r"—")
    flag("en dash", r"–")
    flag("em dash entity", r"&mdash;")
    flag("en dash entity", r"&ndash;")
    flag("curly quote", r"[‘’“”]")
    for w in BANNED_WORDS:
        flag(f"banned word '{w}'", rf"\b{w}\w*\b", flags=re.IGNORECASE)
    for w in RELATIVE_DATES:
        flag(f"relative date '{w}'", rf"\b{w}\b", flags=re.IGNORECASE)
    return problems


def main() -> int:
    all_problems: list[str] = []
    for path in TARGETS:
        if not path.exists():
            print(f"warning: {path} not found, skipping")
            continue
        all_problems.extend(check_file(path))

    if all_problems:
        print("Style-guide violations in public prose:\n")
        for p in all_problems:
            print("  " + p)
        print(f"\n{len(all_problems)} violation(s). The style guide asks for no em or en "
              "dashes, straight quotes, no filler vocabulary, and no relative dates.")
        return 1

    print(f"Prose style check passed: {len(TARGETS)} files clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
