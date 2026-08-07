#!/usr/bin/env python3
"""Create an isolated operator-development workspace from the course template."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "triton-operator"
TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".sh"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create workspaces/<name> for a new Triton operator task."
    )
    parser.add_argument("name", help="lowercase slug, for example fused-silu")
    parser.add_argument("--title", help="human-readable task title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slug = args.name.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise SystemExit("name must match [a-z0-9][a-z0-9-]*")

    title = args.title.strip() if args.title else slug.replace("-", " ").title()
    destination = ROOT / "workspaces" / slug
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing workspace: {destination}")
    if not TEMPLATE.is_dir():
        raise SystemExit(f"template directory is missing: {TEMPLATE}")

    shutil.copytree(TEMPLATE, destination)
    replacements = {
        "{{TASK_SLUG}}": slug,
        "{{TASK_NAME}}": title,
    }
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            content = content.replace(old, new)
        path.write_text(content, encoding="utf-8")

    print(f"created: {destination.relative_to(ROOT)}")
    print(f"next: edit {destination.relative_to(ROOT) / 'TASK.md'}")
    print("then ask the coding assistant to read AGENTS.md and this TASK.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
