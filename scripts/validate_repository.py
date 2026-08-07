#!/usr/bin/env python3
"""Validate the teaching knowledge base without requiring accelerator hardware."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "AGENTS.md",
    "README.md",
    "docs/AI_CONTEXT.md",
    "docs/AI_TASK_RECIPES.md",
    "docs/KNOWLEDGE_INDEX.md",
    "docs/OPERATOR_DEVELOPMENT_PLAYBOOK.md",
    "docs/TASK_CATALOG.md",
    "templates/triton-operator/TASK.md",
    "templates/triton-operator/kernel.py",
]
LINK_ROOTS = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs",
    ROOT / "materials" / "README.md",
    ROOT / "workspaces" / "README.md",
]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def teaching_markdown() -> list[Path]:
    files: list[Path] = []
    for entry in LINK_ROOTS:
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            files.extend(entry.rglob("*.md"))
    files.extend((ROOT / "modules").glob("*/README.md"))
    files.extend((ROOT / "modules").glob("*/exercises/*.md"))
    files.extend((ROOT / "templates").rglob("*.md"))
    return sorted(set(files))


def check_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_links(errors: list[str]) -> None:
    for markdown in teaching_markdown():
        text = markdown.read_text(encoding="utf-8", errors="replace")
        for match in LINK_RE.finditer(text):
            raw = match.group(1).strip().split("#", 1)[0]
            if not raw or re.match(r"(?:https?|mailto):", raw):
                continue
            target = (markdown.parent / unquote(raw)).resolve()
            if not target.exists():
                relative = markdown.relative_to(ROOT)
                errors.append(f"broken local link in {relative}: {raw}")


def check_python_syntax(errors: list[str]) -> None:
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", "tmp", "__pycache__"} for part in path.parts):
            continue
        try:
            compile(path.read_bytes(), str(path), "exec")
        except (SyntaxError, ValueError) as exc:
            errors.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")


def check_nested_metadata(errors: list[str]) -> None:
    for path in ROOT.rglob(".git"):
        if path == ROOT / ".git":
            continue
        errors.append(f"nested .git metadata: {path.relative_to(ROOT)}")
    for path in ROOT.rglob("*.pyc"):
        errors.append(f"Python cache committed to workspace: {path.relative_to(ROOT)}")

    result = subprocess.run(
        ["git", "ls-files", "-s"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("160000 "):
                errors.append(f"gitlink/submodule remains: {line.rsplit(chr(9), 1)[-1]}")


def verify_manifest(relative: str, errors: list[str]) -> int:
    manifest = ROOT / relative
    if not manifest.is_file():
        errors.append(f"missing manifest: {relative}")
        return 0
    count = 0
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, filename = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed manifest line {relative}:{number}")
            continue
        path = ROOT / filename
        if not path.is_file():
            errors.append(f"manifest target missing: {filename}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {filename}")
        count += 1
    return count


def check_workspaces(errors: list[str]) -> None:
    root = ROOT / "workspaces"
    if not root.is_dir():
        errors.append("missing workspaces directory")
        return
    required = {"TASK.md", "kernel.py", "test_operator.py", "benchmark.py", "REPORT.md"}
    for task in root.iterdir():
        if not task.is_dir():
            continue
        missing = sorted(name for name in required if not (task / name).is_file())
        if missing:
            errors.append(f"workspace {task.name} missing: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also enforce hashes for imported source snapshots",
    )
    args = parser.parse_args()

    errors: list[str] = []
    check_required(errors)
    check_links(errors)
    check_python_syntax(errors)
    check_nested_metadata(errors)
    check_workspaces(errors)
    course_count = verify_manifest("records/courseware.sha256", errors)
    source_count = 0
    if args.strict:
        source_count = verify_manifest("records/source_manifest.sha256", errors)

    mode = "strict" if args.strict else "development"
    print(f"mode: {mode}")
    print(f"teaching Markdown: {len(teaching_markdown())}")
    print(f"course files verified: {course_count}")
    if args.strict:
        print(f"source snapshot files verified: {source_count}")
    if errors:
        print(f"FAILED: {len(errors)} problem(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("OK: structure, links, syntax, metadata and selected checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
