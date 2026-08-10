#!/usr/bin/env python3
"""Validate the teaching knowledge base without requiring accelerator hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
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
    "docs/OFFICIAL_OPENCOURSE.md",
    "docs/OPERATOR_DEVELOPMENT_PLAYBOOK.md",
    "docs/TASK_CATALOG.md",
    "templates/triton-operator/TASK.md",
    "templates/triton-operator/kernel.py",
]
LINK_ROOTS = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs",
    ROOT / "knowledge",
    ROOT / "materials" / "README.md",
    ROOT / "workspaces" / "README.md",
]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
COURSEWARE_SUFFIXES = {".pdf", ".docx", ".ppt", ".pptx"}


def teaching_markdown() -> list[Path]:
    files: list[Path] = []
    for entry in LINK_ROOTS:
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            files.extend(entry.rglob("*.md"))
    files.extend((ROOT / "modules").glob("*/README.md"))
    files.extend((ROOT / "modules").glob("*/exercises/*.md"))
    files.extend((ROOT / "modules").glob("*/labs/official-*/README.md"))
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


def check_knowledge_portability(errors: list[str]) -> None:
    root = ROOT / "knowledge"
    if not root.is_dir():
        return

    forbidden = (b"/Users/", b"/private/var/", b"xwechat_files", b"wxid_")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for marker in forbidden:
            if marker in data:
                errors.append(
                    f"non-portable local marker in {path.relative_to(ROOT)}: "
                    f"{marker.decode('ascii')}"
                )

    for registry in root.rglob("sources.jsonl"):
        for number, line in enumerate(
            registry.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
                source_path = Path(source["path"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                errors.append(
                    f"malformed knowledge source {registry.relative_to(ROOT)}:{number}: {exc}"
                )
                continue
            if source_path.is_absolute():
                errors.append(
                    f"absolute knowledge source path in {registry.relative_to(ROOT)}:{number}"
                )
            elif not (ROOT / source_path).is_file():
                errors.append(f"knowledge source missing: {source_path.as_posix()}")

    for index in root.rglob("rag.sqlite3"):
        try:
            connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                errors.append(
                    f"knowledge index integrity failure in {index.relative_to(ROOT)}: "
                    f"{integrity}"
                )
            for (raw_path,) in connection.execute("SELECT DISTINCT path FROM documents"):
                source_path = Path(raw_path)
                if source_path.is_absolute():
                    errors.append(
                        f"absolute source path in knowledge index {index.relative_to(ROOT)}"
                    )
                elif not (ROOT / source_path).is_file():
                    errors.append(
                        f"knowledge index source missing: {source_path.as_posix()}"
                    )
            connection.close()
        except (sqlite3.Error, OSError) as exc:
            errors.append(f"cannot inspect knowledge index {index.relative_to(ROOT)}: {exc}")


def verify_manifest(
    relative: str, errors: list[str], coverage_root: str | None = None
) -> int:
    manifest = ROOT / relative
    if not manifest.is_file():
        errors.append(f"missing manifest: {relative}")
        return 0
    count = 0
    listed: list[str] = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, filename = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed manifest line {relative}:{number}")
            continue
        if filename in listed:
            errors.append(f"duplicate manifest target: {filename}")
            continue
        listed.append(filename)
        path = ROOT / filename
        if not path.is_file():
            errors.append(f"manifest target missing: {filename}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {filename}")
        count += 1

    if coverage_root is not None:
        asset_root = ROOT / coverage_root
        assets = {
            path.relative_to(ROOT).as_posix()
            for path in asset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in COURSEWARE_SUFFIXES
        }
        listed_set = set(listed)
        for filename in sorted(assets - listed_set):
            errors.append(f"course file missing from manifest: {filename}")
        for filename in sorted(listed_set - assets):
            errors.append(f"manifest target outside course file set: {filename}")
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


def check_official_opencourse(errors: list[str]) -> tuple[int, int]:
    edition = ROOT / "records" / "official_opencourse_2026_edition.tsv"
    tree = ROOT / "records" / "official_opencourse_tree.tsv"
    edition_count = 0
    tree_count = 0

    if not edition.is_file():
        errors.append("missing official OpenCourse edition inventory")
    else:
        official_paths: set[str] = set()
        local_paths: set[str] = set()
        allowed_status = {
            "exact-local",
            "archived",
            "archived-variant",
            "archived-upstream-mislabeled",
        }
        for number, line in enumerate(
            edition.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                errors.append(f"malformed official edition row:{number}")
                continue
            expected, raw_bytes, official_path, status, local_path = fields
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                errors.append(f"invalid official edition SHA-256:{number}")
            try:
                expected_bytes = int(raw_bytes)
            except ValueError:
                errors.append(f"invalid official edition byte count:{number}")
                continue
            if status not in allowed_status:
                errors.append(f"invalid official edition status:{number}: {status}")
            if official_path in official_paths:
                errors.append(f"duplicate official edition path: {official_path}")
            if local_path in local_paths:
                errors.append(f"duplicate official edition local path: {local_path}")
            official_paths.add(official_path)
            local_paths.add(local_path)

            local = Path(local_path)
            if local.is_absolute():
                errors.append(f"absolute official edition local path:{number}")
                continue
            target = ROOT / local
            if not target.is_file():
                errors.append(f"official edition local target missing: {local_path}")
                continue
            data = target.read_bytes()
            if len(data) != expected_bytes:
                errors.append(f"official edition byte mismatch: {local_path}")
            if hashlib.sha256(data).hexdigest() != expected:
                errors.append(f"official edition checksum mismatch: {local_path}")
            edition_count += 1
        if edition_count != 24:
            errors.append(f"official edition coverage is {edition_count}, expected 24")

    if not tree.is_file():
        errors.append("missing official OpenCourse tree inventory")
    else:
        text = tree.read_text(encoding="utf-8")
        commit = "cb6b9e0a3c01d9cd31bc2187127a6f44f6626990"
        if f"# commit={commit}\n" not in text:
            errors.append("official OpenCourse tree commit marker changed")
        paths: set[str] = set()
        for number, line in enumerate(text.splitlines(), 1):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                errors.append(f"malformed official tree row:{number}")
                continue
            mode, object_type, object_id, raw_bytes, path = fields
            if not re.fullmatch(r"[0-7]{6}", mode):
                errors.append(f"invalid official tree mode:{number}")
            if object_type != "blob":
                errors.append(f"unexpected official tree object type:{number}")
            if not re.fullmatch(r"[0-9a-f]{40}", object_id):
                errors.append(f"invalid official tree object id:{number}")
            if raw_bytes != "-" and not raw_bytes.isdigit():
                errors.append(f"invalid official tree byte count:{number}")
            if path in paths:
                errors.append(f"duplicate official tree path: {path}")
            paths.add(path)
            tree_count += 1
        if tree_count != 503:
            errors.append(f"official OpenCourse tree coverage is {tree_count}, expected 503")

    return edition_count, tree_count


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
    check_knowledge_portability(errors)
    check_workspaces(errors)
    official_edition_count, official_tree_count = check_official_opencourse(errors)
    course_count = verify_manifest(
        "records/courseware.sha256", errors, coverage_root="materials"
    )
    source_count = 0
    if args.strict:
        source_count = verify_manifest("records/source_manifest.sha256", errors)

    mode = "strict" if args.strict else "development"
    print(f"mode: {mode}")
    print(f"teaching Markdown: {len(teaching_markdown())}")
    print(f"course files verified: {course_count}")
    print(f"official edition files mapped: {official_edition_count}")
    print(f"official OpenCourse tree entries: {official_tree_count}")
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
