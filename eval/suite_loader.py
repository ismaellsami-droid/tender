from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "books"


def book_dir(book_id: str) -> Path:
    return BOOKS_DIR / book_id


def tests_dir(book_id: str) -> Path:
    return book_dir(book_id) / "tests"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite(path: Path) -> Dict[str, Any]:
    try:
        suite = read_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, col {exc.colno}: {exc.msg}"
        ) from exc

    if "tests" not in suite or not isinstance(suite["tests"], list):
        raise ValueError("Suite JSON must contain a top-level 'tests' list.")
    if "suite_id" not in suite or not isinstance(suite["suite_id"], str):
        suite["suite_id"] = suite.get("id") if isinstance(suite.get("id"), str) else path.stem
    return suite


def resolve_suite_path(
    *,
    book_id: str,
    suite: Optional[str] = None,
    test_file: Optional[str] = None,
) -> Path:
    if test_file:
        return Path(test_file)

    manifest_path = tests_dir(book_id) / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    suite_name = suite or manifest.get("default_suite") or "work"
    return tests_dir(book_id) / f"suite_{suite_name}.json"
