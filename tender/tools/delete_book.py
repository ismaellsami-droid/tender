#!/usr/bin/env python3
"""delete_book.py — Cleanly remove a book and all its associated data.

Deletes:
  1. OpenAI vector store files  (individual files inside the store)
  2. OpenAI vector store        (the store itself)
  3. books/<book_id>/           (source, output, toc, tests, build state…)

The corpus registry is dynamic (reads books/ at runtime), so deleting the
book directory is sufficient to deregister the corpus.

Usage:
    # Dry-run — show what would be deleted
    python tender/tools/delete_book.py --book-id thomas_hobbes-leviathan_book_01 --dry-run

    # Full delete (prompts for confirmation)
    python tender/tools/delete_book.py --book-id thomas_hobbes-leviathan_book_01

    # Skip confirmation prompt
    python tender/tools/delete_book.py --book-id thomas_hobbes-leviathan_book_01 --yes
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _delete_openai_resources(vs_id: str, dry_run: bool) -> None:
    from openai import OpenAI
    client = OpenAI()

    print(f"  Listing files in vector store {vs_id}...")
    file_ids: list[str] = []
    try:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=100)
        while True:
            file_ids.extend(f.id for f in page.data)
            if not page.has_next_page():
                break
            page = page.get_next_page()
        print(f"  Found {len(file_ids)} file(s)")
    except Exception as e:
        print(f"  ⚠️  Could not list vector store files: {e}")

    if dry_run:
        print(f"  [dry-run] Would delete vector store {vs_id}")
        print(f"  [dry-run] Would delete {len(file_ids)} OpenAI file(s)")
        return

    try:
        client.vector_stores.delete(vs_id)
        print(f"  ✓ Vector store deleted")
    except Exception as e:
        print(f"  ⚠️  Could not delete vector store: {e}")

    deleted, errors = 0, 0
    for fid in file_ids:
        try:
            client.files.delete(fid)
            deleted += 1
        except Exception:
            errors += 1
    if file_ids:
        print(f"  ✓ {deleted} file(s) deleted" + (f" ({errors} errors)" if errors else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove a book and all its associated data.",
    )
    parser.add_argument("--book-id", required=True, help="Book ID (matches books/<book-id>/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without doing anything")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    book_dir = ROOT / "books" / args.book_id

    if not book_dir.exists():
        print(f"❌ Book directory not found: {book_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = _read_json(book_dir / "book.json")
    label = cfg.get("label", args.book_id)

    state_path = book_dir / "build_state.json"
    vs_id: str = ""
    if state_path.exists():
        state = _read_json(state_path)
        vs_id = (state.get("index") or {}).get("upload", {}).get("vector_store_id", "")

    tag = "[dry-run] " if args.dry_run else ""
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Deletion plan for: {label}")
    print(f"  {tag}OpenAI vector store : {vs_id if vs_id else '(none)'}")
    print(f"  {tag}Book directory      : books/{args.book_id}/")

    if args.dry_run:
        print("\n✅ Dry-run complete. No files were deleted.")
        return

    if not args.yes:
        print()
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    print()
    if vs_id:
        _delete_openai_resources(vs_id, dry_run=False)

    shutil.rmtree(book_dir)
    print(f"  ✓ books/{args.book_id}/ deleted")
    print(f"\n✅ {label} fully removed.")


if __name__ == "__main__":
    main()
