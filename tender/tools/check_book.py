#!/usr/bin/env python3
"""
check_book.py — Final quality check for a prepared book.txt

Checks:
  [1] Artifact report    : lists lines removed by artifact cleaning (verify nothing important)
  [2] Content diff       : book.txt (headers stripped) must match cleaned source exactly
  [3] Mid-line injections: anchors that split a source line → header injected mid-paragraph

Usage:
  python tender/tools/check_book.py --book-dir books/my-book/
  python tender/tools/check_book.py --book-dir books/my-book/ --fix

  --fix  Auto-corrects mid-line anchors in book_structure.json, then re-runs
         prepare_source.py --structure to regenerate book.txt
"""

import argparse
import difflib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ── Artifact cleaning (mirrors prepare_source.py) ─────────────────────────────

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,8})\s*$")


def clean_artifacts(text: str):
    """Return (cleaned_text, removed_list). Mirrors prepare_source.clean_artifacts."""
    lines = text.split("\n")
    line_counts = Counter(l.strip() for l in lines if l.strip())
    repeated = {
        line for line, count in line_counts.items()
        if count >= 3 and len(line) < 100
    }
    cleaned, removed = [], []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in repeated:
            removed.append((i + 1, f"repeated {line_counts[stripped]}×", stripped))
            continue
        if stripped and _PAGE_NUMBER_RE.match(stripped):
            removed.append((i + 1, "page number", stripped))
            continue
        cleaned.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()
    return result, removed


def strip_headers(book_text: str) -> str:
    """Remove injected ## / ### lines from book.txt, normalize blank lines."""
    lines = [l for l in book_text.split("\n") if not re.match(r"^#{1,3} ", l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ── Check 1: Artifact report ──────────────────────────────────────────────────

def check_artifacts(raw_text: str) -> list:
    _, removed = clean_artifacts(raw_text)
    print("\n[1] Artifact cleaning report")
    if not removed:
        print("  ✅  No lines removed")
    else:
        for line_num, kind, text in removed:
            print(f"  line {line_num:>5}  [{kind}]  {repr(text)}")
    return removed


# ── Check 2: Diff ─────────────────────────────────────────────────────────────

def check_diff(book_text: str, raw_text: str) -> bool:
    cleaned_src, _ = clean_artifacts(raw_text)
    book_stripped = strip_headers(book_text)

    print("\n[2] Content diff (book.txt stripped vs cleaned source)")
    if book_stripped == cleaned_src:
        print("  ✅  Perfect match — zero diff")
        return True

    src_lines = cleaned_src.splitlines()
    book_lines = book_stripped.splitlines()
    diffs = list(difflib.unified_diff(src_lines, book_lines, lineterm="", n=3))
    changed = [d for d in diffs if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
    print(f"  ❌  {len(changed)} differing lines:")
    for d in diffs:
        print(f"    {d}")
    return False


# ── Check 3: Mid-line injections ──────────────────────────────────────────────

def check_mid_line_injections(structure: list, raw_text: str, fix: bool = False) -> list:
    """
    For each anchor in the structure, check whether it starts at the beginning
    of a line in the cleaned source. If not, the header will be injected
    mid-paragraph, producing a whitespace diff.

    With fix=True, updates each item's anchor in-place to the start of its line.
    """
    cleaned_src, _ = clean_artifacts(raw_text)
    issues = []

    for item in structure:
        anchor = item.get("anchor", "")
        if not anchor:
            continue
        pos = cleaned_src.find(anchor)
        if pos == -1:
            # Missing anchors are caught by prepare_source validation; skip here.
            continue
        if pos > 0 and cleaned_src[pos - 1] != "\n":
            # Find the start of this line
            line_start = cleaned_src.rfind("\n", 0, pos) + 1
            full_line = cleaned_src[line_start:].split("\n")[0]
            # Suggest the first 80 chars of that line as the new anchor
            suggested = full_line[:80]
            issues.append({
                "item": item,
                "old_anchor": anchor,
                "full_line": full_line,
                "suggested_anchor": suggested,
            })

    print(f"\n[3] Mid-line injection check")
    if not issues:
        print("  ✅  All anchors start at line beginnings")
    else:
        label_width = 40
        for iss in issues:
            item = iss["item"]
            if item["type"] == "section":
                label = f"§{item['number']} — {item['title']}"
            else:
                label = f"Ch.{item['number']} — {item['title']}"
            verb = "🔧 Fixed" if fix else "⚠️  Found"
            print(f"\n  {verb}: {label}")
            print(f"    old anchor : {repr(iss['old_anchor'][:70])}")
            print(f"    full line  : {repr(iss['full_line'][:70])}")
            if not fix:
                print(f"    suggestion : {repr(iss['suggested_anchor'][:70])}")

        if fix:
            for iss in issues:
                iss["item"]["anchor"] = iss["suggested_anchor"]

    return issues


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Final quality check for a prepared book.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--book-dir", type=Path, required=True,
        help="Book directory (must contain raw/source.txt, source/book.txt, source/book_structure.json)",
    )
    p.add_argument(
        "--fix", action="store_true",
        help="Auto-fix mid-line anchors in book_structure.json and re-run prepare_source.py",
    )
    args = p.parse_args()

    book_dir = args.book_dir
    raw_path       = book_dir / "raw"    / "source.txt"
    book_path      = book_dir / "source" / "book.txt"
    structure_path = book_dir / "source" / "book_structure.json"

    missing = [p for p in [raw_path, book_path, structure_path] if not p.exists()]
    if missing:
        for m in missing:
            print(f"❌ File not found: {m}")
        return 1

    raw_text  = raw_path.read_text(encoding="utf-8")
    book_text = book_path.read_text(encoding="utf-8")
    structure = json.loads(structure_path.read_text(encoding="utf-8"))

    print(f"Checking: {book_path}")
    print("─" * 60)

    check_artifacts(raw_text)
    diff_ok = check_diff(book_text, raw_text)
    issues  = check_mid_line_injections(structure, raw_text, fix=args.fix)

    # ── Apply fixes and re-run ─────────────────────────────────────────────────
    if args.fix and issues:
        structure_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Saved {len(issues)} fixed anchor(s) → {structure_path}")

        # Read book title from first line of book.txt  ("# Book Title Here")
        first_line = book_text.splitlines()[0]
        book_title = first_line[2:].strip() if first_line.startswith("# ") else ""

        print("\n  Re-running prepare_source.py --structure …")
        cmd = [
            sys.executable, "tender/tools/prepare_source.py",
            "--input",     str(raw_path),
            "--output",    str(book_path),
            "--structure", str(structure_path),
        ]
        if book_title:
            cmd += ["--book-title", book_title]

        result = subprocess.run(cmd, cwd=Path(__file__).parents[2])
        if result.returncode != 0:
            print("  ❌ prepare_source.py failed — check output above")
            return 1

        # Re-run diff to confirm clean
        book_text = book_path.read_text(encoding="utf-8")
        print("\n[2] Re-checking diff after fix …")
        diff_ok = check_diff(book_text, raw_text)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    all_ok = diff_ok and not issues
    if all_ok:
        print("✅ All checks passed — book.txt is clean.")
    else:
        problems = []
        if not diff_ok:
            problems.append("content diff mismatch")
        if issues:
            problems.append(f"{len(issues)} mid-line injection(s)")
        print(f"⚠️  Issues: {', '.join(problems)}")
        if issues and not args.fix:
            print("   → Re-run with --fix to auto-correct anchors and regenerate book.txt")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
