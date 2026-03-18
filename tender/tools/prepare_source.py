#!/usr/bin/env python3
"""prepare_source.py — Convert raw book text or PDF to canonical Tenber Markdown format.

Pipeline (the LLM never rewrites the text):
  Pass 0 — PDF extraction    : if input is .pdf, extract text via extract_pdf.py
                                  (pdfplumber with layout-aware column handling)
  Pass 1 — Artifact cleaning : remove page numbers, running headers/footers (regex)
  Pass 2 — Structure extraction: LLM reads text and returns a structure JSON
                                  (chapter numbers, titles, verbatim anchors)
  Pass 2b— Intro detection   : chapters with text before §1 get an explicit
                                  §1 — Chapter Introduction entry added
  Pass 3 — Header injection  : Python locates each anchor and inserts Markdown
                                  headers programmatically — text is verbatim
  Pass 4 — Validation        : verbatim + coverage checks

The structure JSON is saved as a side-car file so you can inspect and edit it
before re-running injection if the LLM made mistakes.

Usage:
    # From plain text
    python tender/tools/prepare_source.py \\
        --input  books/leviathan_book02/raw/source.txt \\
        --output books/leviathan_book02/source/book.txt \\
        --book-title "Leviathan" \\
        --hint "17th century philosophy, chapters numbered, sections with marginal titles"

    # From PDF (auto-detected by extension)
    python tender/tools/prepare_source.py \\
        --input  books/leviathan_book02/raw/source.pdf \\
        --output books/leviathan_book02/source/book.txt \\
        --book-title "Leviathan" \\
        --hint "17th century philosophy, chapters numbered, sections with marginal titles"

    # From PDF with explicit 2-column layout
    python tender/tools/prepare_source.py \\
        --input  books/leviathan_book02/raw/source.pdf \\
        --output books/leviathan_book02/source/book.txt \\
        --book-title "Leviathan" \\
        --layout double

    # Re-run injection only (after manually editing the structure JSON)
    python tender/tools/prepare_source.py \\
        --input  books/leviathan_book02/raw/source.pdf \\
        --output books/leviathan_book02/source/book.txt \\
        --book-title "Leviathan" \\
        --structure books/leviathan_book02/source/book_structure.json

Requires for PDF support:
    pip install pdfplumber   # recommended (layout-aware)
    pip install pymupdf      # fallback (basic extraction)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tender.tools.extract_pdf import extract_pdf

DEFAULT_MODEL = "gpt-4.1"
ANCHOR_WORDS = 12   # number of words the LLM must copy verbatim as anchor


# ─── Artifact cleaning ────────────────────────────────────────────────────────

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,8})\s*$")


def clean_artifacts(text: str) -> str:
    """Remove page numbers and repeated running headers/footers.

    Heuristics:
    - Lines that appear >= 3 times and are short (< 100 chars) → running header/footer
    - Lines that are only a number (arabic or roman) → page number
    - Normalize runs of blank lines to max 2
    """
    lines = text.split("\n")

    # Count non-empty line frequencies
    line_counts: Counter = Counter(l.strip() for l in lines if l.strip())

    # Short lines that repeat 3+ times are likely headers/footers
    repeated = {
        line for line, count in line_counts.items()
        if count >= 3 and len(line) < 100
    }

    cleaned: List[str] = []
    removed_artifacts = 0
    for line in lines:
        stripped = line.strip()
        if stripped in repeated:
            removed_artifacts += 1
            continue
        if stripped and _PAGE_NUMBER_RE.match(stripped):
            removed_artifacts += 1
            continue
        cleaned.append(line)

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    if removed_artifacts:
        print(f"  Artifact cleaning: removed {removed_artifacts} lines "
              f"(page numbers / running headers)")

    return result


# ─── Structure extraction (LLM) ───────────────────────────────────────────────

_STRUCTURE_SCHEMA = {
    "type": "object",
    "required": ["structure"],
    "additionalProperties": False,
    "properties": {
        "structure": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "number", "title", "anchor"],
                "additionalProperties": False,
                "properties": {
                    "type":   {"type": "string", "enum": ["chapter", "section"]},
                    "number": {"type": "integer", "minimum": 1},
                    "title":  {"type": "string"},
                    "anchor": {"type": "string", "minLength": 10},
                },
            },
        }
    },
}

_SYSTEM_PROMPT = """\
You are a structural analyst. Your only job is to identify the chapter and section \
boundaries of a book text and return a JSON object — you must NOT rewrite, \
paraphrase, summarize, or modify ANY part of the text.

Return a JSON object with a "structure" array. Each item must have:
  - "type": "chapter" or "section"
  - "number": integer (chapters restart from 1 per book; sections restart from 1 per chapter)
  - "title": the exact title as it appears in the text (use "No Title" if there is none)
  - "anchor": copy VERBATIM the first {anchor_words} words of the CONTENT that \
follows this chapter/section heading (not the heading itself — the content body). \
This anchor will be used to locate the insertion point in the original text, \
so it MUST be an exact substring of the source.

Rules:
- Do not skip any chapter or section.
- Sections are numbered within their chapter (§1, §2, ... reset at each chapter).
- If a section has no explicit title, use "No Title".
- The anchor must be taken from the body text, not from a heading line.
- Copy the anchor character-for-character including punctuation and capitalisation.
""".format(anchor_words=ANCHOR_WORDS)


def extract_structure_llm(
    text: str,
    book_title: str,
    hint: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Call the LLM to extract structure metadata. Returns list of structure items."""
    from openai import OpenAI
    client = OpenAI()

    user_prompt = (
        f'Book title: "{book_title}"\n'
        f'Context hint: {hint}\n\n'
        f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---\n\n"
        f"Return the structure JSON now."
    )

    print(f"  Sending {len(text):,} chars to {model} for structure extraction...")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "book_structure",
                "schema": _STRUCTURE_SCHEMA,
                "strict": True,
            }
        },
    )

    raw = resp.output_text
    data = json.loads(raw)
    structure = data["structure"]
    print(f"  LLM returned {len(structure)} structure items "
          f"({sum(1 for s in structure if s['type'] == 'chapter')} chapters, "
          f"{sum(1 for s in structure if s['type'] == 'section')} sections)")
    return structure


# ─── Anchor finding ───────────────────────────────────────────────────────────

def _normalize_ws(s: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", s).strip()


def _first_n_words(s: str, n: int) -> str:
    return " ".join(s.split()[:n])


def find_anchor_position(text: str, anchor: str, label: str) -> int:
    """Find the character position of an anchor in text.

    Tries in order:
      1. Exact match
      2. Normalized whitespace match
      3. First 6 words match (in case LLM cut the anchor slightly short/long)

    Raises ValueError with a helpful message if nothing matches.
    """
    # 1. Exact
    pos = text.find(anchor)
    if pos != -1:
        return pos

    # 2. Normalised whitespace — find in normalised text, map back to original
    norm_text = _normalize_ws(text)
    norm_anchor = _normalize_ws(anchor)
    norm_pos = norm_text.find(norm_anchor)
    if norm_pos != -1:
        # Map normalised position back: count words up to norm_pos
        words_before = len(norm_text[:norm_pos].split())
        all_words = text.split()
        if words_before < len(all_words):
            target_word = all_words[words_before]
            pos = text.find(target_word)
            if pos != -1:
                return pos

    # 3. First 6 words only
    short = _first_n_words(anchor, 6)
    pos = text.find(short)
    if pos != -1:
        return pos

    raise ValueError(
        f"Anchor not found for [{label}].\n"
        f"  Anchor: {anchor!r}\n"
        f"  Tip: edit the structure JSON and re-run with --structure."
    )


# ─── Chapter intro pre-processing ─────────────────────────────────────────────

def _preprocess_intros(
    text: str,
    structure: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inject explicit §1 — Chapter Introduction entries where needed.

    A chapter needs an intro section when:
      - it has no sections at all (entire body is intro text), OR
      - its first section anchor resolves to a different position than the
        chapter anchor (meaning there is body text before §1).

    In both cases a §1 — Chapter Introduction is prepended and all subsequent
    section numbers are incremented by one.

    Chapters whose first-section anchor coincides with the chapter anchor
    (the section starts at the very first word of the chapter) are left
    untouched.
    """
    result: List[Dict[str, Any]] = []
    intros_added = 0
    i = 0
    while i < len(structure):
        item = structure[i]
        if item["type"] != "chapter":
            result.append(item)
            i += 1
            continue

        result.append(item)

        # Collect contiguous sections for this chapter
        ch_sections: List[Dict[str, Any]] = []
        j = i + 1
        while j < len(structure) and structure[j]["type"] == "section":
            ch_sections.append(structure[j])
            j += 1

        # Decide whether intro text exists before §1
        if not ch_sections:
            needs_intro = True
        else:
            try:
                ch_pos  = find_anchor_position(text, item["anchor"],
                                               f"chapter {item['number']}")
                sec_pos = find_anchor_position(text, ch_sections[0]["anchor"],
                                               f"section {ch_sections[0]['number']}")
                needs_intro = (ch_pos != sec_pos)
            except ValueError:
                needs_intro = False

        if needs_intro:
            result.append({
                "type":   "section",
                "number": 1,
                "title":  "Chapter Introduction",
                "anchor": item["anchor"],
            })
            for sec in ch_sections:
                result.append({**sec, "number": sec["number"] + 1})
            intros_added += 1
        else:
            result.extend(ch_sections)

        i = j

    if intros_added:
        print(f"  {intros_added} chapter introduction section(s) added")
    return result


# ─── Header injection ─────────────────────────────────────────────────────────

def _make_header(item: Dict[str, Any]) -> str:
    if item["type"] == "chapter":
        return f"## Chapter {item['number']} — {item['title']}"
    elif item["type"] == "unnumbered":
        return f"## {item['title']}"
    else:
        return f"### §{item['number']} — {item['title']}"


def inject_headers(text: str, structure: List[Dict[str, Any]], book_title: str) -> str:
    """Locate each anchor in text and insert the Markdown header before it.

    The text content is never modified — only headers are inserted.
    """
    # Build list of (position, header_string)
    insertions: List[Tuple[int, str]] = []
    errors: List[str] = []

    for item in structure:
        label = f"{item['type']} {item['number']} — {item['title']}"
        try:
            pos = find_anchor_position(text, item["anchor"], label)
            insertions.append((pos, _make_header(item)))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        print("\n⚠️  Some anchors could not be located:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        print(
            "\nEdit the structure JSON file and re-run with --structure "
            "to fix these before injecting.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Sort by position descending so inserting doesn't shift earlier positions.
    # Tiebreak: sections before chapters at the same position, so that when
    # both share the same anchor (chapter start == §1 start), the chapter
    # header is inserted last and therefore ends up above the section header.
    insertions.sort(key=lambda x: (-x[0], 0 if x[1].startswith("###") else 1))

    result = text
    for pos, header in insertions:
        prefix = result[:pos]
        # Ensure header starts on its own line
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        result = prefix + header + "\n\n" + result[pos:]

    # Prepend H1 book title
    result = f"# {book_title}\n\n" + result.strip() + "\n"

    # Clean up any triple+ blank lines introduced by injection
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


# ─── Output validation ────────────────────────────────────────────────────────

_HEADER_RE = re.compile(r"^#{1,3} .+$", re.MULTILINE)


def _extract_section_bodies(output_text: str) -> List[Tuple[str, str]]:
    """Parse the injected Markdown and return [(header, body), ...] for every section."""
    parts = _HEADER_RE.split(output_text)
    headers = _HEADER_RE.findall(output_text)
    # parts[0] is text before the first header (usually empty)
    sections = []
    for header, body in zip(headers, parts[1:]):
        body = body.strip()
        if body:
            sections.append((header.strip(), body))
    return sections


def validate_output(output_text: str, cleaned_text: str) -> bool:
    """Run two checks on the injected output:

    1. Verbatim: every section body must exist word-for-word in the cleaned source.
    2. Coverage: section bodies concatenated must equal the cleaned source
                 (no text lost, no text added).

    Returns True if both checks pass, False otherwise.
    Prints a detailed report.
    """
    sections = _extract_section_bodies(output_text)
    if not sections:
        print("  ❌ No sections found in output — injection may have failed.", file=sys.stderr)
        return False

    print(f"\n[4/4] Validation ({len(sections)} sections)...")

    norm_cleaned = _normalize_ws(cleaned_text)
    verbatim_errors: List[str] = []

    # ── Check 1: verbatim ─────────────────────────────────────────────────────
    for header, body in sections:
        norm_body = _normalize_ws(body)
        if norm_body not in norm_cleaned:
            verbatim_errors.append(header)

    if verbatim_errors:
        print(f"  ❌ Verbatim check FAILED — {len(verbatim_errors)} section(s) "
              f"not found in original text:")
        for h in verbatim_errors:
            print(f"     • {h}")
    else:
        print(f"  ✓  Verbatim: all {len(sections)} section bodies found in original text")

    # ── Check 2: coverage ─────────────────────────────────────────────────────
    all_bodies = " ".join(_normalize_ws(body) for _, body in sections)
    # Normalize both sides the same way
    norm_all = _normalize_ws(all_bodies)

    # Build expected: cleaned text with no header lines (there are none in cleaned_text)
    norm_expected = norm_cleaned

    # Character counts for a quantitative view
    covered = len(norm_all)
    expected = len(norm_expected)
    ratio = covered / expected if expected else 0

    if abs(covered - expected) <= max(10, int(expected * 0.001)):
        # Within 0.1% tolerance (whitespace normalisation can introduce tiny diffs)
        print(f"  ✓  Coverage: sections cover {ratio:.1%} of the source "
              f"({covered:,} / {expected:,} chars)")
        coverage_ok = True
    else:
        gap = expected - covered
        print(f"  ❌ Coverage check FAILED — {gap:,} chars unaccounted for "
              f"({ratio:.1%} coverage)")
        print(f"     Some text may have been lost during injection or cleaning.")
        coverage_ok = False

    passed = not verbatim_errors and coverage_ok
    if passed:
        print("  ✅ All validation checks passed.")
    else:
        print("\n  Fix the issues above, edit the structure JSON, and re-run with --structure.")
    return passed


# ─── Structure report ─────────────────────────────────────────────────────────

def print_structure_report(structure: List[Dict[str, Any]]) -> None:
    print("\n  Structure found:")
    for item in structure:
        if item["type"] == "chapter":
            print(f"    Ch.{item['number']:>2}  {item['title']}")
        elif item["type"] == "unnumbered":
            print(f"    [—]    {item['title']}")
        else:
            print(f"          §{item['number']}  {item['title']}")


# ─── Structure HTML preview ───────────────────────────────────────────────────

def generate_structure_preview(
    structure: List[Dict[str, Any]],
    book_title: str,
    output_path: Path,
) -> None:
    """Write a self-contained HTML file visualising the book structure."""
    import json as _json

    ch_count  = sum(1 for s in structure if s["type"] == "chapter")
    sec_count = sum(1 for s in structure if s["type"] == "section")

    data_js = _json.dumps(
        [
            {
                "type":   item["type"],
                "number": item["number"],
                "title":  item["title"],
                "anchor": item["anchor"][:70],
                "intro":  item.get("title") == "Chapter Introduction",
            }
            for item in structure
        ],
        ensure_ascii=False,
        indent=2,
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Structure — {book_title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; margin: 0; padding: 2rem; color: #222; }}
  h1   {{ font-size: 1.1rem; color: #555; margin-bottom: 1.2rem; font-weight: 400; }}
  .chapter {{ background: #fff; border-left: 4px solid #2563eb; border-radius: 6px;
               margin: 1.2rem 0 0.3rem; padding: 0.7rem 1rem;
               box-shadow: 0 1px 3px rgba(0,0,0,.07); }}
  .chapter-label {{ font-size: .7rem; text-transform: uppercase;
                    letter-spacing: .08em; color: #2563eb; font-weight: 600; }}
  .chapter-title {{ font-size: 1rem; font-weight: 600; margin-top: .15rem; }}
  .section {{ background: #fff; border-left: 3px solid #e5e7eb; border-radius: 4px;
              margin: .25rem 0 .25rem 1.5rem; padding: .45rem 1rem;
              display: flex; align-items: baseline; gap: .7rem; }}
  .section.intro {{ border-left-color: #7c3aed; }}
  .section.orphan {{ margin-left: 0; border-left-color: #f59e0b; }}
  .sec-num   {{ font-size: .72rem; color: #9ca3af; min-width: 2.2rem; }}
  .sec-title {{ font-size: .88rem; color: #374151; }}
  .sec-title.notitle {{ color: #9ca3af; font-style: italic; }}
  .sec-title.intro-label {{ color: #7c3aed; font-style: italic; }}
  .anchor    {{ font-size: .7rem; color: #bbb; margin-left: auto;
                max-width: 45%; text-align: right;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .stats     {{ font-size: .8rem; color: #6b7280; margin-bottom: 1.2rem; }}
  .badge     {{ display: inline-block; background: #2563eb; color: #fff;
                border-radius: 3px; padding: .1rem .4rem; font-size: .7rem;
                margin-right: .4rem; }}
  .badge.sec {{ background: #6b7280; }}
</style>
</head>
<body>
<h1>{book_title}</h1>
<div class="stats">
  <span class="badge">{ch_count} chapters</span>
  <span class="badge sec">{sec_count} sections</span>
  &nbsp;{ch_count + sec_count} items total
</div>
<div id="tree"></div>
<script>
const data = {data_js};
const tree = document.getElementById('tree');
let currentChapter = null;
data.forEach(item => {{
  if (item.type === 'chapter') {{
    currentChapter = item;
    const d = document.createElement('div');
    d.className = 'chapter';
    d.innerHTML = '<div class="chapter-label">Chapter ' + item.number + '</div>'
                + '<div class="chapter-title">' + item.title + '</div>';
    tree.appendChild(d);
  }} else {{
    const d = document.createElement('div');
    const orphan = currentChapter === null ? ' orphan' : '';
    const intro  = item.intro ? ' intro' : '';
    d.className = 'section' + orphan + intro;
    const tc = item.intro ? 'sec-title intro-label'
             : item.title === 'No Title' ? 'sec-title notitle'
             : 'sec-title';
    d.innerHTML = '<span class="sec-num">§' + item.number + '</span>'
                + '<span class="' + tc + '">' + item.title + '</span>'
                + '<span class="anchor">' + item.anchor + '…</span>';
    tree.appendChild(d);
  }}
}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    print(f"  Preview → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert raw book text to canonical Tenber Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full pipeline\n"
            "  python tender/tools/prepare_source.py \\\n"
            "      --input  books/leviathan_book02/raw/source.txt \\\n"
            "      --output books/leviathan_book02/source/book.txt \\\n"
            "      --book-title 'Leviathan' \\\n"
            "      --hint '17th century philosophy, sections with marginal titles'\n\n"
            "  # Inject only (after editing the structure JSON)\n"
            "  python tender/tools/prepare_source.py \\\n"
            "      --input  books/leviathan_book02/raw/source.txt \\\n"
            "      --output books/leviathan_book02/source/book.txt \\\n"
            "      --book-title 'Leviathan' \\\n"
            "      --structure books/leviathan_book02/source/book_structure.json\n"
        ),
    )
    p.add_argument("--input",       required=True,  type=Path, help="Raw source text file")
    p.add_argument("--output",      required=True,  type=Path, help="Output canonical Markdown file")
    p.add_argument("--book-title",  required=True,             help="Book title for the H1 header")
    p.add_argument("--hint",        default="",                help="Context hint for the LLM (author, period, structure...)")
    p.add_argument("--model",       default=DEFAULT_MODEL,     help=f"OpenAI model for structure extraction (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--structure", type=Path, default=None,
        help="Path to an existing structure JSON — skip LLM extraction and inject directly",
    )
    p.add_argument(
        "--extract-only", action="store_true",
        help="Only run artifact cleaning + LLM extraction; do not inject headers",
    )
    p.add_argument(
        "--layout", default="auto", choices=["auto", "single", "double"],
        help="PDF column layout hint: auto (default), single, or double. "
             "Only used when --input is a .pdf file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Structure JSON lives next to the output file
    structure_path = args.output.parent / (args.output.stem + "_structure.json")

    # ── Pass 0: PDF extraction (if needed) ───────────────────────────────────
    if args.input.suffix.lower() == ".pdf":
        print(f"\n[0/4] PDF text extraction (pdfplumber, layout={args.layout})...")
        raw_text = extract_pdf(args.input, layout=args.layout)
        print(f"  {len(raw_text):,} chars extracted")
    else:
        raw_text = args.input.read_text(encoding="utf-8")

    # ── Pass 1: Artifact cleaning ─────────────────────────────────────────────
    print("\n[1/4] Artifact cleaning...")
    cleaned_text = clean_artifacts(raw_text)
    print(f"  Input:   {len(raw_text):,} chars")
    print(f"  Cleaned: {len(cleaned_text):,} chars")

    # ── Pass 2: Structure extraction ──────────────────────────────────────────
    if args.structure:
        print(f"\n[2/4] Loading structure from {args.structure}...")
        structure = json.loads(args.structure.read_text(encoding="utf-8"))
        print(f"  {len(structure)} items loaded")
    else:
        print(f"\n[2/4] Structure extraction via LLM ({args.model})...")
        structure = extract_structure_llm(
            cleaned_text,
            book_title=args.book_title,
            hint=args.hint,
            model=args.model,
        )
        print_structure_report(structure)

        # Save structure JSON for review / re-use
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        structure_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  Structure saved → {structure_path}")
        print(f"  ⚠️  Review this file before accepting the result.")

    if args.extract_only:
        print("\n✅ extract-only mode — injection skipped.")
        print(f"   Edit {structure_path} if needed, then re-run with --structure.")
        return

    # ── Pass 2b: Chapter intro detection ──────────────────────────────────────
    structure = _preprocess_intros(cleaned_text, structure)

    # ── Pass 3: Header injection ───────────────────────────────────────────────
    print(f"\n[3/4] Injecting headers...")
    output_text = inject_headers(cleaned_text, structure, book_title=args.book_title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")

    ch_count = len(re.findall(r"^## Chapter", output_text, re.MULTILINE))
    sec_count = len(re.findall(r"^### §", output_text, re.MULTILINE))
    print(f"  {ch_count} chapter headers, {sec_count} section headers injected")
    print(f"  Output: {len(output_text):,} chars → {args.output}")

    # ── Pass 4: Validation ────────────────────────────────────────────────────
    passed = validate_output(output_text, cleaned_text)

    if passed:
        preview_path = args.output.parent / "structure_preview.html"
        generate_structure_preview(structure, args.book_title, preview_path)
        print(f"\n✅ Done. Review {args.output} before running build_book.py index.")
    else:
        print(f"\n⚠️  Validation failed. Fix the structure JSON and re-run:")
        print(f"   --structure {structure_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
