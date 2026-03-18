#!/usr/bin/env python3
"""extract_epub.py — Extract clean text from an EPUB file for Tenber.

EPUB is a ZIP archive containing HTML/XHTML files ordered by an OPF spine.
This script reads the spine order, strips HTML tags, and outputs clean UTF-8
text — no font encoding issues, no missing spaces.

Usage:
    python tender/tools/extract_epub.py \\
        --input  books/mybook/raw/source.epub \\
        --output books/mybook/raw/source.txt

Requires:
    pip install ebooklib beautifulsoup4
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

# Tags whose content should be skipped entirely (navigation, scripts, etc.)
_SKIP_TAGS = {"nav", "script", "style", "head"}

# Block-level tags that should produce a blank line separation
_BLOCK_TAGS = {
    "p", "div", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "tr",
}


def _html_to_text(html: str) -> str:
    """Convert an HTML/XHTML string to plain text, preserving paragraph breaks."""
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(html, "html.parser")

    # Remove skip tags entirely
    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()

    lines: List[str] = []
    current: List[str] = []

    def flush() -> None:
        text = " ".join(current).strip()
        if text:
            lines.append(text)
        current.clear()

    def walk(node) -> None:
        if isinstance(node, NavigableString):
            text = str(node)
            # Collapse whitespace within inline text
            text = re.sub(r"[\r\n\t ]+", " ", text)
            if text.strip():
                current.append(text.strip())
        elif isinstance(node, Tag):
            name = node.name.lower() if node.name else ""
            if name in _BLOCK_TAGS:
                flush()
                for child in node.children:
                    walk(child)
                flush()
            else:
                for child in node.children:
                    walk(child)

    body = soup.find("body") or soup
    for child in body.children:
        walk(child)
    flush()

    return "\n\n".join(lines)


def extract_epub(epub_path: Path) -> str:
    """Extract ordered plain text from an EPUB file.

    Reads the OPF spine to get document order, converts each HTML chapter
    to plain text, and concatenates with page separators.

    Returns:
        Full book text as a UTF-8 string.
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    # Collect items in spine order
    spine_ids = [item_id for item_id, _ in book.spine]
    spine_items = []
    for item_id in spine_ids:
        item = book.get_item_with_id(item_id)
        if item is not None:
            spine_items.append(item)

    # Fallback: if spine is empty, use all document items
    if not spine_items:
        spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    chapters: List[str] = []
    skipped = 0

    for item in spine_items:
        try:
            html = item.get_content().decode("utf-8", errors="replace")
        except Exception:
            skipped += 1
            continue

        text = _html_to_text(html).strip()
        if text:
            chapters.append(text)
        else:
            skipped += 1

    print(f"  Processed {len(chapters)} chapters ({skipped} empty/skipped)")

    full_text = "\n\n".join(chapters)

    # Normalize runs of blank lines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    return full_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract clean text from an EPUB file for Tenber.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tender/tools/extract_epub.py \\\n"
            "      --input  books/mybook/raw/source.epub \\\n"
            "      --output books/mybook/raw/source.txt\n"
        ),
    )
    p.add_argument("--input",  required=True, type=Path, help="Input EPUB file")
    p.add_argument("--output", required=True, type=Path, help="Output text file (.txt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.input.suffix.lower() != ".epub":
        print(f"❌ Input must be an EPUB file (got: {args.input.suffix})", file=sys.stderr)
        sys.exit(1)

    print(f"\n[EPUB extraction] {args.input.name}")

    text = extract_epub(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    print(f"  {len(text):,} chars written → {args.output}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
