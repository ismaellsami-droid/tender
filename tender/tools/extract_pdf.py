#!/usr/bin/env python3
"""extract_pdf.py — Layout-aware PDF text extraction for Tenber.

Extracts text from a digitally-born PDF while preserving reading order,
handling multi-column layouts, encarts/marginalia, and footnotes.

Primary backend: pdfplumber (word-level bounding boxes, better column handling)
Fallback: PyMuPDF (fitz)

Usage:
    # Auto-detect layout (default)
    python tender/tools/extract_pdf.py \\
        --input  books/mybook/raw/source.pdf \\
        --output books/mybook/raw/source.txt

    # Force 2-column layout
    python tender/tools/extract_pdf.py \\
        --input  books/mybook/raw/source.pdf \\
        --output books/mybook/raw/source.txt \\
        --layout double

    # Keep marginalia, skip footnotes
    python tender/tools/extract_pdf.py \\
        --input  books/mybook/raw/source.pdf \\
        --output books/mybook/raw/source.txt \\
        --keep-marginalia --skip-footnotes
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Soft hyphen at line break: "knowl-\nedge" → "knowledge"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# Fraction of page height above which a block is considered a footnote
_FOOTNOTE_Y_THRESHOLD = 0.85

# Gap fraction of page width that triggers 2-column detection
_COLUMN_GAP_RATIO = 0.20


# ─── Column detection ─────────────────────────────────────────────────────────

def _detect_columns(words: List[dict], page_width: float) -> Optional[float]:
    """Return the x midpoint of a detected column gap, or None if single-column.

    Analyses the horizontal distribution of word x0 positions. If there is a
    gap wider than _COLUMN_GAP_RATIO * page_width between two clusters,
    the page is treated as 2-column and the gap midpoint is returned.
    """
    if not words or page_width <= 0:
        return None

    xs = sorted(w["x0"] for w in words)
    if len(xs) < 10:
        return None

    gap_threshold = _COLUMN_GAP_RATIO * page_width

    # Find the largest gap between consecutive unique x0 values
    best_gap = 0.0
    best_mid = None
    for i in range(1, len(xs)):
        gap = xs[i] - xs[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_mid = (xs[i] + xs[i - 1]) / 2

    if best_gap >= gap_threshold and best_mid is not None:
        # Sanity: mid should be somewhere in the middle third of the page
        if page_width * 0.25 < best_mid < page_width * 0.75:
            return best_mid

    return None


# ─── pdfplumber extraction ────────────────────────────────────────────────────

def _extract_page_pdfplumber(
    page,
    layout: str,
    keep_marginalia: bool,
    skip_footnotes: bool,
    main_content_ratio: float,
) -> str:
    """Extract text from a single pdfplumber page.

    Returns the page text as a string.
    """
    width = page.width
    height = page.height

    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return ""

    # ── Footnote separation ───────────────────────────────────────────────────
    footnote_y = _FOOTNOTE_Y_THRESHOLD * height
    main_words = [w for w in words if w["top"] <= footnote_y]
    foot_words = [w for w in words if w["top"] > footnote_y]

    # ── Column detection ──────────────────────────────────────────────────────
    if layout == "auto":
        col_mid = _detect_columns(main_words, width)
        is_double = col_mid is not None
    elif layout == "double":
        is_double = True
        col_mid = width / 2
    else:
        is_double = False
        col_mid = None

    # ── Main text extraction ──────────────────────────────────────────────────
    if is_double and col_mid is not None:
        # Left column
        left_bbox = (0, 0, col_mid, footnote_y)
        right_bbox = (col_mid, 0, width, footnote_y)
        left_text = page.within_bbox(left_bbox).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
        right_text = page.within_bbox(right_bbox).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
        main_text = left_text.rstrip("\n") + "\n" + right_text
    else:
        if keep_marginalia:
            bbox = (0, 0, width, footnote_y)
        else:
            # Exclude right-margin encarts
            main_w = main_content_ratio * width
            bbox = (0, 0, main_w, footnote_y)
        main_text = page.within_bbox(bbox).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""

    # ── Footnotes ─────────────────────────────────────────────────────────────
    if foot_words and not skip_footnotes:
        foot_bbox = (0, footnote_y, width, height)
        foot_text = page.within_bbox(foot_bbox).extract_text(
            x_tolerance=3, y_tolerance=3
        ) or ""
        if foot_text.strip():
            main_text = main_text.rstrip("\n") + "\n\n" + foot_text

    return main_text


def _extract_pdfplumber(
    pdf_path: Path,
    layout: str,
    keep_marginalia: bool,
    skip_footnotes: bool,
    main_content_ratio: float,
) -> str:
    import pdfplumber

    pages_text: List[str] = []
    double_count = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = _extract_page_pdfplumber(
                page,
                layout=layout,
                keep_marginalia=keep_marginalia,
                skip_footnotes=skip_footnotes,
                main_content_ratio=main_content_ratio,
            )
            # Count double-column pages for reporting
            if layout in ("double", "auto"):
                words = page.extract_words(x_tolerance=3, y_tolerance=3)
                if layout == "double" or (
                    layout == "auto"
                    and _detect_columns(words, page.width) is not None
                ):
                    double_count += 1
            pages_text.append(text)

    raw = "\n".join(pages_text)
    raw = _HYPHEN_BREAK_RE.sub(r"\1\2", raw)

    print(f"  Extracted {total} pages via pdfplumber")
    if double_count:
        print(f"  2-column layout detected on {double_count}/{total} pages")

    return raw


# ─── PyMuPDF fallback ─────────────────────────────────────────────────────────

def _extract_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages: List[str] = []
        for page in doc:
            pages.append(page.get_text("text"))  # type: ignore[attr-defined]
        doc.close()
        raw = "\n".join(pages)
        raw = _HYPHEN_BREAK_RE.sub(r"\1\2", raw)
        print(f"  Extracted {len(pages)} pages via PyMuPDF (fallback)")
        return raw

    except ImportError:
        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install pdfplumber   # recommended\n"
            "  pip install pymupdf      # fallback"
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_pdf(
    pdf_path: Path,
    layout: str = "auto",
    keep_marginalia: bool = False,
    skip_footnotes: bool = False,
    main_content_ratio: float = 0.6,
) -> str:
    """Extract text from a PDF, preserving reading order.

    Args:
        pdf_path:           Path to the PDF file.
        layout:             "auto" (detect columns per page), "single", or "double".
        keep_marginalia:    If True, include side encarts/marginalia (single-column only).
        skip_footnotes:     If True, exclude text in the bottom 15% of pages.
        main_content_ratio: Fraction of page width considered main content (default 0.6).
                            Only used in single-column mode without keep_marginalia.

    Returns:
        Extracted text as a UTF-8 string.
    """
    try:
        return _extract_pdfplumber(
            pdf_path,
            layout=layout,
            keep_marginalia=keep_marginalia,
            skip_footnotes=skip_footnotes,
            main_content_ratio=main_content_ratio,
        )
    except ImportError:
        print(
            "  ⚠️  pdfplumber not installed — falling back to PyMuPDF "
            "(no column-awareness). Install pdfplumber for better results.",
            file=sys.stderr,
        )
        return _extract_pymupdf(pdf_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layout-aware PDF text extraction for Tenber.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Auto-detect layout\n"
            "  python tender/tools/extract_pdf.py \\\n"
            "      --input  books/mybook/raw/source.pdf \\\n"
            "      --output books/mybook/raw/source.txt\n\n"
            "  # Force 2-column, skip footnotes\n"
            "  python tender/tools/extract_pdf.py \\\n"
            "      --input  books/mybook/raw/source.pdf \\\n"
            "      --output books/mybook/raw/source.txt \\\n"
            "      --layout double --skip-footnotes\n"
        ),
    )
    p.add_argument("--input",  required=True, type=Path, help="Input PDF file")
    p.add_argument("--output", required=True, type=Path, help="Output text file (.txt)")
    p.add_argument(
        "--layout", default="auto", choices=["auto", "single", "double"],
        help="Column layout hint (default: auto-detect per page)",
    )
    p.add_argument(
        "--keep-marginalia", action="store_true",
        help="Include right-margin encarts (single-column mode only)",
    )
    p.add_argument(
        "--skip-footnotes", action="store_true",
        help="Exclude text in the bottom 15%% of each page",
    )
    p.add_argument(
        "--main-content-ratio", type=float, default=0.6, metavar="RATIO",
        help="Fraction of page width considered main content (default: 0.6, "
             "used in single-column mode when --keep-marginalia is not set)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.input.suffix.lower() != ".pdf":
        print(f"❌ Input must be a PDF file (got: {args.input.suffix})", file=sys.stderr)
        sys.exit(1)

    print(f"\n[PDF extraction] {args.input.name}  layout={args.layout}")

    text = extract_pdf(
        args.input,
        layout=args.layout,
        keep_marginalia=args.keep_marginalia,
        skip_footnotes=args.skip_footnotes,
        main_content_ratio=args.main_content_ratio,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")

    print(f"  {len(text):,} chars written → {args.output}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
