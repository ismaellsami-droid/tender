# Tenber — Adding a New Book

## Quick start

1. Copy this folder and rename it to your `book_id`:
   ```bash
   cp -r books/_template/ books/my_book_id/
   ```

2. Drop your raw book file in `raw/` (PDF or TXT — this folder is git-ignored):
   ```
   books/my_book_id/raw/source.pdf   ← or source.txt
   ```

3. **(PDF or EPUB only)** Extract text and inspect before processing:
   ```bash
   # From PDF
   python tender/tools/extract_pdf.py \
       --input  books/my_book_id/raw/source.pdf \
       --output books/my_book_id/raw/source.txt \
       --layout auto   # or: single, double

   # From EPUB (cleaner output, recommended when available)
   python tender/tools/extract_epub.py \
       --input  books/my_book_id/raw/source.epub \
       --output books/my_book_id/raw/source.txt
   ```
   Open `raw/source.txt` and verify the reading order is correct.
   Fix manually if needed. → You now have `raw/source.txt`.

4. Convert to canonical Markdown (from `raw/source.txt` — whether it came from a PDF or was provided directly):
   ```bash
   python tender/tools/prepare_source.py \
       --input  books/my_book_id/raw/source.txt \
       --output books/my_book_id/source/book.txt \
       --book-title "My Book Title" \
       --hint "optional context hint for the LLM (period, structure...)"
   ```
   Open `source/structure_preview.html` to review the detected structure visually.
   Edit `book_structure.json` to correct titles, add/remove sections, fix anchors,
   then re-run with `--structure` to skip the LLM and re-inject directly:
   ```bash
   python tender/tools/prepare_source.py \
       --input  books/my_book_id/raw/source.txt \
       --output books/my_book_id/source/book.txt \
       --book-title "My Book Title" \
       --structure books/my_book_id/source/book_structure.json
   ```

5. Once the structure looks right, run the final quality check:
   ```bash
   python tender/tools/check_book.py --book-dir books/my_book_id/
   ```
   This verifies: (a) which artifact lines were removed, (b) zero-diff between
   `book.txt` and the original source, (c) no headers injected mid-paragraph.
   If mid-line injection issues are found, auto-fix and regenerate:
   ```bash
   python tender/tools/check_book.py --book-dir books/my_book_id/ --fix
   ```

6. Fill in the remaining files (see below).

7. Run the ingestion pipeline:
   ```bash
   # Stage 1: split, upload to OpenAI, register corpus (~1 min)
   python tender/tools/build_book.py index --book-id my_book_id
   ```

---

## Files you must provide

| File | Description |
|---|---|
| `raw/source.pdf` or `raw/source.txt` | Original book file (git-ignored) |
| `book.json` | Book config (IDs, labels, models) |
| `source/book.txt` | Full text in canonical Markdown format (output of prepare_source.py) |
| `glossary_terms.json` | Glossary of key terms (≤ 3 words each) |
| `glossary_graph.json` | Glossary graph used by anchor detection |
| `summary.txt` | 2-4 paragraph book summary |
| `questions.json` | 5-15 central questions about the book |

---

## book.json

```json
{
  "book_id":         "leviathan_book02",
  "book_slug":       "leviathan",
  "book_number":     2,
  "book_title_slug": "of_commonwealth",
  "label":           "Hobbes — Leviathan (Book II)",
  "reference_label": "Leviathan II",
  "model":           "gpt-4.1-mini",
  "retrieval_k":     20,
  "selection_k":     8
}
```

---

## source/book.txt — Canonical Markdown format

```markdown
# Book Title

## Chapter 1 — Chapter Title

### §1 — Section Title

{text of section}

### §2 — No Title

{text of section}

## Chapter 2 — Another Chapter

### §1 — Section Title

{text of section}
```

**Rules:**
- H1 = book title (one per file)
- H2 = `## Chapter {N} — {Title}` (em dash, en dash, or hyphen accepted)
- H3 = `### §{N} — {Title}` (use `No Title` if the section has no title)
- No H4+

---

## glossary_terms.json

```json
[
  {
    "term": "Absurdity",
    "quote": "Words whereby we conceive nothing but the sound are those we call Absurd.",
    "chapter_ref": "Chapter 5, Of Reason and Science, Section: Of Error and Absurdity",
    "importance": "High",
    "aliases": ["Absurditie", "Absurd"],
    "frequency": 14,
    "chapters": [2, 4, 5, 8]
  }
]
```

- `term`: canonical name, **≤ 3 words**
- `importance`: `"High"`, `"Medium"`, or `"Low"`
- `aliases`: alternate spellings or forms (can be `[]`)

---

## What the pipeline produces

```
books/my_book_id/
├── raw/                     ← your original file (git-ignored)
│   └── source.pdf
├── source/
│   ├── book.txt               ← canonical Markdown (output of prepare_source.py)
│   ├── book_structure.json    ← LLM structure side-car (editable, re-injectable)
│   └── structure_preview.html ← visual TOC preview (auto, regenerated each run)
├── build_state.json         ← pipeline progress (auto)
├── toc.json                 ← table of contents (auto)
├── output/                  ← per-section .txt files (auto)
├── glossary_terms.json      ← glossary provided by you
└── glossary_graph.json      ← graph provided by you
```

---

## From-step options

| Stage | Steps |
|---|---|
| `index` | `split` → `toc` → `upload` → `register` |

Full spec: [tender/tools/BOOK_FORMAT_SPEC.md](../../tender/tools/BOOK_FORMAT_SPEC.md)
