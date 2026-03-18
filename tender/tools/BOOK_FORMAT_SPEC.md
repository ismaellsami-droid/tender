# Tenber — Book Format Specification

Reference document for preparing a new book for the Tenber pipeline.

---

## 1. Folder structure

Create a folder `books/<book_id>/` with the following layout:

```
books/<book_id>/
│
│  ── PROVIDED (you fill in before running the pipeline) ──────
├── book.json                ← book config
├── source/
│   └── book.txt             ← canonical Markdown source
├── glossary_terms.json      ← glossary (your terms)
├── glossary_graph.json      ← glossary graph (your graph)
├── summary.txt              ← 2-4 paragraph book summary
└── questions.json           ← top N central questions
│
│  ── GENERATED (produced by build_book.py) ───────────────────
├── build_state.json         ← pipeline progress tracker
├── output/                  ← per-section .txt files
└── toc.json                 ← table of contents
```

Use `books/_template/` as a starting point — copy it and fill in.

---

## 2. book.json

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

| Field | Description |
|---|---|
| `book_id` | Unique identifier, matches the folder name |
| `book_slug` | Short slug for the work (e.g. `leviathan`) |
| `book_number` | Part number as integer |
| `book_title_slug` | Slug of the book part title (e.g. `of_commonwealth`) |
| `label` | Human-readable label shown in the UI |
| `reference_label` | Short citation label (e.g. `Leviathan II`) |
| `model` | OpenAI model for the retrieval/answer pipeline |
| `retrieval_k` | Number of passages retrieved (default 20) |
| `selection_k` | Number of passages selected for the LLM (default 8) |

---

## 3. source/book.txt — Canonical Markdown format

The source file must follow this exact structure:

```markdown
# Book Title

## Chapter 1 — Chapter Title

### §1 — Section Title

{text of section one}

### §2 — Another Section Title

{text of section two}

## Chapter 2 — Second Chapter Title

### §1 — No Title

{text...}
```

### Rules

| Level | Format | Example |
|---|---|---|
| H1 | `# {Book Title}` | `# Leviathan` |
| H2 | `## Chapter {N} — {Title}` | `## Chapter 1 — Of Sense` |
| H3 | `### §{N} — {Title}` | `### §1 — No Title` |

- **H1**: one per file, book title only (not used for splitting)
- **H2**: each chapter on its own line, strict format with `—` (em dash, en dash, or hyphen accepted)
- **H3**: each section on its own line; use `§N — No Title` when the original has no section title
- **No H4+**
- Section content starts on the line after the H3 header

### Output filename convention (generated automatically)

```
{book_slug}_book{N:02d}_{book_title_slug}_ch_{ch:02d}_{ch_slug}_s{sec:02d}_{sec_slug}.txt
```

Example: `leviathan_book01_of_man_ch_01_of_sense_s01_no_title.txt`

---

## 4. glossary_terms.json

A JSON array of term objects:

```json
[
  {
    "term": "Absurdity",
    "quote": "Words whereby we conceive nothing but the sound, are those we call Absurd.",
    "chapter_ref": "Chapter 5, Of Reason and Science, Section: Of Error and Absurdity",
    "importance": "High",
    "aliases": ["Absurditie", "Absurd"],
    "frequency": 14,
    "chapters": [2, 4, 5, 8, 12, 14]
  }
]
```

| Field | Required | Description |
|---|---|---|
| `term` | yes | Canonical term name (≤ 3 words) |
| `quote` | yes | Defining or representative sentence from the text |
| `chapter_ref` | yes | Human-readable source location |
| `importance` | yes | `"High"`, `"Medium"`, or `"Low"` |
| `aliases` | yes | Alternate spellings or forms (can be `[]`) |
| `frequency` | yes | Approximate occurrence count |
| `chapters` | yes | Chapter numbers where the term appears |

**Note:** Only terms with ≤ 3 words.

---

## 5. summary.txt

Plain text, 2-4 paragraphs. No Markdown formatting required.

```
This book explores the nature of man as a rational being driven by desire and fear...

The central argument develops through twelve chapters, starting from the mechanics
of sensation and proceeding to language, reason, and the passions...
```

---

## 6. questions.json

A JSON array of strings — the most important questions a reader might ask about this book:

```json
[
  "What is the relationship between sense and imagination according to Hobbes?",
  "How does Hobbes define reason and what distinguishes it from prudence?",
  "What role do the passions play in human motivation?",
  "Why does Hobbes reject the notion of objective good and evil?",
  "What is the significance of language for Hobbes's philosophy?"
]
```

Aim for 5-15 questions covering the main themes and arguments.

---

## 7. Running the pipeline

```bash
# Stage 1: split, build TOC, upload to OpenAI, register corpus (~1 min)
python tender/tools/build_book.py index --book-id <book_id>

# Force redo from a step
python tender/tools/build_book.py index --book-id <book_id> --from-step upload
```

Available `--from-step` values:
- **index**: `split`, `toc`, `upload`, `register`
