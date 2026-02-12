import sys
import os
import re
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

# =========================
# CONFIGURATION
# =========================

BOOK_SLUG = "leviathan"
BOOK_NUMBER = 1
BOOK_TITLE_SLUG = "of_man"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# =========================
# HELPERS
# =========================

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def normalize_markdown(text: str) -> str:
    # Normaliser fins de lignes
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Supprimer BOM éventuel
    text = text.lstrip("\ufeff")

    # Remplacer séparateurs Unicode (Word / RTF)
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")

    # Forcer espace après les # : "#Title" -> "# Title"
    text = re.sub(r"^(#{1,6})(\S)", r"\1 \2", text, flags=re.MULTILINE)

    # Supprimer indentation avant les headers
    text = re.sub(r"^\s+(#{1,6}\s+)", r"\1", text, flags=re.MULTILINE)

    return text

# =========================
# MAIN
# =========================

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 split.py <input_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    markdown_text = normalize_markdown(raw_text)

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "chapter_title"),
            ("###", "section_title"),
        ]
    )

    docs = splitter.split_text(markdown_text)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chapter_index = 0
    section_index = 0
    last_chapter_raw = None

    for doc in docs:
        meta = doc.metadata

        chapter_raw = meta.get("chapter", "")
        chapter_title_raw = meta.get("chapter_title", "")
        section_title_raw = meta.get("section_title", "no_title")

        # Nouveau chapitre → incrément
        if chapter_raw != last_chapter_raw:
            chapter_index += 1
            section_index = 0
            last_chapter_raw = chapter_raw

        section_index += 1

        # Slugs
        chapter_title = slugify(chapter_title_raw) or "no_title"
        section_title = slugify(section_title_raw) or "no_title"

        # Numéros formatés
        book_num = f"book{BOOK_NUMBER:02d}"
        ch_num = f"ch_{chapter_index:02d}"
        sec_num = f"s{section_index:02d}"

        filename = (
            f"{BOOK_SLUG}_"
            f"{book_num}_"
            f"{BOOK_TITLE_SLUG}_"
            f"{ch_num}_"
            f"{chapter_title}_"
            f"{sec_num}_"
            f"{section_title}.txt"
        )

        filepath = OUTPUT_DIR / filename

        with open(filepath, "w", encoding="utf-8") as out:
            out.write(doc.page_content.strip())

    print(f"✅ Done — {len(docs)} files written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
