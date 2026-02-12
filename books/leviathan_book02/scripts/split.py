import sys
import os
import re
import html
from html.parser import HTMLParser
from typing import List, Dict, Optional
import hashlib
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

BOOK_SLUG = "leviathan"
BOOK_NUMBER = 2
BOOK_TITLE_SLUG = "of_common-wealth"  # FIXE: Book II = Of Common-Wealth
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# =========================
# HELPERS
# =========================

_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
MAX_SLUG_LEN = 80  # safe sur macOS

def shorten_slug(slug: str, max_len: int = MAX_SLUG_LEN) -> str:
    if len(slug) <= max_len:
        return slug

    h = hashlib.md5(slug.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:max_len-7]}_{h}"

def roman_to_int(s: str) -> Optional[int]:
    s = s.strip().upper()
    if not s or any(ch not in _ROMAN_MAP for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if total > 0 else None

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.lstrip("\ufeff")
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")
    return html.unescape(s)

def one_line(s: str) -> str:
    s = normalize_text(s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slugify_keep_hyphen(text: str) -> str:
    """
    - conserve les '-' (common-wealth)
    - espaces -> _
    - supprime le reste
    """
    t = one_line(text).lower().strip()
    t = re.sub(r"[^a-z0-9\-\s]+", "", t)  # garde a-z 0-9 - et espaces
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "no_title"

def clean_paragraph(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()

def parse_h2(h2_text: str) -> Dict[str, Optional[object]]:
    """
    h2: "CHAPTER XVII. OF THE CAUSES, ... COMMON-WEALTH"
    -> chapter_num = 17
    -> chapter_title = "OF THE CAUSES, ... COMMON-WEALTH"
    """
    t = one_line(h2_text)

    m = re.search(r"\bCHAPTER\s+([IVXLCDM]+)\b\.?", t, flags=re.IGNORECASE)
    chapter_num = roman_to_int(m.group(1)) if m else None

    # retire "CHAPTER XVII." du début
    chapter_title = re.sub(r"^\s*CHAPTER\s+[IVXLCDM]+\s*\.?\s*", "", t, flags=re.IGNORECASE).strip()
    chapter_title = chapter_title or None

    return {"chapter_num": chapter_num, "chapter_title": chapter_title}

# =========================
# PARSER
# =========================

class HobbesHTMLParser(HTMLParser):
    """
    Chapter = <div class="chapter">
    h2 définit chapitre (num + titre)
    h3 définit section
    p contenu

    Si des <p> apparaissent après h2 sans h3 -> section fake "No Title"
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)

        self.in_chapter = False
        self.div_depth = 0

        self.in_h2 = False
        self.in_h3 = False
        self.in_p = False

        self.h2_buf: List[str] = []
        self.h3_buf: List[str] = []
        self.p_buf: List[str] = []

        self.chapters: List[Dict] = []
        self.current: Optional[Dict] = None
        self.current_section: Optional[Dict] = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "div" and attrs.get("class") == "chapter":
            self.in_chapter = True
            self.div_depth = 1
            self.current = {"chapter_num": None, "chapter_title": None, "sections": []}
            self.current_section = None
            return

        if self.in_chapter and tag == "div":
            self.div_depth += 1

        if not self.in_chapter:
            return

        if tag == "h2":
            self.in_h2 = True
            self.h2_buf = []
            return

        if tag == "h3":
            self.in_h3 = True
            self.h3_buf = []
            self._start_section("No Title")  # placeholder, remplacé en fin de h3
            return

        if tag == "p":
            self.in_p = True
            self.p_buf = []
            return

        if tag == "br" and (self.in_h2 or self.in_h3):
            if self.in_h2:
                self.h2_buf.append("\n")
            else:
                self.h3_buf.append("\n")

    def handle_endtag(self, tag):
        if self.in_chapter and tag == "div":
            self.div_depth -= 1
            if self.div_depth == 0:
                self._flush_p()
                if self.current is not None:
                    self.chapters.append(self.current)
                self.current = None
                self.current_section = None
                self.in_chapter = False
            return

        if not self.in_chapter:
            return

        if tag == "h2":
            self.in_h2 = False
            info = parse_h2("".join(self.h2_buf))
            if self.current is not None:
                self.current["chapter_num"] = info["chapter_num"]
                self.current["chapter_title"] = info["chapter_title"]
            return

        if tag == "h3":
            self.in_h3 = False
            title = one_line("".join(self.h3_buf)) or "No Title"
            if self.current_section is not None:
                self.current_section["h3"] = title
            return

        if tag == "p":
            self.in_p = False
            self._flush_p()
            return

    def handle_data(self, data):
        if not self.in_chapter:
            return
        if self.in_h2:
            self.h2_buf.append(data)
        elif self.in_h3:
            self.h3_buf.append(data)
        elif self.in_p:
            self.p_buf.append(data)

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")

    def _start_section(self, h3_title: str):
        if self.current is None:
            return
        sec = {"h3": h3_title, "paras": []}
        self.current["sections"].append(sec)
        self.current_section = sec

    def _ensure_section(self):
        # <p> avant tout <h3> => section fake
        if self.current_section is None:
            self._start_section("No Title")

    def _flush_p(self):
        text = clean_paragraph("".join(self.p_buf))
        self.p_buf = []
        if not text:
            return
        self._ensure_section()
        assert self.current_section is not None
        self.current_section["paras"].append(text)

# =========================
# MAIN
# =========================

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 split.py <input_file.html>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    parser = HobbesHTMLParser()
    parser.feed(raw)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files = 0

    for ch in parser.chapters:
        chapter_num = ch.get("chapter_num")
        chapter_title = ch.get("chapter_title") or "No Title"

        if not isinstance(chapter_num, int):
            # fallback stable (au cas où)
            chapter_num = 0

        book_num = f"book{BOOK_NUMBER:02d}"
        ch_num = f"ch_{chapter_num:02d}"

        chapter_title_slug = shorten_slug(
            slugify_keep_hyphen(str(chapter_title))
        )

        section_index = 0
        for sec in ch.get("sections", []):
            h3_title = str(sec.get("h3") or "No Title")
            paras = sec.get("paras") or []
            if not isinstance(paras, list):
                continue

            content = "\n\n".join([p.strip() for p in paras if isinstance(p, str) and p.strip()]).strip()
            if not content:
                continue

            section_index += 1
            sec_num = f"s{section_index:02d}"
            section_title_slug = shorten_slug(
                slugify_keep_hyphen(h3_title)
            )
            filename = (
                f"{BOOK_SLUG}_"
                f"{book_num}_"
                f"{BOOK_TITLE_SLUG}_"
                f"{ch_num}_"
                f"{chapter_title_slug}_"
                f"{sec_num}_"
                f"{section_title_slug}.txt"
            )

            filepath = OUTPUT_DIR / filename
            with open(filepath, "w", encoding="utf-8") as out:
                out.write(content + "\n")

            total_files += 1

    print(f"✅ Done — {total_files} files written to {OUTPUT_DIR}")
    print(f"   Chapters detected: {len(parser.chapters)}")

if __name__ == "__main__":
    main()
