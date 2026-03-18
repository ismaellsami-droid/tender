#!/usr/bin/env python3
"""build_book.py — Automated book pipeline for Tenber.

Usage:
    # Split source, upload to OpenAI, build TOC, register corpus (~1 min)
    python tender/tools/build_book.py index --book-id leviathan_book02

    # Force redo from a step (resets that step and all subsequent ones)
    python tender/tools/build_book.py index --book-id leviathan_book02 --from-step upload

Prerequisites (books/<book_id>/):
    book.json            — book config (see BOOK_FORMAT_SPEC.md)
    source/book.txt      — canonical Markdown source
    glossary_terms.json  — glossary (provided by user)
    glossary_graph.json  — glossary graph (provided by user)
    summary.txt          — book summary (provided by user)
    questions.json       — top N central questions (provided by user)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent   # project root
TOOLS_DIR = Path(__file__).resolve().parent            # tender/tools/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Step lists ───────────────────────────────────────────────────────────────

INDEX_STEPS = ["split", "toc", "upload", "register"]


# ─── Utilities ────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_hex(parts: List[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    data = path.read_bytes()
    return _sha256_hex([
        path.name.encode("utf-8"),
        b"\0",
        data,
    ])


def _dir_fingerprint(path: Path) -> str:
    if not path.exists() or not path.is_dir():
        return "missing"
    parts: List[bytes] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        parts.extend([rel, b"\0", child.read_bytes(), b"\0"])
    return _sha256_hex(parts or [b"empty"])


def _book_dir(book_id: str) -> Path:
    return ROOT / "books" / book_id


def _output_dir(book_id: str) -> Path:
    return _book_dir(book_id) / "output"


# ─── Config + State ───────────────────────────────────────────────────────────

_REQUIRED_FIELDS = [
    "book_id", "book_slug", "book_number", "book_title_slug",
    "label", "reference_label", "model",
]


def _load_book_config(book_id: str) -> Dict[str, Any]:
    path = _book_dir(book_id) / "book.json"
    if not path.exists():
        raise FileNotFoundError(
            f"book.json not found at {path}\n"
            f"Copy books/_template/ to books/{book_id}/ and fill it in."
        )
    cfg = _read_json(path)
    missing = [f for f in _REQUIRED_FIELDS if f not in cfg]
    if missing:
        raise ValueError(f"Missing fields in book.json: {missing}")
    return cfg


def _load_state(book_id: str) -> Dict[str, Any]:
    path = _book_dir(book_id) / "build_state.json"
    if not path.exists():
        return {"book_id": book_id, "index": {}, "graph": {}}
    return _read_json(path)


def _save_state(book_id: str, state: Dict[str, Any]) -> None:
    _write_json(_book_dir(book_id) / "build_state.json", state)


def _mark_done(state: Dict[str, Any], stage: str, step: str, **extra) -> None:
    state.setdefault(stage, {})[step] = {"status": "done", "ts": _now(), **extra}


def _is_done(state: Dict[str, Any], stage: str, step: str) -> bool:
    return state.get(stage, {}).get(step, {}).get("status") == "done"


def _step_data(state: Dict[str, Any], stage: str, step: str) -> Dict[str, Any]:
    return state.get(stage, {}).get(step, {})


def _reset_from(state: Dict[str, Any], stage: str, steps: List[str], from_idx: int) -> None:
    for step in steps[from_idx:]:
        state.setdefault(stage, {}).pop(step, None)


def _index_step_fingerprint(book_dir: Path, cfg: Dict[str, Any], step: str) -> str:
    source_path = book_dir / "source" / "book.txt"
    book_path = book_dir / "book.json"
    toc_path = book_dir / "toc.json"
    output_dir = book_dir / "output"
    upload_state = _load_state(cfg["book_id"]).get("index", {}).get("upload", {})
    vector_store_id = str(upload_state.get("vector_store_id", ""))

    if step == "split":
        payload = {
            "step": step,
            "book_json": _file_fingerprint(book_path),
            "source": _file_fingerprint(source_path),
            "output_dir": _dir_fingerprint(output_dir),
        }
    elif step == "toc":
        payload = {
            "step": step,
            "book_json": _file_fingerprint(book_path),
            "source": _file_fingerprint(source_path),
            "toc_json": _file_fingerprint(toc_path),
        }
    elif step == "upload":
        payload = {
            "step": step,
            "book_json": _file_fingerprint(book_path),
            "output_dir": _dir_fingerprint(output_dir),
        }
    elif step == "register":
        payload = {
            "step": step,
            "book_json": _file_fingerprint(book_path),
            "vector_store_id": vector_store_id,
        }
    else:
        raise ValueError(f"Unknown index step: {step}")

    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _invalidate_stale_index_steps(
    state: Dict[str, Any],
    book_dir: Path,
    cfg: Dict[str, Any],
) -> bool:
    for idx, step in enumerate(INDEX_STEPS):
        step_state = _step_data(state, "index", step)
        if step_state.get("status") != "done":
            continue
        stored_fp = str(step_state.get("fingerprint", ""))
        current_fp = _index_step_fingerprint(book_dir, cfg, step)
        if stored_fp and stored_fp == current_fp:
            continue
        print(f"  State invalidated from '{step}' — inputs or outputs changed.")
        _reset_from(state, "index", INDEX_STEPS, idx)
        return True
    return False


# ─── Split ────────────────────────────────────────────────────────────────────

_H1_RE = re.compile(r"^#\s+(.+)$")
_H2_RE = re.compile(r"^##\s+Chapter\s+(\d+)\s+[—–-]+\s+(.+)$", re.IGNORECASE)
_H3_RE = re.compile(r"^###\s+§(\d+)\s+[—–-]+\s+(.+)$", re.IGNORECASE)


def _normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", text, flags=re.MULTILINE)
    text = re.sub(r"^\s+(#{1,6}\s+)", r"\1", text, flags=re.MULTILINE)
    return text


def _parse_source(source_path: Path) -> List[Dict[str, Any]]:
    """Parse canonical Markdown into ordered section records.

    Chapter Introduction sections are already explicit in book.txt (injected by
    prepare_source.py), so this parser just maps ### §N headers to sections.
    Text between a ## Chapter header and the first ### §N is silently ignored
    (there should be none after prepare_source.py has run correctly).
    """
    raw = source_path.read_text(encoding="utf-8")
    text = _normalize_markdown(raw)
    lines = text.split("\n")

    sections: List[Dict[str, Any]] = []
    current_ch_num = 0
    current_ch_title = ""
    current_sec_num = 0
    current_sec_title = "No Title"
    current_content: List[str] = []
    in_section = False

    def flush() -> None:
        if current_ch_num == 0 or not in_section:
            return
        content = "\n".join(current_content).strip()
        if not content:
            return
        sections.append({
            "chapter": current_ch_num,
            "chapter_title": current_ch_title,
            "section": current_sec_num,
            "section_title": current_sec_title,
            "content": content,
        })

    for line in lines:
        stripped = line.strip()
        m2 = _H2_RE.match(stripped)
        m3 = _H3_RE.match(stripped)

        if _H1_RE.match(stripped):
            continue
        if m2:
            flush()
            current_ch_num = int(m2.group(1))
            current_ch_title = m2.group(2).strip()
            current_sec_num = 0
            current_content = []
            in_section = False
        elif m3:
            flush()
            current_sec_num = int(m3.group(1))
            current_sec_title = m3.group(2).strip()
            current_content = []
            in_section = True
        else:
            if in_section:
                current_content.append(line)

    flush()
    return sections


def _assign_output_filenames(sections: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Assign deterministic output filenames to parsed section records."""
    book_slug = cfg["book_slug"]
    book_number = int(cfg["book_number"])
    book_title_slug = cfg["book_title_slug"]
    book_num_str = f"book{book_number:02d}"

    seen_filenames: set[str] = set()
    for item in sections:
        ch_num = int(item["chapter"])
        ch_title = str(item["chapter_title"])
        sec_num = int(item["section"])
        sec_title = str(item["section_title"])
        ch_slug = _slugify(ch_title) or "no_title"
        sec_slug = _slugify(sec_title) or "no_title"
        filename = (
            f"{book_slug}_{book_num_str}_{book_title_slug}_"
            f"ch_{ch_num:02d}_{ch_slug}_"
            f"s{sec_num:02d}_{sec_slug}.txt"
        )
        if filename in seen_filenames:
            raise RuntimeError(
                "Filename collision during split: "
                f"{filename} (check duplicate/similar chapter or section titles)"
            )
        seen_filenames.add(filename)
        item["filename"] = filename


def _split_source(
    sections: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    output_dir: Path,
) -> int:
    """Split parsed canonical Markdown into per-section .txt files. Returns file count."""
    _assign_output_filenames(sections, cfg)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in sections:
        filename = str(item["filename"])
        content = str(item["content"])
        (output_dir / filename).write_text(content, encoding="utf-8")

    return len(sections)


# ─── TOC ──────────────────────────────────────────────────────────────────────

def _build_toc(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build table of contents from parsed section records."""
    chapters: Dict[int, Dict[str, Any]] = {}
    for item in sections:
        ch_num = int(item["chapter"])
        ch_title = str(item["chapter_title"])
        sec_num = int(item["section"])
        sec_title = str(item["section_title"])
        filename = str(item["filename"])
        if ch_num not in chapters:
            chapters[ch_num] = {"chapter": ch_num, "chapter_title": ch_title, "sections": []}
        chapters[ch_num]["sections"].append({
            "section": sec_num,
            "section_title": sec_title,
            "filename": filename,
        })
    return [chapters[k] for k in sorted(chapters)]


# ─── Upload ───────────────────────────────────────────────────────────────────

def _upload_and_index(output_dir: Path, cfg: Dict[str, Any]) -> str:
    """Upload all .txt files to OpenAI, create a vector store, return vector_store_id."""
    from openai import OpenAI
    client = OpenAI()

    txt_files = sorted(output_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {output_dir}")

    print(f"  Uploading {len(txt_files)} files to OpenAI...")
    file_ids: List[str] = []
    for i, path in enumerate(txt_files, 1):
        with open(path, "rb") as fh:
            obj = client.files.create(file=fh, purpose="assistants")
            file_ids.append(obj.id)
        if i % 20 == 0 or i == len(txt_files):
            print(f"    {i}/{len(txt_files)} uploaded")

    print(f"  Creating vector store '{cfg['label']}'...")
    vs = client.vector_stores.create(name=cfg["label"])
    vs_id = vs.id
    print(f"  vector_store_id = {vs_id}")

    print(f"  Adding files to vector store (batch)...")
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vs_id,
        file_ids=file_ids,
    )
    while True:
        b = client.vector_stores.file_batches.retrieve(
            vector_store_id=vs_id,
            batch_id=batch.id,
        )
        print(f"    status: {b.status}  counts: {b.file_counts}")
        if b.status in ("completed", "failed", "cancelled"):
            if b.status != "completed":
                raise RuntimeError(f"Vector store batch ended with status={b.status}")
            break
        time.sleep(3)

    return vs_id


# ─── Register ─────────────────────────────────────────────────────────────────

def _register_corpus(cfg: Dict[str, Any], vs_id: str) -> None:
    from tender.corpora.registry import get_corpus

    corpus = get_corpus(cfg["book_id"])
    if corpus.vector_store_id != vs_id:
        raise RuntimeError(
            "Registry discovery returned a stale vector_store_id: "
            f"expected {vs_id}, got {corpus.vector_store_id}"
        )


# ─── INDEX command ────────────────────────────────────────────────────────────

def cmd_index(book_id: str, from_step: Optional[str], until_step: Optional[str]) -> None:
    cfg = _load_book_config(book_id)
    state = _load_state(book_id)
    book_dir = _book_dir(book_id)
    output_dir = _output_dir(book_id)
    steps = INDEX_STEPS
    start_idx = steps.index(from_step) if from_step else 0
    until_idx = steps.index(until_step) if until_step else len(steps) - 1

    if start_idx > until_idx:
        raise ValueError("--from-step must be earlier than or equal to --until-step")

    if from_step:
        _reset_from(state, "index", steps, start_idx)
        _save_state(book_id, state)

    # ── 0. Validate required user inputs ──────────────────────────────────────
    required_inputs = [
        (book_dir / "source" / "book.txt",      "source/book.txt — canonical Markdown source"),
        (book_dir / "glossary_terms.json",       "glossary_terms.json — glossary"),
    ]
    missing = [(p, desc) for p, desc in required_inputs if not p.exists()]
    if missing:
        lines = "\n".join(f"  - {desc}" for _, desc in missing)
        raise FileNotFoundError(f"Missing required inputs:\n{lines}")

    if _invalidate_stale_index_steps(state, book_dir, cfg):
        _save_state(book_id, state)

    source = book_dir / "source" / "book.txt"
    sections = _parse_source(source)
    _assign_output_filenames(sections, cfg)

    # ── 1. split ──────────────────────────────────────────────────────────────
    label = "[1/4] split"
    if start_idx <= steps.index("split") and not _is_done(state, "index", "split"):
        print(f"\n{label} — splitting source Markdown...")
        n = _split_source(sections, cfg, output_dir)
        print(f"  {n} section files → {output_dir.relative_to(ROOT)}")
        _mark_done(
            state,
            "index",
            "split",
            files_written=n,
            fingerprint=_index_step_fingerprint(book_dir, cfg, "split"),
        )
        _save_state(book_id, state)
    else:
        print(f"{label} — already done, skipping.")
    if until_idx == steps.index("split"):
        print(f"\n⏹️ index stage stopped after 'split' for '{book_id}'")
        return

    # ── 2. toc ────────────────────────────────────────────────────────────────
    label = "[2/4] toc"
    if start_idx <= steps.index("toc") and not _is_done(state, "index", "toc"):
        print(f"\n{label} — building table of contents...")
        toc = _build_toc(sections)
        toc_path = book_dir / "toc.json"
        _write_json(toc_path, toc)
        total_sec = sum(len(ch["sections"]) for ch in toc)
        print(f"  {len(toc)} chapters, {total_sec} sections → toc.json")
        _mark_done(
            state,
            "index",
            "toc",
            chapters=len(toc),
            sections=total_sec,
            fingerprint=_index_step_fingerprint(book_dir, cfg, "toc"),
        )
        _save_state(book_id, state)
    else:
        print(f"{label} — already done, skipping.")
    if until_idx == steps.index("toc"):
        print(f"\n⏹️ index stage stopped after 'toc' for '{book_id}'")
        return

    # ── 3. upload ─────────────────────────────────────────────────────────────
    label = "[3/4] upload"
    if start_idx <= steps.index("upload") and not _is_done(state, "index", "upload"):
        print(f"\n{label} — uploading to OpenAI vector store...")
        vs_id = _upload_and_index(output_dir, cfg)
        _mark_done(
            state,
            "index",
            "upload",
            vector_store_id=vs_id,
            fingerprint=_index_step_fingerprint(book_dir, cfg, "upload"),
        )
        _save_state(book_id, state)
        print(f"  Done. vector_store_id = {vs_id}")
    else:
        vs_id = _step_data(state, "index", "upload").get("vector_store_id", "")
        print(f"{label} — already done (vs={vs_id}), skipping.")
    if until_idx == steps.index("upload"):
        print(f"\n⏹️ index stage stopped after 'upload' for '{book_id}'")
        return

    # ── 4. register ───────────────────────────────────────────────────────────
    label = "[4/4] register"
    if start_idx <= steps.index("register") and not _is_done(state, "index", "register"):
        vs_id = _step_data(state, "index", "upload").get("vector_store_id", "")
        if not vs_id:
            raise RuntimeError("No vector_store_id in state — run upload step first.")
        print(f"\n{label} — validating corpus discovery from book metadata...")
        _register_corpus(cfg, vs_id)
        _mark_done(
            state,
            "index",
            "register",
            fingerprint=_index_step_fingerprint(book_dir, cfg, "register"),
        )
        _save_state(book_id, state)
    else:
        print(f"{label} — already done, skipping.")

    print(f"\n✅ index stage complete for '{book_id}'")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tenber book pipeline: split, upload to OpenAI, and register corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tender/tools/build_book.py index --book-id leviathan_book02\n"
            "  python tender/tools/build_book.py index --book-id leviathan_book02 --from-step upload\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Split, upload to OpenAI, and register corpus")
    p_index.add_argument("--book-id", required=True, help="Matches books/<book-id>/")
    p_index.add_argument(
        "--from-step", choices=INDEX_STEPS, default=None,
        help="Re-run from this step (resets it and all subsequent steps)",
    )
    p_index.add_argument(
        "--until-step", choices=INDEX_STEPS, default=None,
        help="Stop after this step instead of continuing to later steps",
    )

    args = parser.parse_args()

    try:
        if args.command == "index":
            cmd_index(args.book_id, args.from_step, args.until_step)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
