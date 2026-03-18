#!/usr/bin/env python3
# Archived: legacy graph-generation workflow.
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z])")
CHAPTER_RE = re.compile(r"_ch_(\d{1,3})_", re.IGNORECASE)
SECTION_RE = re.compile(r"_s(\d{1,3})_", re.IGNORECASE)
NON_WORD_BOUNDARY = r"(?<![A-Za-z0-9_])"
NON_WORD_BOUNDARY_END = r"(?![A-Za-z0-9_])"


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    source_file: str
    chapter: Optional[int]
    section: Optional[int]
    sentence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-term sentence evidence from a chunked corpus. "
            "A term is matched by its canonical name and aliases."
        )
    )
    parser.add_argument("--glossary", type=Path, required=True, help="Path to glossary_terms.json")
    parser.add_argument("--corpus-dir", type=Path, required=True, help="Directory containing chunked .txt files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for extracted term sentence files",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive term matching (default is case-insensitive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing per-term files instead of resuming",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 means all terms",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_inputs(glossary_path: Path, corpus_dir: Path) -> Dict[str, Any]:
    glossary_raw = glossary_path.read_text(encoding="utf-8")
    files = sorted(p for p in corpus_dir.glob("*.txt") if p.is_file())
    payload = {
        "glossary_sha256": sha256_text(glossary_raw),
        "corpus_files": [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in files
        ],
    }
    payload["fingerprint_sha256"] = sha256_text(json.dumps(payload, sort_keys=True))
    return payload


def slugify_term(term: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", term.strip()).strip("_").lower()
    return slug or "term"


def parse_glossary(glossary_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Glossary JSON must be an array of term records.")
    out: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        term = str(rec.get("term", "")).strip()
        if not term:
            continue
        aliases_raw = rec.get("aliases") or []
        aliases: List[str] = []
        if isinstance(aliases_raw, list):
            for a in aliases_raw:
                s = str(a).strip()
                if s:
                    aliases.append(s)
        out.append(
            {
                "term": term,
                "importance": rec.get("importance"),
                "frequency": rec.get("frequency"),
                "chapters": rec.get("chapters") if isinstance(rec.get("chapters"), list) else [],
                "variants": list(dict.fromkeys([term, *aliases])),
            }
        )
    return out


def split_sentences(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sentences: List[str] = []
    for line in lines:
        parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(line) if p.strip()]
        if parts:
            sentences.extend(parts)
    return sentences


def parse_chapter_section(filename: str) -> Tuple[Optional[int], Optional[int]]:
    chapter: Optional[int] = None
    section: Optional[int] = None
    m_ch = CHAPTER_RE.search(filename)
    m_sec = SECTION_RE.search(filename)
    if m_ch:
        chapter = int(m_ch.group(1))
    if m_sec:
        section = int(m_sec.group(1))
    return chapter, section


def iter_sentence_records(corpus_dir: Path) -> Iterable[SentenceRecord]:
    for path in sorted(corpus_dir.glob("*.txt")):
        chapter, section = parse_chapter_section(path.name)
        text = path.read_text(encoding="utf-8")
        sentences = split_sentences(text)
        for idx, sentence in enumerate(sentences, start=1):
            sentence_id = f"{path.stem}::s{idx:04d}"
            yield SentenceRecord(
                sentence_id=sentence_id,
                source_file=path.name,
                chapter=chapter,
                section=section,
                sentence=sentence,
            )


def compile_variant_regex(variant: str, case_sensitive: bool) -> re.Pattern[str]:
    words = [w for w in re.split(r"\s+", variant.strip()) if w]
    escaped = r"\s+".join(re.escape(w) for w in words)
    pattern = f"{NON_WORD_BOUNDARY}{escaped}{NON_WORD_BOUNDARY_END}"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


def extract_for_term(
    term_record: Dict[str, Any],
    sentence_records: Sequence[SentenceRecord],
    case_sensitive: bool,
) -> Dict[str, Any]:
    variants = [v for v in term_record["variants"] if isinstance(v, str) and v.strip()]
    variant_patterns: List[Tuple[str, re.Pattern[str]]] = [
        (v, compile_variant_regex(v, case_sensitive=case_sensitive)) for v in variants
    ]

    hits: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for record in sentence_records:
        matched_variants: List[str] = []
        for variant, pat in variant_patterns:
            if pat.search(record.sentence):
                matched_variants.append(variant)
        if not matched_variants:
            continue

        key = (record.sentence_id, record.sentence)
        if key in seen:
            continue
        seen.add(key)

        hits.append(
            {
                "sentence_id": record.sentence_id,
                "source_file": record.source_file,
                "chapter": record.chapter,
                "section": record.section,
                "sentence": record.sentence,
                "matched_variants": sorted(set(matched_variants), key=lambda x: x.lower()),
            }
        )

    return {
        "term": term_record["term"],
        "variants": variants,
        "importance": term_record.get("importance"),
        "frequency": term_record.get("frequency"),
        "chapters": term_record.get("chapters") or [],
        "sentence_count": len(hits),
        "sentences": hits,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    glossary_path: Path = args.glossary
    corpus_dir: Path = args.corpus_dir
    output_dir: Path = args.output_dir
    terms_dir = output_dir / "terms"
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "index.json"

    if not glossary_path.exists():
        raise SystemExit(f"Glossary file not found: {glossary_path}")
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")

    current_fingerprint = fingerprint_inputs(glossary_path, corpus_dir)
    previous_manifest = load_manifest(manifest_path)
    if previous_manifest and not args.overwrite:
        prev_fp = previous_manifest.get("input_fingerprint", {}).get("fingerprint_sha256")
        curr_fp = current_fingerprint.get("fingerprint_sha256")
        if prev_fp and curr_fp and prev_fp != curr_fp:
            raise SystemExit(
                "Input fingerprint changed since previous run. "
                "Re-run with --overwrite to rebuild sentence extraction cache."
            )

    glossary_terms = parse_glossary(glossary_path)
    if args.max_terms > 0:
        glossary_terms = glossary_terms[: args.max_terms]

    sentence_records = list(iter_sentence_records(corpus_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    terms_dir.mkdir(parents=True, exist_ok=True)

    term_index: List[Dict[str, Any]] = []
    completed = 0
    skipped = 0

    for i, term_record in enumerate(glossary_terms, start=1):
        term = term_record["term"]
        slug = slugify_term(term)
        term_path = terms_dir / f"{i:04d}_{slug}.json"

        if term_path.exists() and not args.overwrite:
            skipped += 1
            existing = json.loads(term_path.read_text(encoding="utf-8"))
            term_index.append(
                {
                    "term": term,
                    "file": str(term_path.relative_to(output_dir)),
                    "sentence_count": int(existing.get("sentence_count", 0)),
                }
            )
            continue

        extracted = extract_for_term(
            term_record=term_record,
            sentence_records=sentence_records,
            case_sensitive=bool(args.case_sensitive),
        )
        write_json(term_path, extracted)
        completed += 1
        term_index.append(
            {
                "term": term,
                "file": str(term_path.relative_to(output_dir)),
                "sentence_count": extracted["sentence_count"],
            }
        )

    index_payload = {
        "glossary": str(glossary_path),
        "corpus_dir": str(corpus_dir),
        "total_terms": len(glossary_terms),
        "terms": term_index,
    }
    write_json(index_path, index_payload)

    manifest_payload = {
        "tool": "extract_term_sentences.py",
        "input_fingerprint": current_fingerprint,
        "settings": {
            "case_sensitive": bool(args.case_sensitive),
            "max_terms": int(args.max_terms),
        },
        "stats": {
            "total_terms": len(glossary_terms),
            "processed_now": completed,
            "skipped_cached": skipped,
            "total_sentences_indexed": len(sentence_records),
        },
    }
    write_json(manifest_path, manifest_payload)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "processed_now": completed,
                "skipped_cached": skipped,
                "total_terms": len(glossary_terms),
                "total_sentences_indexed": len(sentence_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
