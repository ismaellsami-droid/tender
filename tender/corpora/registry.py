from __future__ import annotations

import json
from pathlib import Path

from tender.corpora.corpus_config import CorpusConfig

ROOT = Path(__file__).resolve().parents[2]
BOOKS_DIR = ROOT / "books"
_CORPORA: dict[str, CorpusConfig] = {}


def register(corpus: CorpusConfig) -> None:
    _CORPORA[corpus.corpus_id] = corpus


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_books() -> dict[str, CorpusConfig]:
    discovered: dict[str, CorpusConfig] = {}
    if not BOOKS_DIR.exists():
        return discovered

    for book_dir in sorted(p for p in BOOKS_DIR.iterdir() if p.is_dir()):
        if book_dir.name.startswith("_"):
            continue

        book_path = book_dir / "book.json"
        state_path = book_dir / "build_state.json"
        if not book_path.exists() or not state_path.exists():
            continue

        try:
            cfg = _read_json(book_path)
            state = _read_json(state_path)
        except Exception:
            continue

        book_id = str(cfg.get("book_id", "")).strip()
        upload = (state.get("index") or {}).get("upload") or {}
        vector_store_id = str(upload.get("vector_store_id", "")).strip()
        if not book_id or not vector_store_id:
            continue

        discovered[book_id] = CorpusConfig(
            corpus_id=book_id,
            label=str(cfg.get("label", book_id)),
            model=str(cfg.get("model", "gpt-4.1-mini")),
            vector_store_id=vector_store_id,
            data_dir=f"books/{book_id}/output",
            reference_label=str(cfg.get("reference_label", "Source")),
            retrieval_k=int(cfg.get("retrieval_k", 20)),
            selection_k=int(cfg.get("selection_k", 8)),
            glossary_path=f"books/{book_id}/glossary_terms.json",
            graph_path=f"books/{book_id}/glossary_graph.json",
        )

    return discovered


def _all_corpora() -> dict[str, CorpusConfig]:
    corpora = _discover_books()
    corpora.update(_CORPORA)
    return corpora


def list_corpora() -> list[CorpusConfig]:
    return list(_all_corpora().values())


def get_corpus(corpus_id: str) -> CorpusConfig:
    corpora = _all_corpora()
    if corpus_id not in corpora:
        raise KeyError(f"Unknown corpus_id: {corpus_id}")
    return corpora[corpus_id]


def default_corpus_id() -> str:
    corpora = _all_corpora()
    if not corpora:
        raise RuntimeError("No corpus registered.")
    return next(iter(corpora))
