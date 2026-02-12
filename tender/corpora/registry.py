from __future__ import annotations
from tender.corpora.hobbes_leviathan_book01 import HOBBES_LEVIATHAN_BOOK01, CorpusConfig
from tender.corpora.hobbes_leviathan_book02 import HOBBES_LEVIATHAN_BOOK02

_CORPORA: dict[str, CorpusConfig] = {
    HOBBES_LEVIATHAN_BOOK01.corpus_id: HOBBES_LEVIATHAN_BOOK01,
}

# Register Book II once it has a vector store id configured.
if HOBBES_LEVIATHAN_BOOK02.vector_store_id:
    _CORPORA[HOBBES_LEVIATHAN_BOOK02.corpus_id] = HOBBES_LEVIATHAN_BOOK02

DEFAULT_CORPUS_ID = next(iter(_CORPORA))

def list_corpora() -> list[CorpusConfig]:
    return list(_CORPORA.values())

def get_corpus(corpus_id: str) -> CorpusConfig:
    if corpus_id not in _CORPORA:
        raise KeyError(f"Unknown corpus_id: {corpus_id}")
    return _CORPORA[corpus_id]

def default_corpus_id() -> str:
    return DEFAULT_CORPUS_ID
