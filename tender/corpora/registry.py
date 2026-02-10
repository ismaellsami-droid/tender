from __future__ import annotations
from tender.corpora.hobbes import HOBBES, CorpusConfig

_CORPORA: dict[str, CorpusConfig] = {
    HOBBES.corpus_id: HOBBES,
}

def list_corpora() -> list[CorpusConfig]:
    return list(_CORPORA.values())

def get_corpus(corpus_id: str) -> CorpusConfig:
    if corpus_id not in _CORPORA:
        raise KeyError(f"Unknown corpus_id: {corpus_id}")
    return _CORPORA[corpus_id]
