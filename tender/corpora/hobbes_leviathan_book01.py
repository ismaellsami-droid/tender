from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    label: str
    model: str
    vector_store_id: str
    data_dir: str
    reference_label: str = "Source"
    retrieval_k: int = 20
    selection_k: int = 8

# Hobbes / Leviathan Book I
HOBBES_LEVIATHAN_BOOK01 = CorpusConfig(
    corpus_id="hobbes_leviathan_book01",
    label="Hobbes — Leviathan (Book I)",
    model="gpt-4.1-mini",
    vector_store_id="vs_69809284f32c8191936031cb849a1dfd",
    data_dir="books/leviathan_book01/output",
    reference_label="Leviathan I",
)
