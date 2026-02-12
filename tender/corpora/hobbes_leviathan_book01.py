from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    label: str
    model: str
    vector_store_id: str
    assistant_id: str | None
    data_dir: str
    reference_label: str = "Source"
    instructions: str | None = None

# Hobbes / Leviathan Book I
HOBBES_LEVIATHAN_BOOK01 = CorpusConfig(
    corpus_id="hobbes_leviathan_book01",
    label="Hobbes — Leviathan (Book I)",
    model="gpt-4.1-mini",
    vector_store_id="vs_69809284f32c8191936031cb849a1dfd",
    assistant_id="asst_mJ0aX3HVGDQCvey3h29r9kgF",  # peut être None si tu veux auto-create
    data_dir="books/leviathan_book01/output",
    reference_label="Leviathan I",
)
