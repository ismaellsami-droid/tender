from __future__ import annotations

import os

from tender.corpora.hobbes_leviathan_book01 import CorpusConfig


HOBBES_LEVIATHAN_BOOK02 = CorpusConfig(
    corpus_id="hobbes_leviathan_book02",
    label="Hobbes — Leviathan (Book II)",
    model="gpt-4.1-mini",
    # Set this env var when Book II has been indexed.
    vector_store_id=os.getenv("TENDER_VS_HOBBES_LEVIATHAN_BOOK02", ""),
    assistant_id=None,
    data_dir="books/leviathan_book02/output",
    reference_label="Leviathan II",
)
