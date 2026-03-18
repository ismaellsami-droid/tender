from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


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
    # Path (relative to project root) to the glossary JSON (glossary_terms.json format)
    glossary_path: Optional[str] = None
    # Path (relative to project root) to the glossary graph JSON (user-provided)
    graph_path: Optional[str] = None
