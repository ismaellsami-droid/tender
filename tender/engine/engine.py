#!/usr/bin/env python3
from __future__ import annotations

from tender.corpora.registry import get_corpus
from tender.engine.base import BaseCorpusEngine, AuditRefused
from dataclasses import dataclass
from typing import Dict, Optional, Any
from openai import OpenAI

DEFAULT_STRICT_INSTRUCTIONS = """You are a source-only retrieval assistant.

TASK:
Given a user question, you must respond ONLY with verbatim quotations from the retrieved sources that help answer the question.

RULES:
- Do NOT explain, summarize, paraphrase, or comment.
- Use ONLY excerpts retrieved via file_search from the attached vector store.
- If you cannot find at least ONE relevant quotation, output ONLY:
  "Je ne peux pas répondre à partir du corpus actuel."
"""



class CorpusEngine(BaseCorpusEngine):
    def __init__(self, corpus_id: str):
        corpus = get_corpus(corpus_id)

        super().__init__(
            corpus_id=corpus.corpus_id,
            model=corpus.model,
            vector_store_id=corpus.vector_store_id,
            instructions=getattr(corpus, "instructions", DEFAULT_STRICT_INSTRUCTIONS),
            assistant_id=getattr(corpus, "assistant_id", None),
        )


_ENGINE_CACHE: Dict[str, CorpusEngine] = {}


def get_engine(corpus_id: str) -> CorpusEngine:
    if corpus_id not in _ENGINE_CACHE:
        _ENGINE_CACHE[corpus_id] = CorpusEngine(corpus_id)
    return _ENGINE_CACHE[corpus_id]


def answer_with_citations_only(question: str, *, corpus_id: str = "hobbes", min_citations: int = 1, trace=None) -> str:
    return get_engine(corpus_id).answer_with_citations_only(question, min_citations=min_citations, trace=trace)


def reset_conversation_thread(*, corpus_id: str = "hobbes") -> str:
    return get_engine(corpus_id).reset_thread()
