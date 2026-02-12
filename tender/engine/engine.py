#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict

from tender.corpora.registry import default_corpus_id, get_corpus
from tender.engine.base import BaseCorpusEngine

DEFAULT_STRICT_INSTRUCTIONS = """You are a source-only retrieval assistant.

TASK:
Given a user question, you must respond ONLY with verbatim quotations from the retrieved sources that help answer the question.

OUTPUT FORMAT (STRICT):
•⁠  ⁠Return between 2 and 4 quotations (minimum 1).
•⁠  ⁠Each quotation must be:
  1) Verbatim (exact text from the source, in English).
  2) Short (1–2 sentences, max ~40 words).
  3) Immediately followed by its reference on the same line, formatted as:
     Source, Ch. <number>, §<number>
     (If the section number cannot be inferred reliably, write:
      Source, Ch. <number>)

RULES:
•⁠  ⁠Do NOT explain, summarize, paraphrase, or comment in any way.
•⁠  ⁠Do NOT add any introductory or concluding text.
•⁠  ⁠Do NOT use outside knowledge.
•⁠  ⁠Use ONLY excerpts retrieved via file_search from the attached vector store.
•⁠  ⁠If you cannot find at least ONE relevant quotation, output ONLY:
  "Je ne peux pas répondre à partir du corpus actuel."
•⁠  ⁠Infer Chapter and Section numbers from the source filename when possible
  (e.g. filename contains "ch_13" and "s01" → Source, Ch. 13, §1).
"""



class CorpusEngine(BaseCorpusEngine):
    def __init__(self, corpus_id: str):
        corpus = get_corpus(corpus_id)
        instructions = corpus.instructions or DEFAULT_STRICT_INSTRUCTIONS

        super().__init__(
            corpus_id=corpus.corpus_id,
            model=corpus.model,
            vector_store_id=corpus.vector_store_id,
            instructions=instructions,
            assistant_id=getattr(corpus, "assistant_id", None),
        )


_ENGINE_CACHE: Dict[str, CorpusEngine] = {}


def get_engine(corpus_id: str) -> CorpusEngine:
    if corpus_id not in _ENGINE_CACHE:
        _ENGINE_CACHE[corpus_id] = CorpusEngine(corpus_id)
    return _ENGINE_CACHE[corpus_id]


def answer_with_citations_only(question: str, *, corpus_id: str = default_corpus_id(), min_citations: int = 1, trace=None) -> str:
    return get_engine(corpus_id).answer_with_citations_only(question, min_citations=min_citations, trace=trace)


def reset_conversation_thread(*, corpus_id: str = default_corpus_id()) -> str:
    return get_engine(corpus_id).reset_thread()
