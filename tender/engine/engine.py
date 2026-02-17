#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tender.corpora.registry import default_corpus_id, get_corpus
from tender.engine.base import BaseCorpusEngine

REFUSAL_TEXT = "Je ne peux pas répondre à partir du corpus actuel."

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

_CH_SEC_FROM_FILENAME_RE = re.compile(r"_ch_(\d{1,2}).*?_s(\d{1,2})_", re.IGNORECASE)


def _ch_sec_from_filename(filename: str) -> tuple[Optional[int], Optional[int]]:
    if not filename:
        return None, None
    m = _CH_SEC_FROM_FILENAME_RE.search(filename)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _normalize_for_match(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = " ".join(s.split())
    return s.lower()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    payload = (text or "").strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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

    def retrieve_passages_v2(self, question: str, *, retrieval_k: int = 20) -> List[Dict[str, Any]]:
        if not self.vector_store_id:
            raise ValueError(f"Missing vector_store_id for corpus '{self.corpus_id}'")
        client = self._get_client()
        page = client.vector_stores.search(
            self.vector_store_id,
            query=question,
            max_num_results=max(1, min(retrieval_k, 50)),
        )
        data = getattr(page, "data", None)
        if data is None:
            data = list(page)

        pool: List[Dict[str, Any]] = []
        for idx, r in enumerate(data or [], 1):
            file_id = getattr(r, "file_id", None) or (r.get("file_id") if isinstance(r, dict) else None)
            filename = getattr(r, "filename", None) or (r.get("filename") if isinstance(r, dict) else None)
            score = getattr(r, "score", None) if not isinstance(r, dict) else r.get("score")
            content = getattr(r, "content", None) if not isinstance(r, dict) else r.get("content")

            text_bits: List[str] = []
            for c in content or []:
                txt = getattr(c, "text", None) if not isinstance(c, dict) else c.get("text")
                if isinstance(txt, str) and txt.strip():
                    text_bits.append(txt.strip())
            snippet = " ".join(text_bits).strip()
            ch, sec = _ch_sec_from_filename(filename or "")
            pool.append(
                {
                    "rank": idx,
                    "file_id": file_id,
                    "filename": filename,
                    "score": score,
                    "snippet": snippet[:500],
                    "passage_text": snippet,
                    "chapter": ch,
                    "section": sec,
                }
            )
        return pool

    def select_passages_v2(
        self,
        pool: List[Dict[str, Any]],
        *,
        selection_k: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        k = max(1, selection_k)
        selected = pool[:k]

        enriched_pool: List[Dict[str, Any]] = []
        for idx, it in enumerate(pool):
            out = dict(it)
            out["selected"] = "Y" if idx < k else "N"
            enriched_pool.append(out)
        return selected, enriched_pool

    def extract_quotes_v2(self, question: str, selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        passages: List[Dict[str, str]] = []
        by_filename: Dict[str, str] = {}

        for it in selected:
            filename = it.get("filename")
            passage = it.get("passage_text")
            if isinstance(filename, str) and filename and isinstance(passage, str) and passage.strip():
                passages.append({"filename": filename, "passage": passage})
                by_filename[filename] = passage

        if not passages:
            return {"refused": True, "raw_output": REFUSAL_TEXT, "quotes": []}

        system = (
            "You are an extract-only assistant.\n"
            "Use ONLY the provided PASSAGES.\n"
            "Do NOT paraphrase. Do NOT invent. Do NOT use outside knowledge.\n"
            "Return JSON only with one of these shapes:\n"
            '{"refusal": true}\n'
            "or\n"
            '{"quotes":[{"quote":"<verbatim from passage>","filename":"<exact filename from passages>"}]}\n'
            "Rules:\n"
            "- 1 to 4 quotes.\n"
            "- quote must be exact verbatim substring from its passage.\n"
            "- filename must be one of provided filenames.\n"
            '- If no relevant quote: {"refusal": true}\n'
        )
        user_payload = {"question": question, "passages": passages}

        client = self._get_client()
        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0,
        )
        out_text = (resp.output_text or "").strip()

        data = _extract_json_object(out_text)
        if not data:
            return {"refused": True, "raw_output": out_text, "quotes": []}
        if data.get("refusal") is True:
            return {"refused": True, "raw_output": out_text, "quotes": []}

        quotes = data.get("quotes")
        if not isinstance(quotes, list):
            return {"refused": True, "raw_output": out_text, "quotes": []}

        valid: List[Dict[str, str]] = []
        for q in quotes:
            if not isinstance(q, dict):
                continue
            qt = q.get("quote")
            fn = q.get("filename")
            if not isinstance(qt, str) or not isinstance(fn, str):
                continue
            if fn not in by_filename:
                continue
            if _normalize_for_match(qt) in _normalize_for_match(by_filename[fn]):
                valid.append({"quote": qt.strip(), "filename": fn})

        valid = valid[:4]
        if not valid:
            return {"refused": True, "raw_output": out_text, "quotes": []}
        return {"refused": False, "raw_output": out_text, "quotes": valid}

    def answer_with_citations_only_v2(
        self,
        question: str,
        *,
        retrieval_k: int = 20,
        selection_k: int = 8,
    ) -> Dict[str, Any]:
        pool = self.retrieve_passages_v2(question, retrieval_k=retrieval_k)
        selected, enriched_pool = self.select_passages_v2(pool, selection_k=selection_k)
        extraction = self.extract_quotes_v2(question, selected)
        return {
            "pool": enriched_pool,
            "selected": selected,
            "quotes": extraction.get("quotes", []),
            "refused": bool(extraction.get("refused", False)),
            "model_output": extraction.get("raw_output", ""),
        }


_ENGINE_CACHE: Dict[str, CorpusEngine] = {}


def get_engine(corpus_id: str) -> CorpusEngine:
    if corpus_id not in _ENGINE_CACHE:
        _ENGINE_CACHE[corpus_id] = CorpusEngine(corpus_id)
    return _ENGINE_CACHE[corpus_id]


def answer_with_citations_only_v2(
    question: str,
    *,
    corpus_id: str = default_corpus_id(),
    retrieval_k: int = 20,
    selection_k: int = 8,
) -> Dict[str, Any]:
    return get_engine(corpus_id).answer_with_citations_only_v2(
        question,
        retrieval_k=retrieval_k,
        selection_k=selection_k,
    )


def reset_conversation_thread(*, corpus_id: str = default_corpus_id()) -> str:
    return get_engine(corpus_id).reset_thread()
