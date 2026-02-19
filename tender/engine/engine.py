#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from tender.corpora.registry import default_corpus_id, get_corpus
from tender.engine.base import BaseCorpusEngine

REFUSAL_TEXT = "Je ne peux pas répondre à partir du corpus actuel."

_CH_SEC_FROM_FILENAME_RE = re.compile(r"_ch_(\d{1,2}).*?_s(\d{1,2})_", re.IGNORECASE)
_CH_ONLY_FROM_FILENAME_RE = re.compile(r"_ch_(\d{1,2})_", re.IGNORECASE)


def _ch_sec_from_filename(filename: str) -> tuple[Optional[int], Optional[int]]:
    if not filename:
        return None, None
    m = _CH_SEC_FROM_FILENAME_RE.search(filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = _CH_ONLY_FROM_FILENAME_RE.search(filename)
    if m2:
        return int(m2.group(1)), None
    return None, None


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

        super().__init__(
            corpus_id=corpus.corpus_id,
            model=corpus.model,
            vector_store_id=corpus.vector_store_id,
        )

    def retrieve_passages(self, question: str, *, retrieval_k: int = 20) -> List[Dict[str, Any]]:
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

    def select_passages(
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

    def extract_quotes(self, question: str, selected: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    def answer_with_citations(
        self,
        question: str,
        *,
        retrieval_k: int = 20,
        selection_k: int = 8,
    ) -> Dict[str, Any]:
        pool = self.retrieve_passages(question, retrieval_k=retrieval_k)
        selected, enriched_pool = self.select_passages(pool, selection_k=selection_k)
        extraction = self.extract_quotes(question, selected)
        return {
            "pool": enriched_pool,
            "selected": selected,
            "quotes": extraction.get("quotes", []),
            "refused": bool(extraction.get("refused", False)),
            "model_output": extraction.get("raw_output", ""),
        }


_ENGINE_CACHE: Dict[str, CorpusEngine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def get_engine(corpus_id: str) -> CorpusEngine:
    if corpus_id in _ENGINE_CACHE:
        return _ENGINE_CACHE[corpus_id]
    with _ENGINE_CACHE_LOCK:
        if corpus_id not in _ENGINE_CACHE:
            _ENGINE_CACHE[corpus_id] = CorpusEngine(corpus_id)
        return _ENGINE_CACHE[corpus_id]


def answer_with_citations(
    question: str,
    *,
    corpus_id: str = default_corpus_id(),
    retrieval_k: int = 20,
    selection_k: int = 8,
) -> Dict[str, Any]:
    return get_engine(corpus_id).answer_with_citations(
        question,
        retrieval_k=retrieval_k,
        selection_k=selection_k,
    )


