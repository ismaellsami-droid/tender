from __future__ import annotations
import re
from typing import Any

from tender.engine.anchor_types import QuestionExplorationResult
from tender.engine.key_concepts import extract_key_concepts


class ExplorationPlanner:
    """Owns the question-level exploration pipeline."""

    def __init__(self, lookup: Any) -> None:
        self._lookup = lookup

    def suggest(self, question: str, *, keyword: str = "", answer_context: dict[str, Any] | None = None) -> QuestionExplorationResult:
        answer_quotes = self._extract_answer_quotes(answer_context)
        if answer_quotes:
            return self._suggest_from_answer(question, answer_quotes, keyword=keyword)

        raw_candidates = self._lookup.top_glossary_candidates(
            question,
            top_k=self._lookup.QUESTION_TOP_K,
            source="question",
        )
        extracted_concepts = extract_key_concepts(question)
        span_pools = self._build_span_pools(question, extracted_concepts, raw_candidates)
        shortlist_payload = self._flatten_span_pools(span_pools) if span_pools else self._lookup._build_question_exploration_candidates(question, raw_candidates)
        exploration_advice: dict[str, Any] = {}
        if shortlist_payload and self._lookup._exploration_advisor is not None:
            try:
                exploration_advice = self._lookup._exploration_advisor(question, keyword, shortlist_payload)
            except Exception:
                exploration_advice = {"suggested_terms": [], "reason": "advisor_call_failed"}
        return QuestionExplorationResult(
            question=question,
            source_mode="question",
            initial_candidates=raw_candidates,
            extracted_concepts=extracted_concepts,
            span_pools=span_pools,
            quote_pools=[],
            candidates=shortlist_payload,
            exploration_advice=exploration_advice,
        )

    def _suggest_from_answer(
        self,
        question: str,
        answer_quotes: list[dict[str, Any]],
        *,
        keyword: str = "",
    ) -> QuestionExplorationResult:
        initial_candidates: list[dict[str, Any]] = []
        quote_pools: list[dict[str, Any]] = []
        merged_candidates: dict[str, dict[str, Any]] = {}

        for index, quote in enumerate(answer_quotes, 1):
            quote_text = str(quote.get("text") or "").strip()
            if not quote_text:
                continue
            quote_label = str(quote.get("ref") or f"Quote {index}").strip()
            phrase_texts = self._split_quote_into_phrases(quote_text)
            for phrase_index, phrase_text in enumerate(phrase_texts, 1):
                raw_candidates = self._lookup.top_glossary_candidates(
                    phrase_text,
                    top_k=self._lookup.QUESTION_TOP_K,
                    source="quote",
                )
                initial_candidates.extend(raw_candidates)
                transformed = self._build_quote_candidates(question, phrase_text, raw_candidates)
                rescored = self._score_quote_pool(question, phrase_text, transformed)
                phrase_label = quote_label if len(phrase_texts) == 1 else f"{quote_label} · sentence {phrase_index}"
                quote_pools.append(
                    {
                        "span": phrase_label,
                        "quote_text": phrase_text,
                        "candidates": rescored,
                    }
                )
                for candidate in rescored:
                    self._lookup._upsert_question_candidate(merged_candidates, dict(candidate))

        shortlist_payload = sorted(
            merged_candidates.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        exploration_advice: dict[str, Any] = {}
        if shortlist_payload and self._lookup._exploration_advisor is not None:
            try:
                exploration_advice = self._lookup._exploration_advisor(question, keyword, shortlist_payload)
            except Exception:
                exploration_advice = {"suggested_terms": [], "reason": "advisor_call_failed"}

        answer_text = " ".join(q.get("text", "") for q in answer_quotes if isinstance(q, dict))
        extracted_concepts = extract_key_concepts(answer_text) if answer_text else []
        return QuestionExplorationResult(
            question=question,
            source_mode="answer",
            initial_candidates=initial_candidates,
            extracted_concepts=extracted_concepts,
            span_pools=[],
            quote_pools=quote_pools,
            candidates=shortlist_payload,
            exploration_advice=exploration_advice,
        )

    def _extract_answer_quotes(self, answer_context: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(answer_context, dict):
            return []
        raw_quotes = answer_context.get("answer_quotes")
        if not isinstance(raw_quotes, list):
            return []
        return [item for item in raw_quotes if isinstance(item, dict) and str(item.get("text") or "").strip()]

    def _split_quote_into_phrases(self, quote_text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", quote_text.strip())
        if not normalized:
            return []
        phrases = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", normalized)
            if part.strip()
        ]
        return phrases or [normalized]

    def _build_quote_candidates(
        self,
        question: str,
        quote_text: str,
        raw_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        transformed: dict[str, dict[str, Any]] = {}
        for item in raw_candidates[: self._lookup.QUESTION_TOP_K]:
            term = str(item.get("term", "")).strip().lower()
            if not term:
                continue
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            term_in_quote = self._lookup._term_present_in_text(quote_text, term, aliases)
            representativeness = self._representativeness_score(term, float(item.get("score") or 0.0))

            if term_in_quote and representativeness is not None and representativeness >= 0.85:
                if term in self._lookup._glossary_by_term:
                    for neighbor in self._lookup._get_graph_neighbors(term):
                        neighbor_term = str(neighbor.get("term", "")).strip().lower()
                        if not neighbor_term:
                            continue
                        if not neighbor.get("is_glossary"):
                            continue
                        if not self._lookup._should_keep_question_neighbor(term, neighbor):
                            continue
                        if self._lookup._term_present_in_text(quote_text, neighbor_term, []):
                            continue
                        payload = {
                            "term": neighbor_term,
                            "score": item.get("score"),
                            "keyword_score": None,
                            "question_score": None,
                            "quote_score": float(item.get("score") or 0.0),
                            "quote": neighbor.get("evidence_quote") or "",
                            "aliases": [],
                            "importance": neighbor.get("importance"),
                            "candidate_sources": ["quote_satellite"],
                            "source_anchor": term,
                            "link_type": neighbor.get("link_type"),
                            "strength": neighbor.get("strength"),
                            "is_glossary": neighbor.get("is_glossary"),
                            "term_in_quote": term_in_quote,
                            "representativeness_score": round(representativeness, 6),
                        }
                        self._lookup._upsert_question_candidate(transformed, payload)
                for reverse_anchor in self._lookup._get_reverse_graph_neighbors(term):
                    reverse_term = str(reverse_anchor.get("term", "")).strip().lower()
                    if not reverse_term:
                        continue
                    if self._lookup._term_present_in_text(quote_text, reverse_term, []):
                        continue
                    payload = {
                        "term": reverse_term,
                        "score": item.get("score"),
                        "keyword_score": None,
                        "question_score": None,
                        "quote_score": float(item.get("score") or 0.0),
                        "quote": reverse_anchor.get("evidence_quote") or "",
                        "aliases": [],
                        "importance": reverse_anchor.get("importance"),
                        "candidate_sources": ["quote_reverse_anchor"],
                        "source_satellite": term,
                        "link_type": reverse_anchor.get("link_type"),
                        "strength": reverse_anchor.get("strength"),
                        "is_glossary": reverse_anchor.get("is_glossary"),
                        "term_in_quote": term_in_quote,
                        "representativeness_score": round(representativeness, 6),
                    }
                    self._lookup._upsert_question_candidate(transformed, payload)
                continue

            payload = dict(item)
            payload["quote_score"] = float(item.get("score") or 0.0)
            payload["term_in_quote"] = term_in_quote
            payload["representativeness_score"] = (
                round(representativeness, 6) if representativeness is not None else None
            )
            payload.setdefault("candidate_sources", ["quote"])
            self._lookup._upsert_question_candidate(transformed, payload)
        return list(transformed.values())

    def _representativeness_score(self, term: str, quote_score: float) -> float | None:
        entry = self._lookup._glossary_entries_by_term.get(term, {})
        glossary_quote = entry.get("quote")
        if not isinstance(glossary_quote, str) or not glossary_quote.strip():
            return None
        term_vec = self._lookup._embed_single(term)
        quote_vec = self._lookup._embed_single(glossary_quote.strip().lower())
        anchor_score = float(term_vec @ quote_vec)
        if anchor_score <= 0:
            return None
        return quote_score / anchor_score

    def _build_span_pools(
        self,
        question: str,
        spans: list[str],
        raw_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not spans:
            return []

        span_pools: list[dict[str, Any]] = []
        for span in spans:
            normalized_span = str(span).strip().lower()
            if not normalized_span:
                continue
            transformed: dict[str, dict[str, Any]] = {}
            for item in raw_candidates[: self._lookup.QUESTION_TOP_K]:
                term = str(item.get("term", "")).strip().lower()
                if not term:
                    continue
                aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
                if self._lookup._term_present_in_text(normalized_span, term, aliases):
                    for neighbor in self._lookup._get_graph_neighbors(term):
                        neighbor_term = str(neighbor.get("term", "")).strip().lower()
                        if not neighbor_term:
                            continue
                        if not neighbor.get("is_glossary"):
                            continue
                        if not self._lookup._should_keep_question_neighbor(term, neighbor):
                            continue
                        if self._lookup._term_present_in_text(normalized_span, neighbor_term, []):
                            continue
                        payload = {
                            "term": neighbor_term,
                            "score": item.get("score"),
                            "keyword_score": None,
                            "question_score": item.get("question_score"),
                            "span_score": None,
                            "quote": neighbor.get("evidence_quote") or "",
                            "aliases": [],
                            "importance": neighbor.get("importance"),
                            "candidate_sources": ["question_satellite"],
                            "source_anchor": term,
                            "link_type": neighbor.get("link_type"),
                            "strength": neighbor.get("strength"),
                            "is_glossary": neighbor.get("is_glossary"),
                        }
                        self._lookup._upsert_question_candidate(transformed, payload)
                    for reverse_anchor in self._lookup._get_reverse_graph_neighbors(term):
                        reverse_term = str(reverse_anchor.get("term", "")).strip().lower()
                        if not reverse_term:
                            continue
                        if self._lookup._term_present_in_text(normalized_span, reverse_term, []):
                            continue
                        payload = {
                            "term": reverse_term,
                            "score": item.get("score"),
                            "keyword_score": None,
                            "question_score": item.get("question_score"),
                            "span_score": None,
                            "quote": reverse_anchor.get("evidence_quote") or "",
                            "aliases": [],
                            "importance": reverse_anchor.get("importance"),
                            "candidate_sources": ["question_reverse_anchor"],
                            "source_satellite": term,
                            "link_type": reverse_anchor.get("link_type"),
                            "strength": reverse_anchor.get("strength"),
                            "is_glossary": reverse_anchor.get("is_glossary"),
                        }
                        self._lookup._upsert_question_candidate(transformed, payload)
                    continue

                payload = dict(item)
                payload.setdefault("candidate_sources", ["question"])
                self._lookup._upsert_question_candidate(transformed, payload)

            rescored = self._score_span_pool(question, normalized_span, list(transformed.values()))
            span_pools.append(
                {
                    "span": normalized_span,
                    "candidates": rescored,
                }
            )
        return span_pools

    def _score_span_pool(
        self,
        question: str,
        span: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        question_vec = self._lookup._embed_single(question.strip().lower())
        span_vec = self._lookup._embed_single(span)
        term_vecs = self._lookup._build_embeddings(
            [str(c.get("term", "")).strip().lower() for c in candidates if str(c.get("term", "")).strip()]
        )
        rescored: list[dict[str, Any]] = []
        for candidate, term_vec in zip(candidates, term_vecs):
            question_score = float(question_vec @ term_vec)
            span_score = float(span_vec @ term_vec)
            final_score = (0.7 * question_score) + (0.3 * span_score)
            payload = dict(candidate)
            payload["question_score"] = round(question_score, 6)
            payload["span_score"] = round(span_score, 6)
            payload["score"] = round(final_score, 6)
            rescored.append(payload)
        rescored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return rescored

    def _score_quote_pool(
        self,
        question: str,
        quote_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        question_vec = self._lookup._embed_single(question.strip().lower())
        quote_vec = self._lookup._embed_single(quote_text.strip().lower())
        term_vecs = self._lookup._build_embeddings(
            [str(c.get("term", "")).strip().lower() for c in candidates if str(c.get("term", "")).strip()]
        )
        rescored: list[dict[str, Any]] = []
        for candidate, term_vec in zip(candidates, term_vecs):
            question_score = float(question_vec @ term_vec)
            quote_score = float(quote_vec @ term_vec)
            final_score = (0.8 * quote_score) + (0.2 * question_score)
            payload = dict(candidate)
            payload["question_score"] = round(question_score, 6)
            payload["quote_score"] = round(quote_score, 6)
            payload["score"] = round(final_score, 6)
            rescored.append(payload)
        rescored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return rescored

    def _flatten_span_pools(self, span_pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for pool in span_pools:
            candidates = pool.get("candidates") if isinstance(pool.get("candidates"), list) else []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                self._lookup._upsert_question_candidate(merged, dict(candidate))
        flattened = list(merged.values())
        flattened.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return flattened
