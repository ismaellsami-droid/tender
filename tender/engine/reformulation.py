from __future__ import annotations

from typing import Any

from tender.engine.anchor_types import QuestionReformulationResult
from tender.engine.key_concepts import extract_key_concepts


class ReformulationPlanner:
    """Owns the question-level reformulation pipeline."""

    def __init__(self, lookup: Any) -> None:
        self._lookup = lookup

    def suggest(self, question: str, keywords: list[str]) -> QuestionReformulationResult:
        normalized_keywords = [kw.strip().lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
        key_concepts = extract_key_concepts(question) or normalized_keywords
        initial_candidates = self._lookup.top_reformulation_candidates(
            question,
            top_k=self._lookup.REFORMULATION_TOP_K,
        )
        span_pools = self._lookup._build_span_reformulation_pools(question, key_concepts, initial_candidates)
        flattened = self._lookup._flatten_span_pools(span_pools)
        span_term_matches = self._lookup._build_span_term_matches(span_pools)
        generated_candidates = self._lookup._build_reformulation_sentences(question, span_term_matches)
        return QuestionReformulationResult(
            question=question,
            keywords=key_concepts,
            initial_candidates=initial_candidates,
            candidates=flattened,
            extracted_concepts=key_concepts,
            span_pools=span_pools,
            span_term_matches=span_term_matches,
            generated_candidates=generated_candidates,
        )
