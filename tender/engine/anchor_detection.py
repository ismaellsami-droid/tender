from __future__ import annotations

import itertools
import logging
import json
import re
from typing import Any, Protocol, Sequence

import numpy as np

from tender.engine.anchor_types import LookupResult, QuestionExplorationResult, QuestionReformulationResult
from tender.engine.exploration import ExplorationPlanner
from tender.engine.reformulation import ReformulationPlanner

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Callable backend that returns one embedding vector per input text."""

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class ExplorationAdvisor(Protocol):
    """Callable advisor for glossary terms worth exploring next."""

    def __call__(self, question: str, keyword: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        ...


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class GlossaryLookup:
    """Keyword-level reformulation plus question-level exploration.

    Reformulation:
        compare each extracted keyword against graph satellites of type `synonym`
        using strict surface match first, then a high-threshold cosine fallback
        → resolve to the corresponding canonical anchor

    Question exploration:
        retrieve canonical glossary top-k from the full question
        for each top-k term:
          - if the term is already present in the question, replace it with its satellites
          - otherwise keep the term itself
        then ask an advisor which candidates are worth exploring next
    """

    SATELLITE_THRESHOLD: float = 0.92
    QUESTION_TOP_K: int = 12
    REFORMULATION_TOP_K: int = 12
    REFORMULATION_INITIAL_MIN_SCORE: float = 0.1
    REFORMULATION_LLM_TOP_K: int = 4
    REFORMULATION_PAIR_THRESHOLD: float = 0.4
    REFORMULATION_FINAL_MIN_SCORE: float = 0.3
    REFORMULATION_SPAN_TOP_K: int = 2
    _IMPORTANCE_RANKS: dict[str, int] = {
        "low": 0,
        "med": 1,
        "medium": 1,
        "high": 2,
    }

    def __init__(
        self,
        glossary_entries: list[dict],
        graph_nodes: list[dict],
        embedding_backend: EmbeddingBackend,
        glossary_by_term: dict,
        exploration_advisor: ExplorationAdvisor | None = None,
    ) -> None:
        self._embedding_backend = embedding_backend
        self._glossary_by_term = glossary_by_term
        self._exploration_advisor = exploration_advisor
        self._result_cache: dict[str, LookupResult] = {}
        self._glossary_entries_by_term: dict[str, dict[str, Any]] = {}

        # ── Canonical term embeddings for question-level retrieval ───────────
        canonical_seen: set[str] = set()
        self._canonical_terms: list[str] = []
        for entry in glossary_entries:
            t = entry.get("term", "")
            if not isinstance(t, str) or not t.strip():
                continue
            nl = t.strip().lower()
            if nl not in canonical_seen:
                canonical_seen.add(nl)
                self._canonical_terms.append(nl)
            self._glossary_entries_by_term[nl] = entry

        self._canonical_embeddings = self._build_embeddings(self._canonical_terms)

        # ── Reformulation satellites: synonym edges only ─────────────────────
        satellite_map: dict[str, str] = {}
        exact_satellite_map: dict[str, str] = {}
        for node in graph_nodes:
            ta = node.get("term", "")
            if not isinstance(ta, str) or not ta.strip():
                continue
            ta_l = ta.strip().lower()
            for edge in node.get("edges", []):
                relation_type = str(edge.get("relation_type", "related")).strip().lower()
                if relation_type != "synonym":
                    continue
                if not self._is_high_strength(edge.get("strength")):
                    continue
                tb = edge.get("term_b", "")
                if not isinstance(tb, str) or not tb.strip():
                    continue
                tb_l = tb.strip().lower()
                if tb_l not in satellite_map:
                    satellite_map[tb_l] = ta_l
                exact_satellite_map[self._normalize_text(tb)] = tb_l

        self._satellite_terms: list[str] = list(satellite_map.keys())
        self._satellite_to_canonical: dict[str, str] = satellite_map
        self._satellite_exact_to_canonical: dict[str, str] = exact_satellite_map
        if self._satellite_terms:
            self._satellite_embeddings = self._build_embeddings(self._satellite_terms)
        else:
            self._satellite_embeddings = np.zeros((0, 1), dtype=np.float32)

        reformulation_seen: set[str] = set()
        reformulation_terms: list[str] = []
        for term in self._canonical_terms:
            if term not in reformulation_seen:
                reformulation_seen.add(term)
                reformulation_terms.append(term)
        for satellite in self._satellite_terms:
            if satellite not in reformulation_seen:
                reformulation_seen.add(satellite)
                reformulation_terms.append(satellite)
        self._reformulation_terms = reformulation_terms
        if self._reformulation_terms:
            self._reformulation_embeddings = self._build_embeddings(self._reformulation_terms)
        else:
            self._reformulation_embeddings = np.zeros((0, 1), dtype=np.float32)

        # ── Adjacency: canonical_lower → neighbors ────────────────────────────
        adjacency: dict[str, list[dict]] = {}
        reverse_adjacency: dict[str, list[dict]] = {}
        for node in graph_nodes:
            ta = node.get("term", "")
            if not isinstance(ta, str) or not ta.strip():
                continue
            ta_l = ta.strip().lower()
            for edge in node.get("edges", []):
                tb = edge.get("term_b", "")
                if not isinstance(tb, str) or not tb.strip():
                    continue
                tb_l = tb.strip().lower()
                link_type = str(edge.get("relation_type", "related"))
                strength = edge.get("strength")
                evidence_quote = edge.get("evidence_quote")
                adjacency.setdefault(ta_l, []).append({
                    "term": tb_l,
                    "link_type": link_type,
                    "is_glossary": tb_l in self._glossary_by_term,
                    "strength": strength,
                    "evidence_quote": evidence_quote,
                })
                reverse_adjacency.setdefault(tb_l, []).append({
                    "term": ta_l,
                    "link_type": link_type,
                    "is_glossary": ta_l in self._glossary_by_term,
                    "strength": strength,
                    "evidence_quote": evidence_quote,
                })

        self._adjacency = adjacency
        self._reverse_adjacency = reverse_adjacency

        # ── Canonical shortlist for question-level retrieval ─────────────────
        self._fallback_embeddings = self._canonical_embeddings
        self._reformulation_planner = ReformulationPlanner(self)
        self._exploration_planner = ExplorationPlanner(self)

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, keyword: str) -> LookupResult:
        """Run keyword-level anchor lookup for one keyword."""
        normalized = keyword.strip().lower()
        if not normalized:
            return LookupResult(
                keyword=keyword, matched_step="no_match",
                canonical_term=None, satellite_term=None, best_score=None,
            )

        cache_key = normalized
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached

        kw_vec = self._embed_single(normalized)

        # Reformulation — strict exact match on synonym satellites first
        exact_satellite = self._satellite_exact_to_canonical.get(self._normalize_text(keyword))
        if exact_satellite is not None:
            exact_canonical = self._satellite_to_canonical[exact_satellite]
            if not self._should_return_anchor(exact_canonical, exact_satellite):
                result = LookupResult(
                    keyword=keyword, matched_step="no_match",
                    canonical_term=None, satellite_term=None, best_score=None,
                )
                self._result_cache[cache_key] = result
                return result
            result = LookupResult(
                keyword=keyword,
                matched_step="satellite",
                canonical_term=exact_canonical,
                satellite_term=exact_satellite,
                best_score=1.0,
                graph_neighbors=self._get_graph_neighbors(exact_canonical),
                synonym_candidates=[],
                exploration_advice={},
            )
            self._result_cache[cache_key] = result
            return result

        # Reformulation — high-threshold embedding match on synonym satellites
        if self._satellite_terms:
            sims = self._satellite_embeddings @ kw_vec
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.SATELLITE_THRESHOLD:
                satellite = self._satellite_terms[best_idx]
                canonical = self._satellite_to_canonical[satellite]
                if not self._should_return_anchor(canonical, satellite):
                    result = LookupResult(
                        keyword=keyword, matched_step="no_match",
                        canonical_term=None, satellite_term=None, best_score=None,
                    )
                    self._result_cache[cache_key] = result
                    return result
                result = LookupResult(
                    keyword=keyword,
                    matched_step="satellite",
                    canonical_term=canonical,
                    satellite_term=satellite,
                    best_score=round(best_score, 6),
                    graph_neighbors=self._get_graph_neighbors(canonical),
                    synonym_candidates=[],
                    exploration_advice={},
                )
                self._result_cache[cache_key] = result
                return result

        result = LookupResult(
            keyword=keyword, matched_step="no_match",
            canonical_term=None, satellite_term=None, best_score=None,
        )
        self._result_cache[cache_key] = result
        return result

    def lookup_keywords(self, keywords: list[str]) -> list[LookupResult]:
        """Batch keyword lookup."""
        return [self.lookup(kw) for kw in keywords]

    def suggest_question_exploration(
        self,
        question: str,
        *,
        keyword: str = "",
        answer_context: dict[str, Any] | None = None,
    ) -> QuestionExplorationResult:
        """Return question-level glossary candidates and optional exploration advice."""
        return self._exploration_planner.suggest(question, keyword=keyword, answer_context=answer_context)

    def suggest_question_reformulation(self, question: str, keywords: list[str]) -> QuestionReformulationResult:
        """Build a ranked reformulation shortlist from question-level glossary + satellite recall."""
        return self._reformulation_planner.suggest(question, keywords)

    def top_glossary_candidates(self, text: str, *, top_k: int = 8, source: str = "keyword") -> list[dict[str, Any]]:
        """Return top-k canonical glossary candidates for an arbitrary text."""
        normalized = text.strip().lower()
        if not normalized or not self._canonical_terms:
            return []
        vec = self._embed_single(normalized)
        sims = self._fallback_embeddings @ vec
        candidate_map: dict[str, dict[str, Any]] = {}
        self._merge_candidate_scores(candidate_map, sims, source=source)
        return sorted(
            candidate_map.values(),
            key=lambda item: float(item["score"]),
            reverse=True,
        )[:top_k]

    def top_reformulation_candidates(self, text: str, *, top_k: int = 12) -> list[dict[str, Any]]:
        """Return top-k candidates over glossary terms plus satellites for reformulation."""
        normalized = text.strip().lower()
        if not normalized or not self._reformulation_terms:
            return []
        vec = self._embed_single(normalized)
        sims = self._reformulation_embeddings @ vec
        rows: list[dict[str, Any]] = []
        for idx, score in enumerate(sims):
            if float(score) < self.REFORMULATION_INITIAL_MIN_SCORE:
                continue
            term = self._reformulation_terms[idx]
            rows.append(self._reformulation_candidate_payload(term, float(score)))
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        return rows[:top_k]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_graph_neighbors(self, canonical: str) -> list[dict]:
        seen: set[str] = set()
        neighbors: list[dict] = []
        for n in self._adjacency.get(canonical, []):
            t = n["term"]
            if t == canonical or t in seen:
                continue
            seen.add(t)
            entry = self._glossary_by_term.get(t, {})
            neighbors.append({
                "term": t,
                "link_type": n["link_type"],
                "is_glossary": n["is_glossary"],
                "importance": entry.get("importance") if entry else None,
                "frequency": entry.get("frequency") if entry else None,
                "strength": n.get("strength"),
                "evidence_quote": n.get("evidence_quote"),
            })
        return neighbors

    def _get_reverse_graph_neighbors(self, satellite: str) -> list[dict]:
        seen: set[str] = set()
        neighbors: list[dict] = []
        for n in self._reverse_adjacency.get(satellite, []):
            t = n["term"]
            if t == satellite or t in seen:
                continue
            if str(n.get("link_type", "")).strip().lower() == "synonym":
                continue
            if not n.get("is_glossary"):
                continue
            seen.add(t)
            entry = self._glossary_by_term.get(t, {})
            neighbors.append({
                "term": t,
                "link_type": n["link_type"],
                "is_glossary": n["is_glossary"],
                "importance": entry.get("importance") if entry else None,
                "frequency": entry.get("frequency") if entry else None,
                "strength": n.get("strength"),
                "evidence_quote": n.get("evidence_quote"),
            })
        return neighbors

    def _get_reverse_synonym_anchors(self, satellite: str) -> list[dict]:
        seen: set[str] = set()
        neighbors: list[dict] = []
        for n in self._reverse_adjacency.get(satellite, []):
            t = n["term"]
            if t == satellite or t in seen:
                continue
            if str(n.get("link_type", "")).strip().lower() != "synonym":
                continue
            if not n.get("is_glossary"):
                continue
            seen.add(t)
            entry = self._glossary_by_term.get(t, {})
            neighbors.append({
                "term": t,
                "link_type": n["link_type"],
                "is_glossary": n["is_glossary"],
                "importance": entry.get("importance") if entry else None,
                "frequency": entry.get("frequency") if entry else None,
                "strength": n.get("strength"),
                "evidence_quote": n.get("evidence_quote"),
            })
        return neighbors

    def _build_question_exploration_candidates(
        self,
        question: str,
        raw_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        transformed: dict[str, dict[str, Any]] = {}
        for item in raw_candidates[:self.QUESTION_TOP_K]:
            term = str(item.get("term", "")).strip().lower()
            if not term:
                continue
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            if self._term_present_in_question(question, term, aliases):
                for neighbor in self._get_graph_neighbors(term):
                    neighbor_term = str(neighbor.get("term", "")).strip().lower()
                    if not neighbor_term:
                        continue
                    if not neighbor.get("is_glossary"):
                        continue
                    if not self._should_keep_question_neighbor(term, neighbor):
                        continue
                    if self._term_present_in_question(question, neighbor_term, []):
                        continue
                    payload = {
                        "term": neighbor_term,
                        "score": item.get("score"),
                        "keyword_score": None,
                        "question_score": item.get("question_score"),
                        "quote": neighbor.get("evidence_quote") or "",
                        "aliases": [],
                        "importance": neighbor.get("importance"),
                        "candidate_sources": ["question_satellite"],
                        "source_anchor": term,
                        "link_type": neighbor.get("link_type"),
                        "strength": neighbor.get("strength"),
                        "is_glossary": neighbor.get("is_glossary"),
                    }
                    self._upsert_question_candidate(transformed, payload)
                for reverse_anchor in self._get_reverse_graph_neighbors(term):
                    reverse_term = str(reverse_anchor.get("term", "")).strip().lower()
                    if not reverse_term:
                        continue
                    if self._term_present_in_question(question, reverse_term, []):
                        continue
                    payload = {
                        "term": reverse_term,
                        "score": item.get("score"),
                        "keyword_score": None,
                        "question_score": item.get("question_score"),
                        "quote": reverse_anchor.get("evidence_quote") or "",
                        "aliases": [],
                        "importance": reverse_anchor.get("importance"),
                        "candidate_sources": ["question_reverse_anchor"],
                        "source_satellite": term,
                        "link_type": reverse_anchor.get("link_type"),
                        "strength": reverse_anchor.get("strength"),
                        "is_glossary": reverse_anchor.get("is_glossary"),
                    }
                    self._upsert_question_candidate(transformed, payload)
                continue

            payload = dict(item)
            payload.setdefault("candidate_sources", ["question"])
            self._upsert_question_candidate(transformed, payload)

        return sorted(
            transformed.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

    def _upsert_question_candidate(self, candidate_map: dict[str, dict[str, Any]], payload: dict[str, Any]) -> None:
        term = str(payload.get("term", "")).strip().lower()
        if not term:
            return
        existing = candidate_map.get(term)
        if existing is None:
            payload["term"] = term
            candidate_map[term] = payload
            return
        existing_score = float(existing.get("score") or 0.0)
        new_score = float(payload.get("score") or 0.0)
        if new_score > existing_score:
            existing["score"] = payload.get("score")
        if payload.get("question_score") is not None:
            existing["question_score"] = payload.get("question_score")
        if payload.get("quote_score") is not None:
            existing["quote_score"] = payload.get("quote_score")
        if payload.get("span_score") is not None:
            existing["span_score"] = payload.get("span_score")
        if payload.get("importance") and not existing.get("importance"):
            existing["importance"] = payload.get("importance")
        if payload.get("quote") and not existing.get("quote"):
            existing["quote"] = payload.get("quote")
        if payload.get("term_in_quote") is not None:
            existing["term_in_quote"] = payload.get("term_in_quote")
        if payload.get("representativeness_score") is not None:
            existing["representativeness_score"] = payload.get("representativeness_score")
        if payload.get("quote_text") and not existing.get("quote_text"):
            existing["quote_text"] = payload.get("quote_text")
        if payload.get("link_type") and not existing.get("link_type"):
            existing["link_type"] = payload.get("link_type")
        if payload.get("strength") and not existing.get("strength"):
            existing["strength"] = payload.get("strength")
        if payload.get("source_anchor") and not existing.get("source_anchor"):
            existing["source_anchor"] = payload.get("source_anchor")
        if payload.get("source_satellite") and not existing.get("source_satellite"):
            existing["source_satellite"] = payload.get("source_satellite")
        if payload.get("is_glossary") is not None and existing.get("is_glossary") is None:
            existing["is_glossary"] = payload.get("is_glossary")
        sources = existing.setdefault("candidate_sources", [])
        for source in payload.get("candidate_sources", []):
            if source not in sources:
                sources.append(source)

    def _term_present_in_text(self, text: str, term: str, aliases: list[str]) -> bool:
        normalized_question = f" {self._normalize_text(text)} "
        candidates = [term, *[str(alias) for alias in aliases]]
        for candidate in candidates:
            normalized_candidate = self._normalize_text(candidate)
            if not normalized_candidate:
                continue
            if f" {normalized_candidate} " in normalized_question:
                return True
        return False

    def _build_span_reformulation_pools(
        self,
        question: str,
        spans: list[str],
        initial_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pools: list[dict[str, Any]] = []
        for span in spans:
            normalized_span = str(span).strip().lower()
            if not normalized_span:
                continue
            expanded_map: dict[str, dict[str, Any]] = {}
            for item in initial_candidates:
                term = str(item.get("term", "")).strip().lower()
                if not term:
                    continue
                aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
                if self._term_present_in_text(normalized_span, term, aliases):
                    if term in self._glossary_by_term:
                        for neighbor in self._get_graph_neighbors(term):
                            if str(neighbor.get("link_type", "")).strip().lower() != "synonym":
                                continue
                            neighbor_term = str(neighbor.get("term", "")).strip().lower()
                            if not neighbor_term:
                                continue
                            payload = {
                                "term": neighbor_term,
                                "score": item.get("score"),
                                "question_score": item.get("question_score"),
                                "best_keyword_score": None,
                                "quote": neighbor.get("evidence_quote") or "",
                                "aliases": [],
                                "importance": neighbor.get("importance"),
                                "candidate_sources": ["reformulation_synonym"],
                                "source_anchor": term,
                                "link_type": neighbor.get("link_type"),
                                "strength": neighbor.get("strength"),
                                "is_glossary": neighbor.get("is_glossary"),
                            }
                            self._upsert_question_candidate(expanded_map, payload)
                    for reverse_anchor in self._get_reverse_synonym_anchors(term):
                        reverse_term = str(reverse_anchor.get("term", "")).strip().lower()
                        if not reverse_term:
                            continue
                        payload = {
                            "term": reverse_term,
                            "score": item.get("score"),
                            "question_score": item.get("question_score"),
                            "best_keyword_score": None,
                            "quote": reverse_anchor.get("evidence_quote") or "",
                            "aliases": [],
                            "importance": reverse_anchor.get("importance"),
                            "candidate_sources": ["reformulation_anchor"],
                            "source_satellite": term,
                            "link_type": reverse_anchor.get("link_type"),
                            "strength": reverse_anchor.get("strength"),
                            "is_glossary": reverse_anchor.get("is_glossary"),
                        }
                        self._upsert_question_candidate(expanded_map, payload)
                    continue

                payload = dict(item)
                payload.setdefault("candidate_sources", ["reformulation_direct"])
                self._upsert_question_candidate(expanded_map, payload)

            rescored = self._score_reformulation_candidates(question, [normalized_span], list(expanded_map.values()))
            filtered: list[dict[str, Any]] = []
            seen_terms: set[str] = set()
            for item in rescored:
                term = str(item.get("term", "")).strip().lower()
                if not term or term in seen_terms:
                    continue
                if float(item.get("question_score") or 0.0) < self.REFORMULATION_INITIAL_MIN_SCORE:
                    continue
                if float(item.get("best_keyword_score") or 0.0) < self.REFORMULATION_PAIR_THRESHOLD:
                    continue
                if float(item.get("final_score") or 0.0) < self.REFORMULATION_FINAL_MIN_SCORE:
                    continue
                aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
                if self._term_present_in_text(normalized_span, term, aliases):
                    continue
                seen_terms.add(term)
                filtered.append(item)
            pools.append(
                {
                    "span": normalized_span,
                    "candidates": filtered[:self.REFORMULATION_SPAN_TOP_K],
                }
            )
        return pools

    def _flatten_span_pools(self, span_pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for pool in span_pools:
            for candidate in pool.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                self._upsert_question_candidate(merged, dict(candidate))
        flattened = list(merged.values())
        flattened.sort(key=lambda item: float(item.get("score") or item.get("final_score") or 0.0), reverse=True)
        return flattened

    def _build_span_term_matches(self, span_pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for pool in span_pools:
            span = str(pool.get("span", "")).strip().lower()
            candidates = pool.get("candidates") if isinstance(pool.get("candidates"), list) else []
            if not span or not candidates:
                continue
            scored_terms: list[dict[str, Any]] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                term = str(candidate.get("term", "")).strip().lower()
                if not term:
                    continue
                scored_terms.append(
                    {
                        "term": term,
                        "cosine": round(float(candidate.get("best_keyword_score") or 0.0), 6),
                        "candidate": candidate,
                    }
                )
            if scored_terms:
                matches.append({"span": span, "matches": scored_terms})
        return matches

    def _build_reformulation_sentences(
        self,
        question: str,
        span_term_matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not question.strip() or not span_term_matches:
            return []

        per_span_options: list[list[dict[str, Any] | None]] = []
        for item in span_term_matches:
            options: list[dict[str, Any] | None] = [None]
            for match in item.get("matches", []):
                options.append(
                    {
                        "span": str(item.get("span", "")).strip(),
                        "term": str(match.get("term", "")).strip(),
                        "cosine": match.get("cosine"),
                    }
                )
            per_span_options.append(options)

        generated: list[dict[str, Any]] = []
        seen_sentences: set[str] = set()
        for combo in itertools.product(*per_span_options):
            replacements = [item for item in combo if item]
            if not replacements:
                continue
            candidate_question = question
            applied: list[dict[str, Any]] = []
            for replacement in sorted(replacements, key=lambda item: len(str(item["span"])), reverse=True):
                new_question, changed = self._replace_first_span(candidate_question, str(replacement["span"]), str(replacement["term"]))
                if not changed:
                    continue
                candidate_question = new_question
                applied.append(replacement)
            normalized_candidate = candidate_question.strip()
            if not applied or not normalized_candidate or normalized_candidate == question.strip():
                continue
            if normalized_candidate in seen_sentences:
                continue
            seen_sentences.add(normalized_candidate)
            generated.append(
                {
                    "text": normalized_candidate,
                    "replacements": [
                        {
                            "span": str(item["span"]),
                            "term": str(item["term"]),
                            "cosine": item.get("cosine"),
                        }
                        for item in applied
                    ],
                }
            )
        return generated

    def _replace_first_span(self, text: str, span: str, term: str) -> tuple[str, bool]:
        if not text or not span or not term:
            return text, False
        pattern = re.compile(re.escape(span), flags=re.IGNORECASE)
        replaced, count = pattern.subn(term, text, count=1)
        return replaced, count > 0

    def _term_present_in_question(self, question: str, term: str, aliases: list[str]) -> bool:
        return self._term_present_in_text(question, term, aliases)

    def _normalize_text(self, text: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return " ".join(normalized.split())

    def _importance_rank(self, importance: Any) -> int:
        if not isinstance(importance, str):
            return -1
        return self._IMPORTANCE_RANKS.get(importance.strip().lower(), -1)

    def _is_high_strength(self, strength: Any) -> bool:
        if isinstance(strength, str):
            return strength.strip().lower() == "high"
        if isinstance(strength, (int, float)):
            return float(strength) >= self.SATELLITE_THRESHOLD
        return False

    def _should_return_anchor(self, canonical: str, satellite: str) -> bool:
        satellite_entry = self._glossary_by_term.get(satellite)
        if not satellite_entry:
            return True
        anchor_entry = self._glossary_by_term.get(canonical)
        return self._importance_rank(anchor_entry.get("importance")) >= self._importance_rank(
            satellite_entry.get("importance")
        )

    def _should_keep_question_neighbor(self, source_anchor: str, neighbor: dict[str, Any]) -> bool:
        if str(neighbor.get("link_type", "")).strip().lower() != "synonym":
            return True
        if not neighbor.get("is_glossary"):
            return False
        source_entry = self._glossary_by_term.get(source_anchor, {})
        return self._importance_rank(neighbor.get("importance")) >= self._importance_rank(
            source_entry.get("importance")
        )

    def _embed_single(self, text: str) -> np.ndarray:
        raw = np.asarray(self._embedding_backend([text]), dtype=np.float32)
        return _normalize_vectors(raw)[0]

    def _build_embeddings(self, texts: list[str]) -> np.ndarray:
        raw = np.asarray(self._embedding_backend(texts), dtype=np.float32)
        return _normalize_vectors(raw)

    def _candidate_payload(self, term: str, score: float) -> dict[str, Any]:
        entry = self._glossary_entries_by_term.get(term, {})
        quote = entry.get("quote")
        aliases = entry.get("aliases")
        return {
            "term": term,
            "score": round(score, 6),
            "keyword_score": None,
            "question_score": None,
            "quote": quote if isinstance(quote, str) else "",
            "aliases": aliases if isinstance(aliases, list) else [],
            "importance": entry.get("importance"),
        }

    def _reformulation_candidate_payload(self, term: str, score: float) -> dict[str, Any]:
        entry = self._glossary_entries_by_term.get(term, {})
        quote = entry.get("quote")
        aliases = entry.get("aliases")
        if term in self._glossary_entries_by_term:
            sources = ["question_glossary"]
        else:
            sources = ["question_satellite"]
        return {
            "term": term,
            "score": round(score, 6),
            "question_score": round(score, 6),
            "best_keyword_score": None,
            "quote": quote if isinstance(quote, str) else "",
            "aliases": aliases if isinstance(aliases, list) else [],
            "importance": entry.get("importance"),
            "candidate_sources": sources,
            "is_glossary": term in self._glossary_entries_by_term,
        }

    def _score_reformulation_candidates(
        self,
        question: str,
        keywords: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        unique_terms = [str(c.get("term", "")).strip().lower() for c in candidates if str(c.get("term", "")).strip()]
        question_vec = self._embed_single(question.strip().lower())
        keyword_vecs = [self._embed_single(keyword) for keyword in keywords]
        term_vecs = self._build_embeddings(unique_terms)
        rescored: list[dict[str, Any]] = []
        for candidate, term_vec in zip(candidates, term_vecs):
            question_score = float(question_vec @ term_vec)
            best_keyword_score = max((float(vec @ term_vec) for vec in keyword_vecs), default=0.0)
            semantic_score = (0.7 * question_score) + (0.3 * best_keyword_score)
            final_score = semantic_score * self._strength_weight(candidate.get("strength"))
            payload = dict(candidate)
            payload["question_score"] = round(question_score, 6)
            payload["best_keyword_score"] = round(best_keyword_score, 6)
            payload["semantic_score"] = round(semantic_score, 6)
            payload["final_score"] = round(final_score, 6)
            payload["score"] = round(final_score, 6)
            rescored.append(payload)
        rescored.sort(key=lambda item: float(item.get("final_score") or item.get("score") or 0.0), reverse=True)
        return rescored

    def _importance_weight(self, importance: Any) -> float:
        rank = self._importance_rank(importance)
        if rank >= 2:
            return 1.08
        if rank == 1:
            return 1.0
        if rank == 0:
            return 0.92
        return 1.0

    def _strength_weight(self, strength: Any) -> float:
        if isinstance(strength, str):
            norm = strength.strip().lower()
            if norm == "high":
                return 1.08
            if norm == "medium":
                return 1.0
            if norm == "low":
                return 0.92
        if isinstance(strength, (int, float)):
            val = float(strength)
            if val >= 0.92:
                return 1.08
            if val >= 0.75:
                return 1.0
            return 0.92
        return 1.0

    def _provenance_weight(self, candidate: dict[str, Any]) -> float:
        sources = set(candidate.get("candidate_sources", []))
        if "reformulation_synonym" in sources or "reformulation_anchor" in sources:
            return 1.08
        if "question_glossary" in sources:
            return 1.0
        if "question_satellite" in sources:
            return 0.98
        return 1.0

    def _merge_candidate_scores(self, candidate_map: dict[str, dict[str, Any]], sims: np.ndarray, *, source: str) -> None:
        for idx, score in enumerate(sims):
            term = self._canonical_terms[idx]
            score_f = float(score)
            existing = candidate_map.get(term)
            if existing is None:
                payload = self._candidate_payload(term, score_f)
                payload["candidate_sources"] = [source]
                payload[f"{source}_score"] = round(score_f, 6)
                candidate_map[term] = payload
                continue
            existing_score = float(existing.get("score", 0.0))
            if score_f > existing_score:
                existing["score"] = round(score_f, 6)
            existing[f"{source}_score"] = round(score_f, 6)
            sources = existing.setdefault("candidate_sources", [])
            if source not in sources:
                sources.append(source)


class OpenAIEmbeddingBackend:
    """Embedding backend backed by the OpenAI embeddings API."""

    def __init__(self, client: object, model: str = "text-embedding-3-small", batch_size: int = 128) -> None:
        self._client = client
        self._model = model
        self._batch_size = max(1, batch_size)

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[Sequence[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i: i + self._batch_size])
            response = self._client.embeddings.create(model=self._model, input=batch)
            for item in sorted(response.data, key=lambda data: data.index):
                vectors.append(item.embedding)
        return vectors


class OpenAIExplorationAdvisor:
    """Suggest glossary terms the user should explore next from step-3 candidates."""

    MODEL: str = "gpt-4o-mini"
    SYSTEM_PROMPT: str = (
        "Given a user question and glossary candidates, choose the glossary terms that would be most useful "
        "for the user to explore next.\n"
        "Prefer terms that help understand or reformulate the user's question.\n"
        "Do not require strict synonymy.\n"
        "Prefer conceptually central glossary terms over weakly related ones.\n"
        "Return up to 3 terms.\n"
        "If none are useful, return an empty list.\n"
        "Return JSON only with keys: suggested_terms, reason."
    )

    def __init__(self, client: Any) -> None:
        self._client = client

    def __call__(self, question: str, keyword: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "question": question,
            "keyword": keyword,
            "step3_topk": candidates,
            "question_topk": candidates,
        }
        resp = self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=250,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return {"suggested_terms": [], "reason": "empty_response"}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Exploration advisor returned non-JSON content: %s", content)
            return {"suggested_terms": [], "reason": "invalid_json"}
        if not isinstance(parsed, dict):
            return {"suggested_terms": [], "reason": "invalid_payload"}
        terms = parsed.get("suggested_terms")
        if not isinstance(terms, list):
            parsed["suggested_terms"] = []
        else:
            parsed["suggested_terms"] = [str(term).strip().lower() for term in terms if str(term).strip()][:3]
        return parsed
