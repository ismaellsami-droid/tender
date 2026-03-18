#!/usr/bin/env python3
"""tender/engine/scoring.py

BookScore and GlossaryGraphMatcher.

BookScore evaluates the retrieval pool quality (pre-answer) from the top
retrieval score only to decide:
  high / med / low  →  answer / answer-with-hint / refuse

GlossaryGraphMatcher matches question keywords against the corpus glossary
and the synonym/antonym graph to produce structured exploration/reformulation hints.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BookScore
# ─────────────────────────────────────────────────────────────────────────────

BOOKSCORE_DEFAULTS: Dict[str, Any] = dict(
    score_low=0.20,
    score_high=0.80,
    w_b1=1.0,
)

BOOKSCORE_ZONE_HIGH: float = 0.75
BOOKSCORE_ZONE_LOW: float = 0.55


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def compute_bookscore(
    candidates: List[Dict[str, Any]],
    *,
    score_low: float = 0.20,
    score_high: float = 0.80,
    w_b1: float = 1.0,
) -> Dict[str, Any]:
    """Compute BookScore from the top retrieval score (post-rerank / pre-answer).

    Candidates must be sorted descending by retrieval / rerank score.

    Returns a dict with:
        bookscore, zone, B1, top1, params_used
    """
    params_used = dict(
        score_low=score_low,
        score_high=score_high,
        w_b1=w_b1,
    )
    _empty: Dict[str, Any] = dict(
        bookscore=0.0, zone="low", B1=0.0, top1=0.0,
        params_used=params_used,
    )

    if not candidates:
        return _empty

    scores: List[float] = []
    for c in candidates:
        s = c.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    if not scores:
        return _empty

    # ── B1: Relevance ───────────────────────────────────────────────────────
    top1 = scores[0]
    denom = score_high - score_low
    B1 = _clamp((top1 - score_low) / denom) if denom > 0 else 0.0

    # ── BookScore + zone ─────────────────────────────────────────────────────
    bookscore = w_b1 * B1
    zone = "high" if bookscore >= BOOKSCORE_ZONE_HIGH else ("low" if bookscore <= BOOKSCORE_ZONE_LOW else "med")

    return dict(
        bookscore=round(bookscore, 6),
        zone=zone,
        B1=round(B1, 6),
        top1=round(top1, 6),
        params_used=params_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Keyword extraction (spaCy)
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_VERBS = frozenset({
    "think", "mean", "know", "does", "according", "make", "say",
    "tell", "call", "use", "be", "do", "have", "get", "go", "come",
    "consider", "describe", "explain", "define", "characterize",
})

_NLP = None  # module-level spaCy model cache


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_keywords(question: str) -> Dict[str, Any]:
    """Extract content keywords and n-gram phrases from a question using spaCy.

    Returns:
        kept     : List[str]  – noun chunks (2-3 tokens, raw) followed by
                                lemmatized single keywords. Chunks first so
                                multi-word glossary terms are matched directly.
        dropped  : List[Dict] – filtered tokens with word/pos/reason
        fallback : bool       – True when kept is empty (use full question)
    """
    try:
        _nlp = _get_nlp()
    except Exception as exc:
        logger.warning("spaCy unavailable (%s) — keyword fallback", exc)
        return {"kept": [], "dropped": [], "fallback": True}

    doc = _nlp(question)

    # ── 1. N-grams (2-3 tokens) — raw text, not lemmatized ──────────────────
    # Two sources:
    #   a) spaCy noun_chunks  → linguistically coherent NPs (e.g. "human life")
    #   b) sliding window     → captures "State of Nature", "Knowledge of Fact", etc.
    #      Rule: first AND last token must be content words (not stopwords, not punct)
    chunks: List[str] = []
    seen_chunks: set = set()

    # a) noun chunks
    for chunk in doc.noun_chunks:
        real_tokens = [t for t in chunk if not t.is_space]
        if 2 <= len(real_tokens) <= 3:
            first, last = real_tokens[0], real_tokens[-1]
            if not first.is_stop and not last.is_stop:
                text = chunk.text.lower().strip()
                if text not in seen_chunks:
                    seen_chunks.add(text)
                    chunks.append(text)

    # b) sliding window 2-grams and 3-grams over non-space, non-punct tokens
    content_tokens = [t for t in doc if not t.is_space and not t.is_punct]
    for n in (2, 3):
        for i in range(len(content_tokens) - n + 1):
            span = content_tokens[i: i + n]
            first, last = span[0], span[-1]
            if first.is_stop or last.is_stop:
                continue
            text = " ".join(t.text.lower() for t in span)
            if text not in seen_chunks:
                seen_chunks.add(text)
                chunks.append(text)

    # ── 2. Individual content tokens — lemmatized (existing logic) ───────────
    single: List[str] = []
    dropped: List[Dict[str, str]] = []

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN", "ADJ", "VERB"):
            dropped.append({"word": token.text, "pos": token.pos_, "reason": "pos_filter"})
            continue
        lemma = token.lemma_.lower()
        if lemma in _GENERIC_VERBS:
            dropped.append({"word": token.text, "pos": token.pos_, "reason": "generic"})
            continue
        if token.is_stop:
            dropped.append({"word": token.text, "pos": token.pos_, "reason": "stopword"})
            continue
        single.append(lemma)

    # Chunks first (more specific), then individual keywords
    kept = chunks + single
    return {"kept": kept, "dropped": dropped, "fallback": len(kept) == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_BATCH = 100

# Embedding similarity threshold for glossary matching.
SIM_MIN: float = 0.60


def _batch_embed(texts: List[str], client: Any, model: str) -> "Any":  # returns np.ndarray
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required. Run: pip install numpy") from exc

    all_embs: List[List[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i: i + _EMBED_BATCH]
        resp = client.embeddings.create(input=batch, model=model)
        for item in sorted(resp.data, key=lambda x: x.index):
            all_embs.append(item.embedding)
    return np.array(all_embs, dtype=np.float32)


def _normalize_rows(mat: "Any") -> "Any":
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# ─────────────────────────────────────────────────────────────────────────────
# GlossaryGraphMatcher
# ─────────────────────────────────────────────────────────────────────────────

class GlossaryGraphMatcher:
    """Match question keywords against the corpus glossary and syn/ant graph.

    Two-step per keyword:
      Step 1 — rapidfuzz (FUZZY_MIN=80): keyword vs all glossary texts (terms+aliases).
               If match → return glossary entry + graph expansion.
      Step 2 — embedding (SIM_MIN=0.50): if no fuzzy match, embed keyword and find
               close glossary texts. Return entries + graph expansion.

    Graph-only node embeddings are no longer used.
    """

    def __init__(
        self,
        glossary: List[Dict[str, Any]],
        graph: List[Dict[str, Any]],
        glossary_embs: "Any",      # np.ndarray (n_gloss_texts, dim), normalised
        glossary_texts: List[str], # texts corresponding to glossary_embs rows
        client: Any,
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self.glossary = glossary
        self.graph = graph
        self._client = client
        self._embed_model = embed_model

        self._glossary_embs = glossary_embs
        self._glossary_texts = glossary_texts

        # text (lowercase) → glossary entry
        self._text_to_glossary: Dict[str, Dict[str, Any]] = {}
        for entry in glossary:
            self._text_to_glossary[entry["term"].lower()] = entry
            for alias in entry.get("aliases", []):
                if isinstance(alias, str):
                    self._text_to_glossary[alias.lower()] = entry

        # graph node (exact) → list of edges containing it
        self._node_to_edges: Dict[str, List[Dict[str, Any]]] = {}
        for edge in graph:
            for t in edge.get("terms", []):
                self._node_to_edges.setdefault(t, []).append(edge)

        self._q_cache: Dict[str, "Any"] = {}

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _embed_kw(self, kw: str) -> "Any":
        if kw not in self._q_cache:
            import numpy as np
            raw = _batch_embed([kw], self._client, self._embed_model)
            norm = float(np.linalg.norm(raw[0]))
            self._q_cache[kw] = raw[0] / norm if norm > 0 else raw[0]
        return self._q_cache[kw]

    # ── Lookup helpers ───────────────────────────────────────────────────────

    def _edges_for_node(self, node: str) -> List[Dict[str, Any]]:
        return list(self._node_to_edges.get(node, []))

    def _edges_for_glossary_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """All graph edges where this glossary term or any of its aliases appear."""
        texts_to_check = [entry["term"]] + [
            a for a in entry.get("aliases", []) if isinstance(a, str)
        ]
        seen: set = set()
        edges: List[Dict[str, Any]] = []
        for t in texts_to_check:
            for e in self._node_to_edges.get(t, []):
                eid = id(e)
                if eid not in seen:
                    seen.add(eid)
                    edges.append(e)
        return edges

    def _glossary_from_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect unique glossary entries referenced by these graph edges."""
        seen_terms: set = set()
        result: List[Dict[str, Any]] = []
        for edge in edges:
            for t in edge.get("terms", []):
                entry = self._text_to_glossary.get(t.lower())
                if entry and entry["term"] not in seen_terms:
                    seen_terms.add(entry["term"])
                    result.append(entry)
        return result

    # ── Core expansion logic ──────────────────────────────────────────────────

    def _expand_keyword(self, kw: str) -> List[Dict[str, Any]]:
        """Build a flat list of terms related to kw via embedding similarity.

        Embedding (SIM_MIN=0.60): kw vs all glossary texts (terms+aliases).
        Matches → return glossary entries + graph expansion.

        Each item: {term, sim, origin, from_term, is_glossary, importance, frequency}.
        Duplicates suppressed (first occurrence wins).
        """
        import numpy as np

        result: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(term: str, sim: Optional[float], origin: str,
                 from_term: Optional[str], entry: Optional[Dict[str, Any]]) -> bool:
            if term in seen:
                return False
            seen.add(term)
            result.append({
                "term": term,
                "sim": round(sim, 4) if sim is not None else None,
                "origin": origin,
                "from_term": from_term,
                "is_glossary": entry is not None,
                "importance": entry.get("importance") if entry else None,
                "frequency": entry.get("frequency") if entry else None,
            })
            return True

        kw_normed = self._embed_kw(kw)
        gloss_sims: "Any" = self._glossary_embs @ kw_normed
        entries: List[Tuple[str, float, Dict[str, Any]]] = []
        seen_entries: set = set()
        for i, sim in enumerate(gloss_sims):
            if float(sim) >= SIM_MIN:
                t = self._glossary_texts[i]
                e = self._text_to_glossary[t.lower()]
                if e["term"] not in seen_entries:
                    seen_entries.add(e["term"])
                    entries.append((e["term"], float(sim), e))
        entries.sort(key=lambda x: -x[1])

        for term, sim, entry in entries:
            _add(term, sim, "Close in Glossary", None, entry)
            for edge in self._edges_for_glossary_entry(entry):
                rel = edge.get("relation_type", "relié")
                for neighbor in edge.get("terms", []):
                    canonical = {entry["term"].lower()} | {
                        a.lower() for a in entry.get("aliases", []) if isinstance(a, str)
                    }
                    if neighbor.lower() in canonical:
                        continue
                    neighbor_entry = self._text_to_glossary.get(neighbor.lower())
                    _add(neighbor, None, f"{rel} of {term}", term, neighbor_entry)

        return result

    def match_question(self, question: str) -> Dict[str, Any]:
        """Extract keywords from question and expand each via glossary + graph.

        Returns:
            keywords           : list of extracted keywords
            keyword_results    : list of {keyword, related_terms, has_matches}
            unmatched_keywords : keywords with no related terms found
            has_matches        : bool (any keyword has at least one related term)
        """
        kw_result = extract_keywords(question)
        keywords: List[str] = kw_result["kept"] if not kw_result["fallback"] else []

        keyword_results: List[Dict[str, Any]] = []
        unmatched: List[str] = []

        for kw in keywords:
            related = self._expand_keyword(kw)
            keyword_results.append({
                "keyword": kw,
                "related_terms": related,
                "has_matches": len(related) > 0,
            })
            if not related:
                unmatched.append(kw)

        return {
            "keywords": keywords,
            "keyword_results": keyword_results,
            "unmatched_keywords": unmatched,
            "has_matches": any(kr["has_matches"] for kr in keyword_results),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loader (with disk cache)
# ─────────────────────────────────────────────────────────────────────────────

def load_glossary_graph_matcher(
    corpus_id: str,
    glossary_path: str,
    graph_path: str,
    client: Any,
    *,
    cache_dir: str = ".cache/glossary_graph_embeddings",
    embed_model: str = _EMBED_MODEL,
) -> GlossaryGraphMatcher:
    """Load (or compute + persist) embeddings for glossary texts.

    Disk cache key: corpus_id + glossary texts list.
    Cache is invalidated automatically when glossary changes.
    """
    import numpy as np

    glossary: List[Dict[str, Any]] = json.loads(
        Path(glossary_path).read_text(encoding="utf-8")
    )
    graph: List[Dict[str, Any]] = json.loads(
        Path(graph_path).read_text(encoding="utf-8")
    )

    # Build glossary texts (term + aliases, deduplicated)
    seen_lower: set = set()
    glossary_texts: List[str] = []
    for entry in glossary:
        t = entry["term"]
        if t.lower() not in seen_lower:
            seen_lower.add(t.lower())
            glossary_texts.append(t)
        for alias in entry.get("aliases", []):
            if isinstance(alias, str) and alias.lower() not in seen_lower:
                seen_lower.add(alias.lower())
                glossary_texts.append(alias)

    cache_path = Path(cache_dir) / f"{corpus_id}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            data = np.load(str(cache_path), allow_pickle=True)
            if list(data["glossary_texts"]) == glossary_texts:
                logger.debug("Glossary embeddings loaded from cache: %s", cache_path)
                return GlossaryGraphMatcher(
                    glossary=glossary,
                    graph=graph,
                    glossary_embs=data["glossary_embs"],
                    glossary_texts=glossary_texts,
                    client=client,
                    embed_model=embed_model,
                )
            logger.info("Glossary data changed — recomputing embeddings")
        except Exception as exc:
            logger.warning("Failed to load embedding cache (%s): %s", cache_path, exc)

    logger.info(
        "Computing embeddings: %d glossary texts (model=%s)",
        len(glossary_texts), embed_model,
    )
    glossary_embs = _normalize_rows(_batch_embed(glossary_texts, client, embed_model))

    try:
        np.savez(
            str(cache_path),
            glossary_texts=np.array(glossary_texts, dtype=object),
            glossary_embs=glossary_embs,
        )
        logger.debug("Saved embeddings to: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to save embedding cache: %s", exc)

    return GlossaryGraphMatcher(
        glossary=glossary,
        graph=graph,
        glossary_embs=glossary_embs,
        glossary_texts=glossary_texts,
        client=client,
        embed_model=embed_model,
    )
