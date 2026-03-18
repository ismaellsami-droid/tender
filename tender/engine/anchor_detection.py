from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Callable backend that returns one embedding vector per input text."""

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


@dataclass(frozen=True)
class LookupResult:
    """Result of a glossary lookup for one keyword."""

    keyword: str
    matched_step: str        # "glossary" | "satellite" | "synonym" | "no_match"
    canonical_term: str | None
    satellite_term: str | None   # step 2 only: satellite term that triggered the match
    best_score: float | None
    graph_neighbors: list[dict] = field(default_factory=list)
    # Each neighbor: {term, link_type, is_glossary, importance, frequency, strength}


class GlossaryLookup:
    """3-step elsif glossary lookup using cosine similarity only.

    Step 1 — Glossary (threshold 0.92):
        embed(keyword) vs canonical glossary term embeddings
        → match: return canonical term + all its glossary_graph neighbors

    Step 2 — Satellites (threshold 0.92):
        embed(keyword) vs all term_b values from the graph
        → match: resolve to the source canonical term + its neighbors
        (reached only when step 1 failed, so no need to filter canonical term_b)

    Step 3 — Synonyms (threshold 0.60):
        embed(keyword) vs enriched descriptor embeddings
        (canonical term + its synonym/definitional neighbors as a text string)
        → match: return the best canonical term
    """

    GLOSSARY_THRESHOLD: float = 0.92
    SATELLITE_THRESHOLD: float = 0.92
    SYNONYM_THRESHOLD: float = 0.60

    def __init__(
        self,
        glossary_entries: list[dict],
        graph_nodes: list[dict],
        embedding_backend: EmbeddingBackend,
        glossary_by_term: dict,
    ) -> None:
        self._embedding_backend = embedding_backend
        self._glossary_by_term = glossary_by_term
        self._result_cache: dict[str, LookupResult] = {}

        # ── Step 1: canonical term embeddings ────────────────────────────────
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

        self._canonical_embeddings = self._build_embeddings(self._canonical_terms)

        # ── Step 2: satellite map ─────────────────────────────────────────────
        # All term_b across all edges → their source canonical term.
        # No filtering needed: step 2 is only reached when step 1 failed.
        satellite_map: dict[str, str] = {}
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
                if tb_l not in satellite_map:
                    satellite_map[tb_l] = ta_l

        self._satellite_terms: list[str] = list(satellite_map.keys())
        self._satellite_to_canonical: dict[str, str] = satellite_map
        if self._satellite_terms:
            self._satellite_embeddings = self._build_embeddings(self._satellite_terms)
        else:
            self._satellite_embeddings = np.zeros((0, 1), dtype=np.float32)

        # ── Adjacency: canonical_lower → neighbors ────────────────────────────
        adjacency: dict[str, list[dict]] = {}
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
                adjacency.setdefault(ta_l, []).append({
                    "term": tb_l,
                    "link_type": str(edge.get("relation_type", "related")),
                    "is_glossary": tb_l in canonical_seen,
                    "strength": edge.get("strength"),
                })

        self._adjacency = adjacency

        # ── Step 3: enriched descriptor embeddings ────────────────────────────
        descriptors: list[str] = []
        for term in self._canonical_terms:
            neighbors = adjacency.get(term, [])
            enrichment = [
                n["term"] for n in neighbors
                if n["link_type"] in ("synonym", "definitional")
            ]
            descriptor = term + (" " + " ".join(enrichment) if enrichment else "")
            descriptors.append(descriptor)
        self._descriptor_embeddings = self._build_embeddings(descriptors)

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, keyword: str) -> LookupResult:
        """Run the 3-step elsif lookup for one keyword."""
        normalized = keyword.strip().lower()
        if not normalized:
            return LookupResult(
                keyword=keyword, matched_step="no_match",
                canonical_term=None, satellite_term=None, best_score=None,
            )

        cached = self._result_cache.get(normalized)
        if cached is not None:
            return cached

        kw_vec = self._embed_single(normalized)

        # Step 1 — Glossary
        if self._canonical_terms:
            sims = self._canonical_embeddings @ kw_vec
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.GLOSSARY_THRESHOLD:
                canonical = self._canonical_terms[best_idx]
                result = LookupResult(
                    keyword=keyword,
                    matched_step="glossary",
                    canonical_term=canonical,
                    satellite_term=None,
                    best_score=round(best_score, 6),
                    graph_neighbors=self._get_graph_neighbors(canonical),
                )
                self._result_cache[normalized] = result
                return result

        # Step 2 — Satellites
        if self._satellite_terms:
            sims = self._satellite_embeddings @ kw_vec
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.SATELLITE_THRESHOLD:
                satellite = self._satellite_terms[best_idx]
                canonical = self._satellite_to_canonical[satellite]
                result = LookupResult(
                    keyword=keyword,
                    matched_step="satellite",
                    canonical_term=canonical,
                    satellite_term=satellite,
                    best_score=round(best_score, 6),
                    graph_neighbors=self._get_graph_neighbors(canonical),
                )
                self._result_cache[normalized] = result
                return result

        # Step 3 — Synonyms via enriched descriptors
        if self._canonical_terms:
            sims = self._descriptor_embeddings @ kw_vec
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= self.SYNONYM_THRESHOLD:
                canonical = self._canonical_terms[best_idx]
                result = LookupResult(
                    keyword=keyword,
                    matched_step="synonym",
                    canonical_term=canonical,
                    satellite_term=None,
                    best_score=round(best_score, 6),
                    graph_neighbors=[],
                )
                self._result_cache[normalized] = result
                return result

        result = LookupResult(
            keyword=keyword, matched_step="no_match",
            canonical_term=None, satellite_term=None, best_score=None,
        )
        self._result_cache[normalized] = result
        return result

    def lookup_keywords(self, keywords: list[str]) -> list[LookupResult]:
        """Batch lookup."""
        return [self.lookup(kw) for kw in keywords]

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
            })
        return neighbors

    def _embed_single(self, text: str) -> np.ndarray:
        raw = np.asarray(self._embedding_backend([text]), dtype=np.float32)
        return _normalize_vectors(raw)[0]

    def _build_embeddings(self, texts: list[str]) -> np.ndarray:
        raw = np.asarray(self._embedding_backend(texts), dtype=np.float32)
        return _normalize_vectors(raw)


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
