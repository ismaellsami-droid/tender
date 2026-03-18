from __future__ import annotations

from tender.engine.anchor_detection import GlossaryLookup, LookupResult


class _FakeEmbeddingBackend:
    """Deterministic embedding backend for tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = {k.lower(): v for k, v in vectors.items()}
        self._default_dim = max((len(v) for v in self._vectors.values()), default=4)

    def __call__(self, texts):
        return [
            self._vectors.get(t.lower(), [0.0] * self._default_dim)
            for t in texts
        ]


# ── Test fixtures ──────────────────────────────────────────────────────────────
#
# Glossary: ["Wisdom", "Justice"]
# Glossary graph nodes:
#   - Wisdom ↔ Prudence  (synonym, both in glossary)
#   - notes  → Wisdom    (synonym, notes not in glossary → satellite)
#
# Descriptor for "wisdom"  = "wisdom prudence notes"  (term + synonym neighbors)
# Descriptor for "justice" = "justice"                (no neighbors)
#
# Vector layout (4D):
#   dim 0 → "wisdom" canonical
#   dim 1 → "justice" canonical
#   dim 2 → "notes" satellite
#   dim 3 → "wisdom prudence notes" enriched descriptor

GLOSSARY_ENTRIES = [
    {"term": "Wisdom", "importance": "High", "frequency": 10},
    {"term": "Justice", "importance": "Med", "frequency": 5},
]

GLOSSARY_GRAPH_NODES = [
    {
        "term": "Wisdom",
        "edges": [
            {"term_b": "Prudence", "relation_type": "synonym", "strength": 0.93},
            {"term_b": "notes", "relation_type": "synonym", "strength": 0.87},
        ],
    },
    {
        "term": "Justice",
        "edges": [],
    },
]

GLOSSARY_BY_TERM = {
    "wisdom": GLOSSARY_ENTRIES[0],
    "justice": GLOSSARY_ENTRIES[1],
}

FAKE_VECTORS = {
    # canonical term embeddings (step 1)
    "wisdom":                  [1.0,  0.0,  0.0,  0.0],
    "justice":                 [0.0,  1.0,  0.0,  0.0],
    # satellite embedding (step 2)
    "notes":                   [0.0,  0.0,  1.0,  0.0],
    # enriched descriptor embeddings (step 3)
    "wisdom prudence notes":   [0.0,  0.0,  0.0,  1.0],
    "justice":                 [0.0,  1.0,  0.0,  0.0],  # no enrichment
    # probe keywords
    "wisdome":                 [0.999, 0.001, 0.0,  0.0],   # → step 1 (glossary)
    "note":                    [0.001, 0.0,   0.999, 0.0],   # → step 2 (satellite)
    "virtue":                  [0.001, 0.0,   0.001, 0.999], # → step 3 (synonym)
    "morality":                [0.25,  0.25,  0.25,  0.25],  # → no_match
}


def build_lookup() -> GlossaryLookup:
    return GlossaryLookup(
        glossary_entries=GLOSSARY_ENTRIES,
        graph_nodes=GLOSSARY_GRAPH_NODES,
        embedding_backend=_FakeEmbeddingBackend(FAKE_VECTORS),
        glossary_by_term=GLOSSARY_BY_TERM,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_glossary_step() -> None:
    """Keyword close to a canonical term → matched_step='glossary'."""
    lookup = build_lookup()
    result = lookup.lookup("wisdome")

    assert result.matched_step == "glossary"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term is None
    assert result.best_score is not None
    assert result.best_score >= 0.92


def test_glossary_returns_graph_neighbors() -> None:
    """Step 1 match returns the canonical term's glossary_graph neighbors."""
    lookup = build_lookup()
    result = lookup.lookup("wisdome")

    assert result.matched_step == "glossary"
    neighbor_terms = [n["term"] for n in result.graph_neighbors]
    assert "prudence" in neighbor_terms


def test_satellite_step() -> None:
    """Keyword close to a satellite term (not canonical) → matched_step='satellite'."""
    lookup = build_lookup()
    result = lookup.lookup("note")

    assert result.matched_step == "satellite"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term == "notes"
    assert result.best_score is not None
    assert result.best_score >= 0.92


def test_satellite_returns_graph_neighbors() -> None:
    """Step 2 match expands to the linked canonical term's neighbors."""
    lookup = build_lookup()
    result = lookup.lookup("note")

    assert result.matched_step == "satellite"
    neighbor_terms = [n["term"] for n in result.graph_neighbors]
    assert "prudence" in neighbor_terms


def test_synonym_step() -> None:
    """Keyword close to an enriched descriptor → matched_step='synonym', no neighbors."""
    lookup = build_lookup()
    result = lookup.lookup("virtue")

    assert result.matched_step == "synonym"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term is None
    assert result.best_score is not None
    assert result.best_score >= 0.60
    assert result.graph_neighbors == []


def test_no_match() -> None:
    """Keyword far from everything → matched_step='no_match'."""
    lookup = build_lookup()
    result = lookup.lookup("morality")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.best_score is None
    assert result.graph_neighbors == []


def test_lookup_keywords_batch() -> None:
    """lookup_keywords runs the full chain for each keyword."""
    lookup = build_lookup()
    results = lookup.lookup_keywords(["wisdome", "morality", "note"])

    assert len(results) == 3
    assert results[0].matched_step == "glossary"
    assert results[1].matched_step == "no_match"
    assert results[2].matched_step == "satellite"


def test_result_cache() -> None:
    """Same keyword returns cached result (same object)."""
    lookup = build_lookup()
    r1 = lookup.lookup("wisdome")
    r2 = lookup.lookup("wisdome")
    assert r1 is r2
