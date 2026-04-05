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


class _FakeExplorationAdvisor:
    """Deterministic exploration advisor for tests."""

    def __init__(self, advice_by_keyword: dict[str, list[str]]) -> None:
        self._advice_by_keyword = {k.lower(): v for k, v in advice_by_keyword.items()}

    def __call__(self, question: str, keyword: str, candidates):
        return {
            "suggested_terms": self._advice_by_keyword.get(keyword.lower(), []),
            "reason": "test fixture",
        }


# ── Test fixtures ──────────────────────────────────────────────────────────────
#
# Glossary: ["Wisdom", "Justice"]
# Glossary graph nodes:
#   - Wisdom ↔ Prudence  (synonym, satellite for reformulation / exploration)
#   - notes  → Wisdom    (synonym, notes not in glossary → satellite)
#   - Justice ↔ Equity   (synonym)
#
# Vector layout (4D):
#   dim 0 → "wisdom" canonical
#   dim 1 → "justice" canonical
#   dim 2 → "notes" satellite
#   dim 3 → fallback semantic probe near "wisdom" but below glossary threshold

GLOSSARY_ENTRIES = [
    {"term": "Wisdom", "importance": "High", "frequency": 10},
    {"term": "Justice", "importance": "Med", "frequency": 5},
]

GLOSSARY_GRAPH_NODES = [
    {
        "term": "Wisdom",
        "edges": [
            {
                "term_b": "Prudence",
                "relation_type": "synonym",
                "strength": "high",
                "evidence_quote": "Prudence is but wisdom by experience.",
            },
            {
                "term_b": "notes",
                "relation_type": "synonym",
                "strength": "medium",
                "evidence_quote": "Notes are signs that preserve wisdom in writing.",
            },
        ],
    },
    {
        "term": "Justice",
        "edges": [
            {
                "term_b": "Equity",
                "relation_type": "synonym",
                "strength": "high",
                "evidence_quote": "Equity is the equal distribution of what in reason belongeth to each.",
            },
            {
                "term_b": "Wisdom",
                "relation_type": "related",
                "strength": "medium",
                "evidence_quote": "Justice depends upon wisdom in counsel.",
            },
        ],
    },
]

GLOSSARY_BY_TERM = {
    "wisdom": GLOSSARY_ENTRIES[0],
    "justice": GLOSSARY_ENTRIES[1],
    "prudence": {"term": "Prudence", "importance": "Medium", "frequency": 4},
    "equity": {"term": "Equity", "importance": "High", "frequency": 3},
}

FAKE_VECTORS = {
    # canonical term embeddings (question-level only)
    "wisdom":                  [1.0,  0.0,  0.0,  0.0],
    "justice":                 [0.0,  1.0,  0.0,  0.0],
    # synonym satellites
    "prudence":                [1.0,  0.0,  0.0,  0.0],
    "notes":                   [0.0,  0.0,  1.0,  0.0],
    "equity":                  [0.0,  1.0,  0.0,  0.0],
    # probe keywords
    "wisdome":                 [0.999, 0.001, 0.0,  0.0],   # → no direct glossary match anymore
    "prudnce":                 [0.999, 0.001, 0.0,  0.0],   # → reformulation fallback to prudence satellite
    "note":                    [0.001, 0.0,   0.999, 0.0],   # → step 2 (satellite)
    "virtue":                  [0.75,  0.66,  0.0,   0.0],   # → step 3 (low-threshold canonical fallback)
    "virtue and prudence":     [0.95,  0.20,  0.0,   0.0],   # → question context still favors wisdom
    "justice and law":         [0.30,  0.98,  0.0,   0.0],   # → question context boosts justice
    "wisdom and justice":      [0.80,  0.79,  0.0,   0.0],
    "morality":                [0.25,  0.25,  0.25,  0.25],  # → no_match
    "what is wisdom?":         [1.0,   0.0,   0.0,   0.0],
}


def build_lookup(advisor=None) -> GlossaryLookup:
    return GlossaryLookup(
        glossary_entries=GLOSSARY_ENTRIES,
        graph_nodes=GLOSSARY_GRAPH_NODES,
        embedding_backend=_FakeEmbeddingBackend(FAKE_VECTORS),
        glossary_by_term=GLOSSARY_BY_TERM,
        exploration_advisor=advisor,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_exact_synonym_satellite_step() -> None:
    """Exact normalized synonym satellite should resolve to its canonical anchor."""
    lookup = build_lookup()
    result = lookup.lookup("prudence")

    assert result.matched_step == "satellite"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term == "prudence"
    assert result.best_score is not None
    assert result.best_score == 1.0


def test_exact_synonym_returns_graph_neighbors() -> None:
    """Exact synonym satellite match still returns canonical graph neighbors."""
    lookup = build_lookup()
    result = lookup.lookup("prudence")

    assert result.matched_step == "satellite"
    neighbor_terms = [n["term"] for n in result.graph_neighbors]
    assert "prudence" in neighbor_terms
    prudence_neighbor = next(n for n in result.graph_neighbors if n["term"] == "prudence")
    assert prudence_neighbor["evidence_quote"] == "Prudence is but wisdom by experience."


def test_synonym_satellite_embedding_fallback() -> None:
    """Keyword close to a synonym satellite should resolve with a high-threshold embedding fallback."""
    lookup = build_lookup()
    result = lookup.lookup("prudnce")

    assert result.matched_step == "satellite"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term == "prudence"
    assert result.best_score is not None
    assert result.best_score >= 0.92


def test_medium_strength_synonym_is_ignored() -> None:
    """Reformulation only considers synonym satellites with high link strength."""
    lookup = build_lookup()
    result = lookup.lookup("note")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.graph_neighbors == []


def test_anchor_is_blocked_when_glossary_satellite_is_more_important() -> None:
    """If the matched satellite is already in the glossary, the anchor must be at least as important."""
    lookup = build_lookup()
    result = lookup.lookup("equity")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.satellite_term is None


def test_anchor_is_allowed_when_glossary_satellite_is_not_more_important() -> None:
    """A glossary satellite can still reformulate if the anchor importance is >= satellite importance."""
    lookup = build_lookup()
    result = lookup.lookup("prudence")

    assert result.matched_step == "satellite"
    assert result.canonical_term == "wisdom"
    assert result.satellite_term == "prudence"


def test_question_exploration_adds_advice_without_anchor_resolution() -> None:
    """Question exploration keeps only eligible synonym satellites for in-question glossary terms."""
    lookup = build_lookup(
        advisor=_FakeExplorationAdvisor({"virtue": ["wisdom", "justice"]}),
    )
    result = lookup.lookup("virtue")
    exploration = lookup.suggest_question_exploration("wisdom and justice", keyword="virtue")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.best_score is None
    assert result.graph_neighbors == []
    assert result.synonym_candidates == []
    assert result.exploration_advice == {}
    assert [item["term"] for item in exploration.candidates] == ["equity"]
    assert "question_satellite" in exploration.candidates[0]["candidate_sources"]
    assert exploration.candidates[0]["keyword_score"] is None
    assert exploration.candidates[0]["question_score"] is not None
    assert exploration.exploration_advice["suggested_terms"] == ["wisdom", "justice"]


def test_question_top_glossary_candidates() -> None:
    """Question-level glossary comparisons are available separately from step 3 keyword lookup."""
    lookup = build_lookup()
    question_candidates = lookup.top_glossary_candidates("justice and law", source="question")

    assert question_candidates[0]["term"] == "justice"
    assert "question" in question_candidates[0]["candidate_sources"]
    assert question_candidates[0]["question_score"] is not None


def test_question_exploration_can_suggest_terms_without_anchor() -> None:
    """Question exploration can still suggest terms even though no anchor is resolved."""
    lookup = build_lookup(
        advisor=_FakeExplorationAdvisor({"virtue": ["wisdom"]}),
    )
    result = lookup.lookup("virtue")
    exploration = lookup.suggest_question_exploration("virtue and prudence", keyword="virtue")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.best_score is None
    assert result.synonym_candidates == []
    assert result.exploration_advice == {}
    assert [item["term"] for item in exploration.candidates[:2]] == ["wisdom", "justice"]
    assert exploration.exploration_advice["suggested_terms"] == ["wisdom"]


def test_question_exploration_skips_satellite_already_present_in_question() -> None:
    """If a top-k term is in the question, its satellites already present in the question should be excluded."""
    lookup = build_lookup()
    exploration = lookup.suggest_question_exploration("wisdom and prudence", keyword="wisdom")

    candidate_terms = [item["term"] for item in exploration.candidates]
    assert "prudence" not in candidate_terms
    assert "notes" not in candidate_terms


def test_question_exploration_filters_synonym_satellites_to_glossary_with_higher_or_equal_importance() -> None:
    """Question-level replacement keeps synonym satellites only if they are glossary terms with >= anchor importance."""
    lookup = build_lookup()
    exploration = lookup.suggest_question_exploration("justice", keyword="justice")

    candidate_terms = [item["term"] for item in exploration.candidates]
    assert candidate_terms == ["equity", "wisdom"]


def test_answer_first_exploration_expands_highly_representative_in_quote_terms() -> None:
    """When a quote already strongly covers an in-quote term, exploration should expand around it instead."""
    glossary_entries = [
        {"term": "Wisdom", "importance": "High", "frequency": 10, "quote": "wisdom"},
        {"term": "Justice", "importance": "Med", "frequency": 5, "quote": "justice"},
        {"term": "Prudence", "importance": "Medium", "frequency": 4, "quote": "prudence"},
    ]
    graph_nodes = [
        {
            "term": "Justice",
            "edges": [
                {
                    "term_b": "Wisdom",
                    "relation_type": "related",
                    "strength": "medium",
                    "evidence_quote": "Justice depends upon wisdom in counsel.",
                }
            ],
        },
        {
            "term": "Wisdom",
            "edges": [
                {
                    "term_b": "Prudence",
                    "relation_type": "synonym",
                    "strength": "high",
                    "evidence_quote": "Prudence is but wisdom by experience.",
                }
            ],
        },
    ]
    glossary_by_term = {entry["term"].lower(): entry for entry in glossary_entries}
    vectors = {
        "wisdom": [1.0, 0.0, 0.0],
        "justice": [0.0, 1.0, 0.0],
        "prudence": [0.8, 0.2, 0.0],
        "what should we explore next?": [0.6, 0.4, 0.0],
    }
    lookup = GlossaryLookup(
        glossary_entries=glossary_entries,
        graph_nodes=graph_nodes,
        embedding_backend=_FakeEmbeddingBackend(vectors),
        glossary_by_term=glossary_by_term,
        exploration_advisor=None,
    )

    exploration = lookup.suggest_question_exploration(
        "What should we explore next?",
        answer_context={"answer_quotes": [{"text": "wisdom", "ref": "Q1"}]},
    )

    candidate_terms = [item["term"] for item in exploration.candidates]
    assert exploration.source_mode == "answer"
    assert "wisdom" not in candidate_terms
    assert "justice" in candidate_terms
    assert exploration.quote_pools


def test_answer_first_exploration_scores_each_sentence_separately() -> None:
    """Answer-first exploration should build one pool per sentence, not one diluted pool per full quote."""
    glossary_entries = [
        {"term": "Warre", "importance": "High", "frequency": 10},
        {"term": "Peace", "importance": "High", "frequency": 10},
    ]
    glossary_by_term = {entry["term"].lower(): entry for entry in glossary_entries}
    vectors = {
        "warre": [1.0, 0.0],
        "peace": [0.0, 1.0],
        "warre.": [1.0, 0.0],
        "peace.": [0.0, 1.0],
        "what should we explore next?": [0.7, 0.7],
    }
    lookup = GlossaryLookup(
        glossary_entries=glossary_entries,
        graph_nodes=[],
        embedding_backend=_FakeEmbeddingBackend(vectors),
        glossary_by_term=glossary_by_term,
        exploration_advisor=None,
    )

    exploration = lookup.suggest_question_exploration(
        "What should we explore next?",
        answer_context={"answer_quotes": [{"text": "Warre. Peace.", "ref": "Q1"}]},
    )

    assert exploration.source_mode == "answer"
    assert len(exploration.quote_pools) == 2
    assert exploration.quote_pools[0]["quote_text"] == "Warre."
    assert exploration.quote_pools[1]["quote_text"] == "Peace."
    assert exploration.quote_pools[0]["candidates"][0]["term"] == "warre"
    assert exploration.quote_pools[1]["candidates"][0]["term"] == "peace"


def test_question_exploration_never_sends_non_glossary_satellites() -> None:
    """Question-level exploration should only forward glossary satellites to the LLM shortlist."""
    lookup = build_lookup()
    exploration = lookup.suggest_question_exploration("wisdom", keyword="wisdom")

    candidate_terms = [item["term"] for item in exploration.candidates]
    assert "notes" not in candidate_terms


def test_question_exploration_adds_non_synonym_reverse_anchors() -> None:
    """If a question term appears as a non-synonym satellite, its anchor should be added to exploration."""
    lookup = build_lookup()
    exploration = lookup.suggest_question_exploration("wisdom", keyword="wisdom")

    justice = next(item for item in exploration.candidates if item["term"] == "justice")
    assert "question_reverse_anchor" in justice["candidate_sources"]
    assert justice["link_type"] == "related"
    assert justice["quote"] == "Justice depends upon wisdom in counsel."


def test_question_reformulation_expands_glossary_terms_to_synonyms() -> None:
    """Question-level reformulation should expand an in-question glossary term to its graph synonyms."""
    lookup = build_lookup()
    reformulation = lookup.suggest_question_reformulation("wisdom", ["wisdom"])

    terms = [item["term"] for item in reformulation.candidates]
    assert "prudence" in terms
    prudence = next(item for item in reformulation.candidates if item["term"] == "prudence")
    assert "reformulation_synonym" in prudence["candidate_sources"]
    assert prudence["source_anchor"] == "wisdom"
    assert prudence["final_score"] > 0


def test_question_reformulation_maps_synonym_satellite_to_anchor() -> None:
    """Question-level reformulation should map an in-question synonym satellite back to its glossary anchor."""
    lookup = build_lookup()
    reformulation = lookup.suggest_question_reformulation("prudence", ["prudence"])

    terms = [item["term"] for item in reformulation.candidates]
    assert "wisdom" in terms
    wisdom = next(item for item in reformulation.candidates if item["term"] == "wisdom")
    assert "reformulation_anchor" in wisdom["candidate_sources"]
    assert wisdom["source_satellite"] == "prudence"
    assert wisdom["best_keyword_score"] is not None


def test_question_reformulation_generates_candidate_sentences_from_span_matches() -> None:
    lookup = build_lookup()
    reformulation = lookup.suggest_question_reformulation("What is wisdom?", ["wisdom"])

    terms = [item["term"] for item in reformulation.candidates[:4]]
    assert "prudence" in terms
    assert reformulation.span_term_matches
    generated_texts = [item["text"] for item in reformulation.generated_candidates]
    assert "What is prudence?" in generated_texts


def test_no_match() -> None:
    """Keyword far from everything → matched_step='no_match'."""
    lookup = build_lookup()
    result = lookup.lookup("morality")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.best_score is None
    assert result.graph_neighbors == []


def test_no_question_exploration_inside_keyword_lookup() -> None:
    """Keyword lookup no longer performs question-level exploration implicitly."""
    lookup = build_lookup()
    result = lookup.lookup("virtue")

    assert result.matched_step == "no_match"
    assert result.canonical_term is None
    assert result.best_score is None
    assert result.graph_neighbors == []
    assert result.synonym_candidates == []
    assert result.exploration_advice == {}


def test_lookup_keywords_batch() -> None:
    """lookup_keywords runs strict reformulation over synonym satellites only."""
    lookup = build_lookup()
    results = lookup.lookup_keywords(["memory", "morality", "note"])

    assert len(results) == 3
    assert results[0].matched_step == "no_match"
    assert results[1].matched_step == "no_match"
    assert results[2].matched_step == "no_match"


def test_result_cache() -> None:
    """Same keyword returns cached result (same object)."""
    lookup = build_lookup()
    r1 = lookup.lookup("prudence")
    r2 = lookup.lookup("prudence")
    assert r1 is r2
