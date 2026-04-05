from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LookupResult:
    """Result of a glossary lookup for one keyword."""

    keyword: str
    matched_step: str
    canonical_term: str | None
    satellite_term: str | None
    best_score: float | None
    graph_neighbors: list[dict] = field(default_factory=list)
    synonym_candidates: list[dict[str, Any]] = field(default_factory=list)
    exploration_advice: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionExplorationResult:
    question: str
    source_mode: str = "question"
    initial_candidates: list[dict[str, Any]] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)
    span_pools: list[dict[str, Any]] = field(default_factory=list)
    quote_pools: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    exploration_advice: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionReformulationResult:
    question: str
    keywords: list[str] = field(default_factory=list)
    initial_candidates: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)
    span_pools: list[dict[str, Any]] = field(default_factory=list)
    span_term_matches: list[dict[str, Any]] = field(default_factory=list)
    generated_candidates: list[dict[str, Any]] = field(default_factory=list)
