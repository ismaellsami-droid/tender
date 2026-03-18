#!/usr/bin/env python3
# Archived: legacy graph-generation workflow.
"""
build_graph.py — Build a concept graph of explicit term links from a corpus.

Detects two link types:
  synonym      — Hobbes equates two glossary terms with "X, or Y" patterns
  definitional — A glossary term is defined; other glossary terms appear in the definition

Usage:
    python tender/tools/build_graph.py \\
        --corpus-dir books/leviathan_book01/output \\
        --glossary   books/leviathan_book01/terms_extracted/terms_final_merged.json \\
        --output     books/leviathan_book01/concept_graph.json

    # Dry-run on first 3 files only:
    python tender/tools/build_graph.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Configurable pattern lists ────────────────────────────────────────────────

# Synonym connectors: the FULL text between two adjacent glossary terms
# must fullmatch one of these (modulo surrounding whitespace) to create a link.
SYNONYM_CONNECTORS: List[str] = [
    # ── Classical (Hobbes / formal prose) ──────────────────────────────────────
    r",?\s+or\s+",                               # "X, or Y"  /  "X or Y"
    r",?\s+and\s+sometimes\s+",                  # "X, and sometimes Y"
    r",?\s+that\s+is,?\s+",                      # "X, that is, Y"
    r",?\s+or\s+as\s+it\s+is\s+called,?\s+",    # "X, or as it is called, Y"
    r"\s+which\s+is\s+also\s+called\s+",         # "X which is also called Y"
    r",?\s+also\s+called\s+",                    # "X, also called Y"
    r",?\s+or\s+in\s+other\s+words,?\s+",        # "X, or in other words, Y"
    r",?\s+otherwise\s+called\s+",               # "X, otherwise called Y"
    r";?\s+and\s+sometimes\s+",                  # "X; and sometimes Y"
    # ── Modern / academic ──────────────────────────────────────────────────────
    r",?\s+also\s+known\s+as\s+",               # "X, also known as Y"
    r",?\s+or\s+rather\s+",                     # "X, or rather Y"
    r",?\s+i\.e\.?,?\s+",                       # "X, i.e., Y"  /  "X, i.e. Y"
    r",?\s+a\.k\.a\.?,?\s+",                    # "X, a.k.a. Y"
    r",?\s+namely\s+",                           # "X, namely Y"
    r",?\s+or\s+simply\s+",                     # "X, or simply Y"
    r",?\s+which\s+(?:we\s+)?(?:also\s+)?call\s+",  # "X, which we call Y" / "X, which call Y"
    # ── Parenthetical ──────────────────────────────────────────────────────────
    r"\s*\(\s*or\s+",                           # "X (or Y"     — closing ) swallowed by _GAP_OK
    r"\s*\(\s*i\.e\.?,?\s+",                    # "X (i.e. Y"
    r"\s*\(\s*also\s+called\s+",                # "X (also called Y"
    r"\s*\(\s*also\s+known\s+as\s+",            # "X (also known as Y"
    r"\s*\(\s*a\.k\.a\.?,?\s+",                 # "X (a.k.a. Y"
    r"\s*\(\s*sometimes\s+called\s+",           # "X (sometimes called Y"
    # ── Em-dash / colon inline ─────────────────────────────────────────────────
    r"\s+[—–]\s+",                              # "X — Y"  (em-dash / en-dash)
]

# Definitional lexical markers — presence in sentence triggers definitional check.
# Ordered from most specific to most generic (to assign confidence correctly).
DEF_MARKERS_HIGH: List[str] = [
    # ── Classical (Hobbes) ─────────────────────────────────────────────────────
    r"\bis\s+called\b",
    r"\bare\s+called\b",
    r"\bI\s+call\b",
    r"\bwe\s+call\b",
    r"\bis\s+defined\s+as\b",
    r"\bsignifieth\b",
    r"\bdenoteth\b",
    r"\bby\s+which\s+I\s+mean\b",
    r"\bI\s+understand\s+by\b",
    r"\bthat\s+is\s+to\s+say\b",
    r"\bconsisteth\s+(?:in|of)\b",
    r"\bis\s+nothing\s+(?:else\s+)?but\b",
    # ── Modern / academic ──────────────────────────────────────────────────────
    r"\brefers?\s+to\b",                        # "X refers to Y"
    r"\breferred\s+to\s+as\b",                  # "referred to as X"
    r"\bknown\s+as\b",                          # "known as X"
    r"\bdefined\s+as\b",                        # "X defined as Y" (broader than "is defined as")
    r"\bwhat\s+(?:we|I|they)\s+call\b",         # "what we call X"
    r"\bwhat\s+is\s+(?:often\s+)?called\b",     # "what is called X"
    r"\bin\s+other\s+words\b",                  # "in other words, X"
    r"\bthat\s+is\b",                           # "X, that is, Y"  (modern shorthand)
    r"\bnamely\b",                              # "X, namely Y"
    r"\bwhich\s+(?:we\s+)?(?:also\s+)?call\b",  # "X, which we call Y"
    r"\bor\s+what\s+(?:we\s+)?call\b",          # "X, or what we call Y"
    r"\bmeans?\s+(?:that\s+)?(?:a\s+|an\s+|the\s+)?(?=[A-Z\"])",  # "X means Y" (capitalised Y only)
    r"\bwhich\s+means\b",                       # "X, which means Y"
    r"\bwhich\s+refers\s+to\b",                 # "X, which refers to Y"
    r"\bconsists?\s+(?:in|of)\b",               # "X consists of Y"  (modern form of consisteth)
    r"\bencompasses?\b",                        # "X encompasses Y"
    r"\bdenotes?\b",                            # "X denotes Y"  (modern form of denoteth)
    r"\bsignifies?\b",                          # "X signifies Y"  (modern form of signifieth)
]

# Low-confidence definitional markers: "is a/the" appears everywhere.
# Only included when --min-confidence=low is passed explicitly.
DEF_MARKERS_LOW: List[str] = [
    r"\bis\s+a\b",
    r"\bare\s+a\b",
    r"\bis\s+the\b",
    r"\bare\s+the\b",
    r"\bmeans?\b",    # "X means Y" without capitalisation guard (very noisy)
    r"\binvolves?\b", # "X involves Y"
]

# Antonym / contrast markers — for LLM qualify pass 2
ANTONYM_MARKERS: List[str] = [
    r"\bcontrasted\s+with\b",
    r"\bopposed\s+to\b",
    r"\bunlike\b",
    r"\bwhereas\b",
    r"\bbut\s+not\b",
    r"\bas\s+opposed\s+to\b",
    r"\bin\s+contrast\s+(?:to|with)\b",
    r"\bdistinguished\s+from\b",
    r"\bdiffereth?\s+from\b",        # Hobbes: "differeth from"
    r"\bdiffer\s+from\b",
    r"\bopposite\s+(?:to|of)\b",
    r"\bcontrary\s+to\b",
    r"\bsave\s+only\b",              # Hobbes: "save only that"
]

# Argumentative / logical dependency markers — for LLM qualify pass 4
ARG_MARKERS: List[str] = [
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bconsequently\b",
    r"\brequires?\b",
    r"\bpresupposes?\b",
    r"\bfollows?\s+from\b",
    r"\bentails?\b",
    r"\bby\s+consequence\b",         # Hobbes: "by consequence"
    r"\bby\s+reason\s+of\b",
    r"\bfor\s+(?:this|that|the)\s+reason\b",
    r"\bwhich\s+is\s+the\s+reason\b",
    r"\bwithout\s+which\b",          # "X without which Y cannot"
    r"\bdepends?\s+(?:on|upon)\b",
    r"\bproceeds?\s+from\b",
]

# Generic co-occurrence / association markers — for LLM qualify pass 5
RELATED_MARKERS: List[str] = [
    r"\bassociated\s+with\b",
    r"\bconnected\s+(?:to|with)\b",
    r"\balong(?:side)?\b",
    r"\btogether\s+with\b",
    r"\brelated\s+to\b",
    r"\bpertains?\s+to\b",
    r"\bbelongs?\s+to\b",
    r"\bin\s+conjunction\s+with\b",
    r"\bcomprehends?\b",             # Hobbes: "X comprehends Y"
    r"\bincludeth\b",                # Hobbes: "includeth"
    r"\bincludes?\b",
    r"\bcomprises?\b",
    r"\bsubject\s+to\b",
    r"\bpart\s+of\b",
]

# ─── Pre-compile ───────────────────────────────────────────────────────────────

# Scan the full sentence for connector occurrences (no anchors)
_CONN_RES    = [re.compile(c, re.IGNORECASE) for c in SYNONYM_CONNECTORS]
_DEF_HIGH    = [re.compile(m, re.IGNORECASE) for m in DEF_MARKERS_HIGH]
_DEF_LOW     = [re.compile(m, re.IGNORECASE) for m in DEF_MARKERS_LOW]
_ANT_RES     = [re.compile(m, re.IGNORECASE) for m in ANTONYM_MARKERS]
_ARG_RES     = [re.compile(m, re.IGNORECASE) for m in ARG_MARKERS]
_RELATED_RES = [re.compile(m, re.IGNORECASE) for m in RELATED_MARKERS]

# Sentence splitter: split on .!? followed by whitespace + uppercase
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"])')

# Raw-word fallback for synonym edges outside the glossary
_RAW_LEFT  = re.compile(r'\b([A-Za-z][A-Za-z\'-]*)\s*$')
_RAW_RIGHT = re.compile(r'^\s*([A-Za-z][A-Za-z\'-]*)\b')
# Allowed "gap" between a glossary term and a connector (punctuation / whitespace)
_GAP_OK    = re.compile(r'^[\s,;:!?()\[\]"\'-]*$')

# ─── Glossary loading ──────────────────────────────────────────────────────────

def load_glossary(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Load glossary JSON (filter to llm_label == KEEP).
    Returns:
        terms     — list of canonical term strings
        alias_map — {lowercase surface form → canonical term}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    terms: List[str] = []
    alias_map: Dict[str, str] = {}

    for rec in data:
        term: str = rec["term"]
        terms.append(term)
        alias_map[term.lower()] = term
        for alias in rec.get("aliases") or []:
            if alias and alias.strip():
                alias_map[alias.strip().lower()] = term

    return terms, alias_map


# ─── Term index & matching ─────────────────────────────────────────────────────

def build_term_patterns(
    alias_map: Dict[str, str],
) -> List[Tuple[re.Pattern, str]]:
    """
    Return list of (compiled_regex, canonical_term) sorted by surface length
    descending so longer/more-specific matches win over shorter ones.
    """
    entries = sorted(alias_map.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    patterns: List[Tuple[re.Pattern, str]] = []
    for surface, canonical in entries:
        words = surface.split()
        escaped = r"\s+".join(re.escape(w) for w in words)
        pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
        patterns.append((pat, canonical))
    return patterns


def build_quick_filter(alias_map: Dict[str, str]) -> Set[str]:
    """Set of all individual lowercase words that appear in any term/alias."""
    tokens: Set[str] = set()
    for surface in alias_map:
        for word in surface.split():
            tokens.add(word)
    return tokens


def find_terms_in_sentence(
    sentence: str,
    patterns: List[Tuple[re.Pattern, str]],
) -> List[Tuple[str, int, int]]:
    """
    Returns [(canonical_term, start, end), …] for all non-overlapping matches,
    preferring longer matches (patterns are sorted longest-first).
    Result is sorted by start position.
    """
    used: Set[int] = set()
    matches: List[Tuple[str, int, int]] = []
    for pat, canonical in patterns:
        for m in pat.finditer(sentence):
            span = set(range(m.start(), m.end()))
            if span & used:
                continue
            used.update(span)
            matches.append((canonical, m.start(), m.end()))
    return sorted(matches, key=lambda x: x[1])


# ─── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split text into sentences; preserve complete sentences."""
    # First split on newlines, then on sentence-ending punctuation
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sentences: List[str] = []
    for line in lines:
        parts = _SENT_RE.split(line)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


# ─── Synonym detection ─────────────────────────────────────────────────────────

def _left_term(
    sentence: str,
    conn_start: int,
    term_matches: List[Tuple[str, int, int]],
) -> Tuple[Optional[str], bool, int, int]:
    """
    Find the term ending just before conn_start.
    Returns (term, in_glossary, span_start, span_end).
    Prefers glossary matches; falls back to last raw word.
    """
    best_canonical: Optional[str] = None
    best_ts = best_te = 0
    for canonical, ts, te in term_matches:
        if te > conn_start:
            continue
        if not _GAP_OK.match(sentence[te:conn_start]):
            continue
        if te > best_te:
            best_canonical, best_ts, best_te = canonical, ts, te
    if best_canonical:
        return best_canonical, True, best_ts, best_te
    # Fallback: last word before connector.
    # Must be capitalised (Hobbes capitalises technical terms) and ≥ 4 chars.
    m = _RAW_LEFT.search(sentence[:conn_start])
    if m and len(m.group(1)) >= 4 and m.group(1)[0].isupper():
        start = m.start(1)
        return m.group(1).lower(), False, start, start + len(m.group(1))
    return None, False, 0, 0


def _right_term(
    sentence: str,
    conn_end: int,
    term_matches: List[Tuple[str, int, int]],
) -> Tuple[Optional[str], bool, int, int]:
    """
    Find the term starting just after conn_end.
    Returns (term, in_glossary, span_start, span_end).
    Prefers glossary matches; falls back to first raw word.
    """
    best_canonical: Optional[str] = None
    best_ts = best_te = len(sentence)
    for canonical, ts, te in term_matches:
        if ts < conn_end:
            continue
        if not _GAP_OK.match(sentence[conn_end:ts]):
            continue
        if ts < best_ts:
            best_canonical, best_ts, best_te = canonical, ts, te
    if best_canonical:
        return best_canonical, True, best_ts, best_te
    # Fallback: first word after connector.
    # Must be capitalised and ≥ 4 chars.
    m = _RAW_RIGHT.match(sentence[conn_end:])
    if m and len(m.group(1)) >= 4 and m.group(1)[0].isupper():
        start = conn_end + m.start(1)
        return m.group(1).lower(), False, start, start + len(m.group(1))
    return None, False, len(sentence), len(sentence)


def find_synonym_links(
    sentence: str,
    term_matches: List[Tuple[str, int, int]],
    source_file: str,
) -> List[Dict[str, Any]]:
    """
    Scan sentence for synonym connectors.  For each hit, resolve the term on
    each side (glossary match preferred; raw word fallback).
    Both terms may be outside the glossary — flagged via *_in_glossary fields.
    """
    links: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()  # dedup within the same sentence

    for conn_re in _CONN_RES:
        for m in conn_re.finditer(sentence):
            conn_start, conn_end = m.start(), m.end()

            left_term, left_in_gloss, left_s, left_e = _left_term(
                sentence, conn_start, term_matches
            )
            right_term, right_in_gloss, right_s, right_e = _right_term(
                sentence, conn_end, term_matches
            )

            if not left_term or not right_term:
                continue
            if left_term.lower() == right_term.lower():
                continue
            # Require at least one side to be a known glossary term
            if not left_in_gloss and not right_in_gloss:
                continue

            pair = (min(left_term.lower(), right_term.lower()),
                    max(left_term.lower(), right_term.lower()))
            if pair in seen:
                continue
            seen.add(pair)

            links.append({
                "term_a": left_term,
                "term_a_in_glossary": left_in_gloss,
                "term_b": right_term,
                "term_b_in_glossary": right_in_gloss,
                "type": "synonym",
                "direction": "bidirectional",
                "weight": 1.0,
                "quote": sentence[left_s:right_e].strip(),
                "source_file": source_file,
                "confidence": "high",
            })
    return links


# ─── Definitional detection — lexical ─────────────────────────────────────────

def _def_marker_positions(
    sentence: str,
    include_low: bool = False,
) -> Tuple[Optional[int], str]:
    """
    Return (position_of_first_marker, confidence) for the earliest definitional
    marker found, or (None, '') if none.
    confidence is "high" (specific markers) or "low" (generic "is a/the").
    """
    earliest: Optional[int] = None
    conf = ""
    for marker_re in _DEF_HIGH:
        m = marker_re.search(sentence)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
            conf = "high"
    if earliest is None and include_low:
        for marker_re in _DEF_LOW:
            m = marker_re.search(sentence)
            if m and (earliest is None or m.start() < earliest):
                earliest = m.start()
                conf = "low"
    return earliest, conf


def _find_subject_term(
    marker_pos: int,
    term_matches: List[Tuple[str, int, int]],
    sentence: str,
) -> Optional[Tuple[str, int, int]]:
    """
    Find the most plausible 'subject' (term being defined):
    1. The term whose span ends closest BEFORE the marker (preceding subject).
    2. Or the term whose span starts closest AFTER the marker (post-copula subject,
       e.g. "is called PRUDENCE").
    Returns (canonical, start, end) or None.
    """
    best_pre: Optional[Tuple[str, int, int]] = None
    best_post: Optional[Tuple[str, int, int]] = None

    for tm in term_matches:
        _, ts, te = tm
        if te <= marker_pos:
            if best_pre is None or te > best_pre[2]:
                best_pre = tm
        elif ts > marker_pos:
            if best_post is None or ts < best_post[1]:
                best_post = tm

    # Prefer term immediately before marker; fall back to first term after.
    return best_pre or best_post


def find_definitional_links(
    sentence: str,
    term_matches: List[Tuple[str, int, int]],
    source_file: str,
    include_low: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect definitional links via lexical markers.
    term_a = the defined term (subject)
    term_b = all other glossary terms in the sentence (used in definition)
    """
    if len(term_matches) < 2:
        return []

    marker_pos, confidence = _def_marker_positions(sentence, include_low=include_low)
    if marker_pos is None:
        return []

    subject = _find_subject_term(marker_pos, term_matches, sentence)
    if subject is None:
        return []

    subject_term = subject[0]
    links: List[Dict[str, Any]] = []
    for tm in term_matches:
        if tm[0] == subject_term:
            continue
        links.append({
            "term_a": subject_term,
            "term_a_in_glossary": True,
            "term_b": tm[0],
            "term_b_in_glossary": True,
            "type": "definitional",
            "direction": "b_used_to_define_a",
            "weight": 0.6,
            "quote": sentence.strip(),
            "source_file": source_file,
            "confidence": confidence,
        })
    return links


# ─── Definitional detection — spaCy grammatical ───────────────────────────────

_COPULAR_LEMMAS = {"be", "signify", "denote", "constitute", "call", "mean"}


def find_definitional_links_spacy(
    sentence: str,
    term_matches: List[Tuple[str, int, int]],
    source_file: str,
    nlp: Any,
) -> List[Dict[str, Any]]:
    """
    Use spaCy dependency parse to find [subject_term] [copula] [complement] structure.
    Only called when lexical method found no links (avoids double-counting).
    """
    if len(term_matches) < 2:
        return []

    try:
        doc = nlp(sentence)
    except Exception:
        return []

    # Find nsubj → copular verb pairs
    subject_term: Optional[str] = None
    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        head = token.head
        if head.lemma_.lower() not in _COPULAR_LEMMAS:
            continue
        # Check if the subject token overlaps with any glossary term match
        for canonical, ts, te in term_matches:
            if ts <= token.idx < te:
                subject_term = canonical
                break
        if subject_term:
            break

    if subject_term is None:
        return []

    links: List[Dict[str, Any]] = []
    for tm in term_matches:
        if tm[0] == subject_term:
            continue
        links.append({
            "term_a": subject_term,
            "term_a_in_glossary": True,
            "term_b": tm[0],
            "term_b_in_glossary": True,
            "type": "definitional",
            "direction": "b_used_to_define_a",
            "weight": 0.6,
            "quote": sentence.strip(),
            "source_file": source_file,
            "confidence": "medium",
        })
    return links


# ─── LLM synonym extraction ───────────────────────────────────────────────────

_LLM_SYN_SYSTEM = """\
You detect explicit synonym relationships in sentences from Hobbes's Leviathan (17th century English).
For each sentence, you are given the list of glossary terms found in it.
For each term, return the EXACT phrase from the sentence that the author explicitly equates as a synonym — or null if no explicit synonym exists for that term.
A synonym exists ONLY when the text explicitly equates two expressions as the same thing (e.g. "X, or Y", "X that is to say Y", "X also called Y", "X otherwise called Y").
Do NOT infer — only copy what is textually explicit.
Return ONLY valid JSON — no explanation, no markdown fences."""

_LLM_SYN_USER_TMPL = """\
For each numbered item below, return one JSON object with field "synonyms": an object mapping each listed term to the exact phrase explicitly given as its synonym in the sentence, or null if none.

{items}

Return a JSON array of {n} objects in the same order."""


def llm_extract_synonyms(
    candidates: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 15,
) -> List[Dict[str, Any]]:
    """Batch-send sentences to LLM to detect explicit synonym relationships.

    Each candidate includes its sentence AND the list of glossary terms found in
    it (from term_matches).  The LLM is anchored per-term: it returns, for each
    term, the exact phrase the sentence explicitly equates as a synonym (or null).

    This avoids substring qualifier-loss on the glossary-term side while keeping
    the prompt efficient (one sentence = one item, 76 API calls for 1143 sentences).
    """
    if not candidates:
        return []

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise SystemExit("openai package required for --llm-syn (pip install openai)") from exc

    client = OpenAI()
    all_links: List[Dict[str, Any]] = []
    n_batches = (len(candidates) + batch_size - 1) // batch_size
    seen_pairs: Set[Tuple[str, str]] = set()

    def _resolve_synonym(raw: str) -> Tuple[Optional[str], bool]:
        """Match a LLM-returned synonym phrase to the glossary (exact only)."""
        t = raw.strip().lower()
        if not t:
            return None, False
        if t in alias_map:
            return alias_map[t], True
        # Accept non-glossary phrase as-is (caller filters on both-in-glossary)
        return raw.strip(), False

    for b in range(n_batches):
        batch = candidates[b * batch_size: (b + 1) * batch_size]

        items_parts: List[str] = []
        for i, c in enumerate(batch):
            terms = list(dict.fromkeys(tm[0] for tm in c["term_matches"]))  # dedup, keep order
            terms_str = ", ".join(terms)
            items_parts.append(
                f'{i + 1}. Sentence: "{c["sentence"][:500]}"\n'
                f'   Terms: {terms_str}'
            )
        items_str = "\n\n".join(items_parts)
        prompt = _LLM_SYN_USER_TMPL.format(items=items_str, n=len(batch))

        try:
            resp = client.responses.create(
                model=model,
                instructions=_LLM_SYN_SYSTEM,
                input=prompt,
            )
            raw = resp.output_text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            decisions = json.loads(raw)
        except Exception as exc:
            print(f"  ⚠  LLM syn batch {b + 1}/{n_batches} failed: {exc}")
            continue

        if not isinstance(decisions, list) or len(decisions) != len(batch):
            print(f"  ⚠  LLM syn batch {b + 1}/{n_batches}: unexpected response shape")
            continue

        batch_links = 0
        for cand, decision in zip(batch, decisions):
            synonyms = decision.get("synonyms") or {}
            if not isinstance(synonyms, dict):
                continue

            for term_a_raw, syn_raw in synonyms.items():
                if not syn_raw or syn_raw == "null":
                    continue

                # term_a must be a known glossary canonical
                term_a = alias_map.get(term_a_raw.strip().lower())
                if not term_a:
                    continue

                term_b, b_in_gloss = _resolve_synonym(str(syn_raw))
                if not term_b:
                    continue
                if term_a.lower() == term_b.lower():
                    continue

                key = (min(term_a.lower(), term_b.lower()), max(term_a.lower(), term_b.lower()))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                all_links.append({
                    "term_a": term_a,
                    "term_a_in_glossary": True,
                    "term_b": term_b,
                    "term_b_in_glossary": b_in_gloss,
                    "type": "synonym",
                    "direction": "bidirectional",
                    "weight": 1.0,
                    "quote": cand["sentence"].strip(),
                    "source_file": cand["source_file"],
                    "confidence": "llm",
                })
                batch_links += 1

        print(f"  LLM syn batch {b + 1}/{n_batches}: {batch_links} pairs from {len(batch)} sentences")

    return all_links


# ─── LLM definitional extraction ──────────────────────────────────────────────

_LLM_SYSTEM = """\
You extract definitional relationships from sentences in Hobbes's Leviathan (17th century English).
For each sentence, a definitional marker was already detected (e.g., "is called", "signifieth", "we call", "consisteth in").
Your task: identify what concept is being defined/named and what label or term it receives.
Return ONLY valid JSON — no explanation, no markdown fences."""

_LLM_USER_TMPL = """\
For each numbered sentence below, return one JSON object with exactly two fields:
- "defined": the concept or thing being defined or named (copy the exact phrase from the sentence, or null if genuinely unclear)
- "as": the label, name or defining term assigned to it (copy the exact word or phrase from the sentence, or null if genuinely unclear)

Sentences:
{items}

Return a JSON array of {n} objects in the same order as the sentences."""


def llm_extract_definitional(
    candidates: List[Dict[str, Any]],
    alias_map: Dict[str, str],
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 15,
) -> List[Dict[str, Any]]:
    """Batch-send definitional candidate sentences to an OpenAI model.

    For each sentence the LLM returns {"defined": ..., "as": ...}.
    Both fields are matched back to the glossary alias_map; at least one side
    must be a known glossary term for the link to be emitted.
    """
    if not candidates:
        return []

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise SystemExit("openai package required for --llm-def (pip install openai)") from exc

    client = OpenAI()
    all_links: List[Dict[str, Any]] = []
    n_batches = (len(candidates) + batch_size - 1) // batch_size

    def _match_to_glossary(text: str) -> Tuple[Optional[str], bool]:
        """Try to resolve a raw LLM string to a canonical glossary term."""
        t = text.strip().lower()
        if not t:
            return None, False
        # Exact alias match
        if t in alias_map:
            return alias_map[t], True
        # Substring: longest alias contained in the text wins
        best: Optional[str] = None
        best_len = 0
        for surface, canonical in alias_map.items():
            if surface in t and len(surface) > best_len:
                best, best_len = canonical, len(surface)
        if best:
            return best, True
        # Not in glossary — return as-is (lowercased) so caller can decide
        return t, False

    for b in range(n_batches):
        batch = candidates[b * batch_size: (b + 1) * batch_size]
        items_str = "\n".join(
            f'{i + 1}. "{c["sentence"][:500]}"' for i, c in enumerate(batch)
        )
        prompt = _LLM_USER_TMPL.format(items=items_str, n=len(batch))

        try:
            resp = client.responses.create(
                model=model,
                instructions=_LLM_SYSTEM,
                input=prompt,
            )
            raw = resp.output_text.strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            decisions = json.loads(raw)
        except Exception as exc:
            print(f"  ⚠  LLM batch {b + 1}/{n_batches} failed: {exc}")
            continue

        if not isinstance(decisions, list) or len(decisions) != len(batch):
            print(f"  ⚠  LLM batch {b + 1}/{n_batches}: unexpected response (got {len(decisions) if isinstance(decisions, list) else type(decisions).__name__}, expected {len(batch)})")
            continue

        batch_links = 0
        for cand, decision in zip(batch, decisions):
            defined_raw = str(decision.get("defined") or "").strip()
            as_raw      = str(decision.get("as")      or "").strip()
            if not defined_raw or not as_raw or defined_raw == "null" or as_raw == "null":
                continue

            defined_term, defined_in_gloss = _match_to_glossary(defined_raw)
            as_term,      as_in_gloss      = _match_to_glossary(as_raw)

            if not defined_term or not as_term:
                continue
            if defined_term == as_term:
                continue
            if not defined_in_gloss and not as_in_gloss:
                continue

            all_links.append({
                "term_a": defined_term,
                "term_a_in_glossary": defined_in_gloss,
                "term_b": as_term,
                "term_b_in_glossary": as_in_gloss,
                "type": "definitional",
                "direction": "b_used_to_define_a",
                "weight": 0.8,
                "quote": cand["sentence"].strip(),
                "source_file": cand["source_file"],
                "confidence": "llm",
            })
            batch_links += 1

        print(f"  LLM batch {b + 1}/{n_batches}: {batch_links} links from {len(batch)} sentences")

    return all_links


# ─── Per-file processing ───────────────────────────────────────────────────────

def process_file(
    path: Path,
    patterns: List[Tuple[re.Pattern, str]],
    quick_filter: Set[str],
    nlp: Optional[Any],
    include_low: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process one corpus file. Returns (links, stats_dict).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    sentences = split_sentences(text)
    source_file = path.name

    links: List[Dict[str, Any]] = []
    stats = {
        "sentences_total": len(sentences),
        "sentences_with_terms": 0,
        "def_sentences": 0,
    }

    for sent in sentences:
        # Quick filter: skip sentences with no glossary-word tokens
        sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
        if not (sent_words & quick_filter):
            continue

        term_matches = find_terms_in_sentence(sent, patterns)
        if not term_matches:
            continue

        stats["sentences_with_terms"] += 1

        # 1. Synonym detection
        syn_links = find_synonym_links(sent, term_matches, source_file)
        links.extend(syn_links)

        # 2. Definitional detection — lexical
        def_links = find_definitional_links(sent, term_matches, source_file, include_low=include_low)
        if def_links:
            stats["def_sentences"] += 1
            links.extend(def_links)
        elif nlp is not None and len(term_matches) >= 2:
            # 3. Definitional detection — spaCy (fallback)
            spacy_links = find_definitional_links_spacy(sent, term_matches, source_file, nlp)
            if spacy_links:
                stats["def_sentences"] += 1
                links.extend(spacy_links)

    return links, stats


# ─── LLM mode: candidate collection ──────────────────────────────────────────

def collect_def_candidates_from_file(
    path: Path,
    patterns: List[Tuple[re.Pattern, str]],
    quick_filter: Set[str],
    include_low: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Like process_file but for --llm-def mode.

    Synonym links are computed normally (regex works well for them).
    Definitional sentences are collected as candidates instead of immediately
    generating links — they'll be batch-sent to the LLM afterward.

    Returns: (synonym_links, def_candidates, stats)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    sentences = split_sentences(text)
    source_file = path.name

    synonym_links: List[Dict[str, Any]] = []
    def_candidates: List[Dict[str, Any]] = []
    stats = {
        "sentences_total": len(sentences),
        "sentences_with_terms": 0,
        "def_sentences": 0,
    }

    for sent in sentences:
        sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
        if not (sent_words & quick_filter):
            continue

        term_matches = find_terms_in_sentence(sent, patterns)
        if not term_matches:
            continue

        stats["sentences_with_terms"] += 1

        # Synonyms: same as always
        synonym_links.extend(find_synonym_links(sent, term_matches, source_file))

        # Definitional: collect sentence if a marker fires
        marker_pos, _ = _def_marker_positions(sent, include_low=include_low)
        if marker_pos is not None:
            stats["def_sentences"] += 1
            def_candidates.append({
                "sentence": sent,
                "term_matches": term_matches,
                "source_file": source_file,
            })

    return synonym_links, def_candidates, stats


def collect_syn_candidates_from_file(
    path: Path,
    patterns: List[Tuple[re.Pattern, str]],
    quick_filter: Set[str],
    include_low: bool = False,
    also_def: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Collect candidates for LLM synonym (and optionally definitional) extraction.

    Every sentence that contains at least one glossary term becomes a synonym
    candidate (the LLM will decide if an explicit synonym relationship exists).
    Definitional candidates (marker-filtered) are only collected when also_def=True.

    Returns: (syn_candidates, def_candidates, stats)
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    sentences = split_sentences(text)
    source_file = path.name

    syn_candidates: List[Dict[str, Any]] = []
    def_candidates: List[Dict[str, Any]] = []
    stats = {
        "sentences_total": len(sentences),
        "sentences_with_terms": 0,
        "def_sentences": 0,
    }

    for sent in sentences:
        sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
        if not (sent_words & quick_filter):
            continue

        term_matches = find_terms_in_sentence(sent, patterns)
        if not term_matches:
            continue

        stats["sentences_with_terms"] += 1
        candidate = {"sentence": sent, "term_matches": term_matches, "source_file": source_file}
        syn_candidates.append(candidate)

        if also_def:
            marker_pos, _ = _def_marker_positions(sent, include_low=include_low)
            if marker_pos is not None:
                stats["def_sentences"] += 1
                def_candidates.append(candidate)

    return syn_candidates, def_candidates, stats


# ─── LLM qualify: recall helpers ─────────────────────────────────────────────

def _collect_synonym_candidates_for_qualify(
    txt_files: List[Path],
    patterns: List[Tuple[re.Pattern, str]],
    quick_filter: Set[str],
) -> List[Dict[str, Any]]:
    """Recall pass 1: pairs connected by a SYNONYM_CONNECTOR.

    term_b may be a non-glossary raw word (potential satellite).
    """
    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for sent in split_sentences(text):
            if not (set(re.findall(r"\b\w+\b", sent.lower())) & quick_filter):
                continue
            term_matches = find_terms_in_sentence(sent, patterns)
            for conn_re in _CONN_RES:
                for m in conn_re.finditer(sent):
                    left, lin, *_ = _left_term(sent, m.start(), term_matches)
                    right, rin, *_ = _right_term(sent, m.end(), term_matches)
                    if not left or not right or left.lower() == right.lower():
                        continue
                    if not lin and not rin:
                        continue
                    key = (min(left.lower(), right.lower()), max(left.lower(), right.lower()), path.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "sentence": sent,
                        "term_a": left,
                        "term_b": right,
                        "source_file": path.name,
                    })
    return candidates


def _collect_marker_candidates_for_qualify(
    txt_files: List[Path],
    patterns: List[Tuple[re.Pattern, str]],
    quick_filter: Set[str],
    marker_res: List[re.Pattern],
) -> List[Dict[str, Any]]:
    """Recall passes 2-5: enumerate all canonical pairs in marker-positive sentences."""
    candidates: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for sent in split_sentences(text):
            sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
            if not (sent_words & quick_filter):
                continue
            if not any(mr.search(sent) for mr in marker_res):
                continue
            term_matches = find_terms_in_sentence(sent, patterns)
            if len(term_matches) < 2:
                continue
            glossary_terms = [tm[0] for tm in term_matches]
            for i in range(len(glossary_terms)):
                for j in range(i + 1, len(glossary_terms)):
                    ta, tb = glossary_terms[i], glossary_terms[j]
                    if ta.lower() == tb.lower():
                        continue
                    key = (min(ta.lower(), tb.lower()), max(ta.lower(), tb.lower()), path.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "sentence": sent,
                        "term_a": ta,
                        "term_b": tb,
                        "source_file": path.name,
                    })
    return candidates


# ─── LLM qualify: prompts + pass function ─────────────────────────────────────

_QUALIFY_SYSTEMS: Dict[str, str] = {
    "synonym": (
        "You analyze sentences from a 17th-century philosophy book (Hobbes, Leviathan). "
        "For each item, determine if the two terms are EXPLICITLY stated to mean the same "
        "thing in that sentence. No inference — only confirm what is textually explicit. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    ),
    "antonym": (
        "You analyze sentences from a 17th-century philosophy book (Hobbes, Leviathan). "
        "For each item, determine if the two terms are EXPLICITLY contrasted or opposed "
        "in that sentence. No inference — only confirm what is textually explicit. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    ),
    "definitional": (
        "You analyze sentences from a 17th-century philosophy book (Hobbes, Leviathan). "
        "For each item, determine if one term explicitly defines or names the other in "
        "that sentence. No inference — only confirm what is textually explicit. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    ),
    "argumentative_neighbor": (
        "You analyze sentences from a 17th-century philosophy book (Hobbes, Leviathan). "
        "For each item, determine if the sentence establishes an explicit argumentative or "
        "logical dependency between the two terms (one requires, entails, presupposes, or "
        "follows from the other). No inference — only confirm what is textually explicit. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    ),
    "related": (
        "You analyze sentences from a 17th-century philosophy book (Hobbes, Leviathan). "
        "For each item, determine if the sentence establishes an explicit conceptual "
        "connection between the two terms that does not fit synonym/antonym/definitional/"
        "argumentative categories. No inference — only confirm what is textually explicit. "
        "Return ONLY valid JSON — no explanation, no markdown fences."
    ),
}

_QUALIFY_TEMPLATES: Dict[str, str] = {
    "synonym": (
        "For each numbered item (sentence + Term A + Term B), determine if the terms are "
        "explicitly stated as synonymous.\n"
        'Return a JSON array of {n} objects: {{"confirmed": true/false, '
        '"direction": "bidirectional"|"a_to_b"|"b_to_a", '
        '"strength": "high"|"medium"|"low", "evidence": "verbatim quote or null"}}\n\n'
        "{items}\n\nReturn a JSON array of {n} objects in the same order."
    ),
    "antonym": (
        "For each numbered item (sentence + Term A + Term B), determine if the terms are "
        "explicitly contrasted or opposed.\n"
        'Return a JSON array of {n} objects: {{"confirmed": true/false, '
        '"strength": "high"|"medium"|"low", "evidence": "verbatim quote or null"}}\n\n'
        "{items}\n\nReturn a JSON array of {n} objects in the same order."
    ),
    "definitional": (
        "For each numbered item (sentence + Term A + Term B), determine if one term "
        "explicitly defines or names the other.\n"
        'Return a JSON array of {n} objects: {{"confirmed": true/false, '
        '"direction": "a_defines_b"|"b_defines_a"|"bidirectional", '
        '"strength": "high"|"medium"|"low", "evidence": "verbatim quote or null"}}\n\n'
        "{items}\n\nReturn a JSON array of {n} objects in the same order."
    ),
    "argumentative_neighbor": (
        "For each numbered item (sentence + Term A + Term B), determine if there is an "
        "explicit logical or argumentative dependency between the terms.\n"
        'Return a JSON array of {n} objects: {{"confirmed": true/false, '
        '"direction": "a_requires_b"|"b_requires_a"|"bidirectional", '
        '"strength": "high"|"medium"|"low", "evidence": "verbatim quote or null"}}\n\n'
        "{items}\n\nReturn a JSON array of {n} objects in the same order."
    ),
    "related": (
        "For each numbered item (sentence + Term A + Term B), determine if the terms are "
        "explicitly connected in a way that is not synonym/antonym/definitional/argumentative.\n"
        'Return a JSON array of {n} objects: {{"confirmed": true/false, '
        '"strength": "high"|"medium"|"low", "evidence": "verbatim quote or null"}}\n\n'
        "{items}\n\nReturn a JSON array of {n} objects in the same order."
    ),
}


def _llm_qualify_pass(
    candidates: List[Dict[str, Any]],
    relation_type: str,
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 20,
) -> List[Dict[str, Any]]:
    """Run one LLM qualify pass for a specific relation type.

    Returns confirmed edges: [{terms, relation_type, direction, strength, evidence_quote, source_file}]
    """
    if not candidates:
        return []

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise SystemExit("openai package required for --llm-qualify (pip install openai)") from exc

    client = OpenAI()
    system = _QUALIFY_SYSTEMS[relation_type]
    tmpl = _QUALIFY_TEMPLATES[relation_type]
    all_edges: List[Dict[str, Any]] = []
    n_batches = (len(candidates) + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch = candidates[b * batch_size: (b + 1) * batch_size]
        items_parts = [
            f'{i + 1}. Sentence: "{c["sentence"][:500]}"\n'
            f'   Term A: "{c["term_a"]}"  |  Term B: "{c["term_b"]}"'
            for i, c in enumerate(batch)
        ]
        prompt = tmpl.format(items="\n\n".join(items_parts), n=len(batch))

        try:
            resp = client.responses.create(model=model, instructions=system, input=prompt)
            raw = resp.output_text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            decisions = json.loads(raw)
        except Exception as exc:
            print(f"  ⚠  qualify [{relation_type}] batch {b + 1}/{n_batches} failed: {exc}")
            continue

        if not isinstance(decisions, list) or len(decisions) != len(batch):
            print(f"  ⚠  qualify [{relation_type}] batch {b + 1}/{n_batches}: unexpected shape")
            continue

        confirmed = 0
        for cand, decision in zip(batch, decisions):
            if not decision.get("confirmed"):
                continue
            all_edges.append({
                "terms": [cand["term_a"], cand["term_b"]],
                "relation_type": relation_type,
                "direction": decision.get("direction", "bidirectional"),
                "strength": decision.get("strength", "medium"),
                "evidence_quote": decision.get("evidence") or cand["sentence"].strip(),
                "source_file": cand["source_file"],
            })
            confirmed += 1

        print(f"  qualify [{relation_type}] batch {b + 1}/{n_batches}: {confirmed}/{len(batch)} confirmed")

    return all_edges


# ─── Satellite term extraction ─────────────────────────────────────────────────

def _split_by_canonicity(
    edges: List[Dict[str, Any]],
    glossary_set: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Classify edge terms as canonical or satellite.

    Convention: terms[0] is always canonical (in glossary_set).
    Directional fields are flipped when terms are swapped.

    Returns (enriched_edges, satellite_list).
    satellite_list: [{"id", "term", "linked_canonical_term"}]
    """
    _FLIP_DIR: Dict[str, str] = {
        "a_to_b": "b_to_a",
        "b_to_a": "a_to_b",
        "a_defines_b": "b_defines_a",
        "b_defines_a": "a_defines_b",
        "a_requires_b": "b_requires_a",
        "b_requires_a": "a_requires_b",
    }

    enriched: List[Dict[str, Any]] = []
    satellites: Dict[str, Dict[str, Any]] = {}  # term_lower → entry
    sat_counter = [0]

    # Deduplicate: keep highest strength per (ta, tb, relation_type)
    _STRENGTH_ORDER = {"high": 3, "medium": 2, "low": 1}
    best_edge: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def _dedup_key(ta: str, tb: str, rel: str) -> Tuple[str, str, str]:
        return (min(ta.lower(), tb.lower()), max(ta.lower(), tb.lower()), rel)

    for edge in edges:
        terms = edge.get("terms", [])
        if len(terms) != 2:
            continue
        t0, t1 = terms[0], terms[1]
        t0_in = t0.lower() in glossary_set
        t1_in = t1.lower() in glossary_set

        if not t0_in and not t1_in:
            continue  # neither canonical — skip

        direction = edge.get("direction", "bidirectional")

        # Ensure terms[0] is always the canonical term
        if t1_in and not t0_in:
            t0, t1 = t1, t0
            t0_in, t1_in = True, False
            direction = _FLIP_DIR.get(direction, direction)

        target_type = "canonical" if t1_in else "satellite"

        if target_type == "satellite":
            tl = t1.lower()
            if tl not in satellites:
                sat_counter[0] += 1
                satellites[tl] = {
                    "id": f"ST{sat_counter[0]:04d}",
                    "term": t1,
                    "linked_canonical_term": t0,
                }

        rel = edge.get("relation_type", "related")
        dk = _dedup_key(t0, t1, rel)
        new_strength = _STRENGTH_ORDER.get(edge.get("strength", "medium"), 2)
        if dk in best_edge:
            existing_strength = _STRENGTH_ORDER.get(best_edge[dk].get("strength", "medium"), 2)
            if new_strength <= existing_strength:
                continue  # already have a better or equal edge

        best_edge[dk] = {
            "terms": [t0, t1],
            "source_type": "canonical",
            "target_type": target_type,
            "relation_type": rel,
            "direction": direction,
            "strength": edge.get("strength", "medium"),
            "evidence_quote": edge.get("evidence_quote", ""),
            "source_file": edge.get("source_file", ""),
        }

    return list(best_edge.values()), list(satellites.values())


# ─── Audit report ─────────────────────────────────────────────────────────────

def print_report(
    all_links: List[Dict[str, Any]],
    total_stats: Dict[str, int],
    n_files: int,
) -> None:
    import random

    def _rel(e: Dict[str, Any]) -> str:
        return e.get("relation_type") or e.get("type") or "unknown"

    by_rel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for lnk in all_links:
        by_rel[_rel(lnk)].append(lnk)

    print("\n" + "─" * 60)
    print("CONCEPT GRAPH — AUDIT REPORT")
    print("─" * 60)
    print(f"Files processed:               {n_files}")
    print(f"Total sentences:               {total_stats.get('sentences_total', 0)}")
    print(f"Sentences with glossary terms: {total_stats.get('sentences_with_terms', 0)}")
    print()
    print("Edges created:")
    for rel, edges in sorted(by_rel.items()):
        print(f"  {rel:<25} {len(edges)}")
    print(f"  {'Total':<25} {len(all_links)}")

    # Top 10 most connected terms
    degree: Dict[str, int] = defaultdict(int)
    for lnk in all_links:
        terms = lnk.get("terms") or [lnk.get("term_a", ""), lnk.get("term_b", "")]
        for t in terms:
            if t:
                degree[t] += 1
    top10 = sorted(degree.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 most connected terms:")
    for term, cnt in top10:
        print(f"  {term:<30} {cnt} edges")

    # Sample edges per type
    random.seed(42)
    for label, subset in by_rel.items():
        sample = random.sample(subset, min(3, len(subset)))
        print(f"\nSample {label} edges:")
        for lnk in sample:
            terms = lnk.get("terms") or [lnk.get("term_a", "?"), lnk.get("term_b", "?")]
            conf = lnk.get("strength") or lnk.get("confidence") or "?"
            quote = lnk.get("evidence_quote") or lnk.get("quote") or ""
            print(f"  [{conf}] {terms[0]} ↔ {terms[1] if len(terms) > 1 else '?'}")
            if quote:
                print(f"    {str(quote)[:120]}")
            print(f"    ← {lnk.get('source_file', '?')}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a concept graph of term links from a corpus."
    )
    p.add_argument(
        "--corpus-dir", required=True,
        help="Directory containing .txt corpus files.",
    )
    p.add_argument(
        "--glossary", required=True,
        help="JSON file with glossary records (llm_label==KEEP will be used).",
    )
    p.add_argument(
        "--output", required=True,
        help="Path for output concept_graph.json.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Process only the first 3 files; do not write output.",
    )
    p.add_argument(
        "--no-spacy", action="store_true",
        help="Disable spaCy grammatical detection (faster).",
    )
    p.add_argument(
        "--min-confidence", choices=["high", "medium", "low"], default="medium",
        help=(
            "Minimum confidence level to include in output. "
            "'high' = only specific lexical markers; "
            "'medium' = also include spaCy grammatical detections (default); "
            "'low' = also include generic 'is a/the' markers (noisy)."
        ),
    )
    p.add_argument(
        "--llm-syn", action="store_true",
        help=(
            "Use an LLM to detect synonym links instead of the regex connector patterns. "
            "Every sentence containing a glossary term is sent; the LLM decides if an "
            "explicit synonym relationship is established. Requires OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--llm-def", action="store_true",
        help=(
            "Use an LLM to extract definitional links instead of the regex heuristic. "
            "When used alone, synonym links are still extracted with regex. "
            "Requires OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--llm-model", default="gpt-4.1-mini",
        help="OpenAI model to use for LLM modes (default: gpt-4.1-mini).",
    )
    p.add_argument(
        "--llm-batch-size", type=int, default=15,
        help="Number of sentences per LLM call (default: 15).",
    )
    p.add_argument(
        "--llm-qualify", action="store_true",
        help=(
            "5-pass unified pipeline: regex recall per relation family → LLM qualify. "
            "Covers synonym, antonym, definitional, argumentative_neighbor, related. "
            "Also writes --satellite-output. Requires OPENAI_API_KEY."
        ),
    )
    p.add_argument(
        "--satellite-output",
        help=(
            "Path for satellite_terms.json (auto-discovered non-canonical terms). "
            "Only used with --llm-qualify. Defaults to same dir as --output."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    corpus_dir = Path(args.corpus_dir)
    glossary_path = Path(args.glossary)
    output_path = Path(args.output)

    if not corpus_dir.is_dir():
        sys.exit(f"Error: corpus directory not found: {corpus_dir}")
    if not glossary_path.is_file():
        sys.exit(f"Error: glossary file not found: {glossary_path}")

    # ── Load glossary ──────────────────────────────────────────────────────────
    print(f"Loading glossary: {glossary_path.name}")
    terms, alias_map = load_glossary(glossary_path)
    print(f"  {len(terms)} KEEP terms, {len(alias_map)} surface forms (incl. aliases)")

    patterns     = build_term_patterns(alias_map)
    quick_filter = build_quick_filter(alias_map)

    # ── Load spaCy (optional) ──────────────────────────────────────────────────
    nlp = None
    if not args.no_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            print("spaCy en_core_web_sm loaded.")
        except Exception as exc:
            print(f"spaCy unavailable ({exc}) — using lexical detection only.")

    # ── Collect corpus files ───────────────────────────────────────────────────
    txt_files = sorted(corpus_dir.glob("*.txt"))
    if args.dry_run:
        txt_files = txt_files[:3]
        print(f"DRY RUN — processing {len(txt_files)} files only.")
    else:
        print(f"Processing {len(txt_files)} corpus files…")

    # ── Process ───────────────────────────────────────────────────────────────
    include_low = (args.min_confidence == "low")
    _CONF_ORDER = {"high": 3, "medium": 2, "low": 1, "llm": 3}
    min_conf_val = _CONF_ORDER[args.min_confidence]

    all_links: List[Dict[str, Any]] = []
    total_stats: Dict[str, int] = defaultdict(int)

    if args.llm_qualify:
        # ── 5-pass recall + LLM qualify ────────────────────────────────────
        glossary_set: Set[str] = {t.lower() for t in terms}
        for alias_lower in alias_map:
            glossary_set.add(alias_lower)

        _PASSES = [
            ("synonym",               _collect_synonym_candidates_for_qualify,   None),
            ("antonym",               _collect_marker_candidates_for_qualify,     _ANT_RES),
            ("definitional",          _collect_marker_candidates_for_qualify,     _DEF_HIGH + _DEF_LOW),
            ("argumentative_neighbor",_collect_marker_candidates_for_qualify,     _ARG_RES),
            ("related",               _collect_marker_candidates_for_qualify,     _RELATED_RES),
        ]

        raw_edges: List[Dict[str, Any]] = []
        for relation_type, collect_fn, marker_res in _PASSES:
            if marker_res is not None:
                candidates = collect_fn(txt_files, patterns, quick_filter, marker_res)
            else:
                candidates = collect_fn(txt_files, patterns, quick_filter)
            print(f"\n[qualify] {relation_type}: {len(candidates)} recall candidates")
            edges_pass = _llm_qualify_pass(
                candidates, relation_type,
                model=args.llm_model,
                batch_size=args.llm_batch_size,
            )
            print(f"  → {len(edges_pass)} confirmed edges")
            raw_edges.extend(edges_pass)
            for k, v in {"sentences_total": 0, "sentences_with_terms": 0, "def_sentences": 0}.items():
                total_stats[k] += 0  # stats not tracked in qualify mode

        print(f"\nTotal raw edges across all passes: {len(raw_edges)}")
        enriched_edges, satellite_list = _split_by_canonicity(raw_edges, glossary_set)
        all_links.extend(enriched_edges)
        print(f"After dedup/canonicity split: {len(enriched_edges)} edges, {len(satellite_list)} satellite terms")

        # Write satellite_terms.json
        if not args.dry_run:
            sat_path = (
                Path(args.satellite_output)
                if args.satellite_output
                else output_path.parent / "satellite_terms.json"
            )
            sat_path.parent.mkdir(parents=True, exist_ok=True)
            sat_path.write_text(
                json.dumps(satellite_list, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Satellite terms written → {sat_path}  ({len(satellite_list)} terms)")

    elif args.llm_syn:
        # ── LLM synonym mode ───────────────────────────────────────────────
        mode_label = "--llm-syn" + (" + --llm-def" if args.llm_def else "")
        print(f"Mode {mode_label}: collecting candidates…")
        all_syn_candidates: List[Dict[str, Any]] = []
        all_def_candidates: List[Dict[str, Any]] = []

        for i, path in enumerate(txt_files, 1):
            syn_cands, def_cands, stats = collect_syn_candidates_from_file(
                path, patterns, quick_filter,
                include_low=include_low,
                also_def=args.llm_def,
            )
            all_syn_candidates.extend(syn_cands)
            all_def_candidates.extend(def_cands)
            for k, v in stats.items():
                total_stats[k] += v
            if i % 20 == 0 or i == len(txt_files):
                print(f"  [{i}/{len(txt_files)}] syn candidates: {len(all_syn_candidates)}  def candidates: {len(all_def_candidates)}")

        n_batches_syn = (len(all_syn_candidates) + args.llm_batch_size - 1) // args.llm_batch_size
        print(f"\nSending {len(all_syn_candidates)} sentences to {args.llm_model} for synonym extraction ({n_batches_syn} batches)…")
        syn_links = llm_extract_synonyms(
            all_syn_candidates,
            alias_map,
            model=args.llm_model,
            batch_size=args.llm_batch_size,
        )
        all_links.extend(syn_links)
        print(f"LLM synonym extraction complete: {len(syn_links)} synonym links")

        if args.llm_def:
            n_batches_def = (len(all_def_candidates) + args.llm_batch_size - 1) // args.llm_batch_size
            print(f"\nSending {len(all_def_candidates)} sentences to {args.llm_model} for definitional extraction ({n_batches_def} batches)…")
            def_links = llm_extract_definitional(
                all_def_candidates,
                alias_map,
                model=args.llm_model,
                batch_size=args.llm_batch_size,
            )
            all_links.extend(def_links)
            print(f"LLM definitional extraction complete: {len(def_links)} definitional links")

    elif args.llm_def:
        # ── LLM definitional only: regex synonyms + LLM definitional ──────
        print("Mode --llm-def: collecting definitional candidates…")
        all_def_candidates_only: List[Dict[str, Any]] = []

        for i, path in enumerate(txt_files, 1):
            syn_links, def_cands, stats = collect_def_candidates_from_file(
                path, patterns, quick_filter, include_low=include_low
            )
            all_links.extend(syn_links)
            all_def_candidates_only.extend(def_cands)
            for k, v in stats.items():
                total_stats[k] += v
            if i % 20 == 0 or i == len(txt_files):
                print(f"  [{i}/{len(txt_files)}] synonym links: {len(all_links)}  def candidates: {len(all_def_candidates_only)}")

        n_batches_def = (len(all_def_candidates_only) + args.llm_batch_size - 1) // args.llm_batch_size
        print(f"\nSending {len(all_def_candidates_only)} candidate sentences to {args.llm_model} ({n_batches_def} batches)…")
        def_links = llm_extract_definitional(
            all_def_candidates_only,
            alias_map,
            model=args.llm_model,
            batch_size=args.llm_batch_size,
        )
        all_links.extend(def_links)
        print(f"LLM extraction complete: {len(def_links)} definitional links")

    else:
        # ── Regex mode (default) ───────────────────────────────────────────
        for i, path in enumerate(txt_files, 1):
            links, stats = process_file(path, patterns, quick_filter, nlp, include_low=include_low)
            links = [l for l in links if _CONF_ORDER.get(l.get("confidence", "low"), 0) >= min_conf_val]
            all_links.extend(links)
            for k, v in stats.items():
                total_stats[k] += v
            if i % 20 == 0 or i == len(txt_files):
                print(f"  [{i}/{len(txt_files)}] links so far: {len(all_links)}")

    # ── Audit report ──────────────────────────────────────────────────────────
    print_report(all_links, total_stats, len(txt_files))

    # ── Write output ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDry run — output NOT written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = {
        "metadata": {
            "corpus": corpus_dir.name,
            "glossary": glossary_path.name,
            "total_terms": len(terms),
            "total_edges": len(all_links),
            "by_relation": {
                rt: sum(1 for e in all_links if e.get("relation_type") == rt or e.get("type") == rt)
                for rt in ("synonym", "antonym", "definitional", "argumentative_neighbor", "related")
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "edges": all_links,
    }
    output_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGraph written → {output_path}  ({output_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
