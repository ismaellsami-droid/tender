#!/usr/bin/env python3
# Archived: legacy graph-generation workflow.
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
CHAPTER_RE = re.compile(r"_ch_(\d{1,3})(?:_|$)", re.IGNORECASE)
# Conservative sentence-break detector: period/!/? followed by whitespace
# and an uppercase letter, or a blank line (paragraph break).
SENTENCE_BREAK_RE = re.compile(r"(?:[.!?]\s+(?=[A-Z])|\n{2,})")
# Token window for definitional proximity (same sentence required in addition).
DEFINITIONAL_WINDOW = 20

# Small built-in stopword list (kept deliberately compact and conservative).
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves",
}

LENGTH_WEIGHT = {1: 0.0, 2: 0.5, 3: 1.0}
SHORT_UNIGRAM_SHADOW_MARGIN = 0.10
SHORT_UNIGRAM_SHADOW_MIN_COVERAGE = 0.60

# Default scoring weights.  Must sum to > 0; finalize_records normalises them
# automatically so you can pass any positive combination via CLI.
DEFAULT_SCORE_W_FREQ: float = 0.35   # log(1 + freq_total)
DEFAULT_SCORE_W_IDF:  float = 0.25   # idf_like × (1 + 0.25 × specificity)
DEFAULT_SCORE_W_DEF:  float = 0.25   # log(1 + definitional_hits)
DEFAULT_SCORE_W_LEN:  float = 0.15   # length_weight (0/0.5/1 for 1/2/3-grams)

# Hobbes-era archaic inflections → canonical modern forms.
# Applied at tokenization time so all downstream logic (scoring, markers,
# STOPWORDS filtering) works on consistent canonical tokens.
ARCHAIC_FORMS: Dict[str, str] = {
    # Third-person singular present ("-eth" suffix)
    "signifieth": "signify",
    "consisteth": "consist",
    "maketh": "make",
    "taketh": "take",
    "giveth": "give",
    "followeth": "follow",
    "calleth": "call",
    "speaketh": "speak",
    "knoweth": "know",
    "belongeth": "belong",
    "proceedeth": "proceed",
    "requireth": "require",
    "concerneth": "concern",
    "dependeth": "depend",
    "produceth": "produce",
    "seeth": "see",
    "understandeth": "understand",
    "holdeth": "hold",
    "desireth": "desire",
    "appeareth": "appear",
    "happeneth": "happen",
    # Auxiliary verb archaisms
    "hath": "have",
    "doth": "do",
    "hast": "have",
    "dost": "do",
    "wilt": "will",
    "wouldst": "would",
    "couldst": "could",
    "shouldst": "should",
    "shalt": "shall",
    "canst": "can",
    # Archaic pronouns / adverbs → modern equivalents already in STOPWORDS
    "onely": "only",
    "wee": "we",
    "himselfe": "himself",
    "herselfe": "herself",
    "themselues": "themselves",
    "whosoeuer": "whoever",
    "whatsoeuer": "whatever",
    "wheresoever": "wherever",
}

# Markers are defined in terms of canonical (post-ARCHAIC_FORMS) tokens.
# signifieth/signifies → signify, consisteth/consists → consist.
MARKERS: Tuple[Tuple[str, ...], ...] = (
    ("is", "called"),
    ("are", "called"),
    ("is", "that"),
    ("signify",),
    ("consist",),
    ("call",),          # lemma of "called/calleth" — the definitional verb itself
    ("by", "this", "i", "mean"),
    ("that", "is", "to", "say"),
)

MARKER_TOKENS = {tok for marker in MARKERS for tok in marker}
MARKER_PHRASES = {" ".join(m) for m in MARKERS}
TERMS_OUTPUT_DIR_RE = re.compile(r"^terms_extracted_v(\d+)$")
LLM_LABELS: Tuple[str, ...] = (
    "KEEP",
    "REJECT_ORTHOGRAPHY_VARIANT",
    "REJECT_OTHER",
    "UNSURE",
)
REJECT_LABELS = {lbl for lbl in LLM_LABELS if lbl.startswith("REJECT_")}
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_MATRIX_EMBED_MODEL = "text-embedding-3-small"
MERGEABLE_REJECT_LABELS = {
    "REJECT_ORTHOGRAPHY_VARIANT",
}

# ── ARCHIVED LISTS (pre-spaCy POS filter) ────────────────────────────────────
# Kept for reference; the POS filter now handles most of these automatically.
# GENERIC_NOISE_TERMS_V0 = {
#     "thing", "things", "one", "many", "every", "first", "second", "third",
#     "without", "much", "others", "another", "make", "made", "well", "yet",
#     "therefore",
# }
# STOPWORDS_V0 — standard English set above (unchanged).
# ─────────────────────────────────────────────────────────────────────────────

# Generic terms unhelpful for retrieval.  The POS filter catches most verbs /
# adverbs / conjunctions automatically; this list covers:
#   • archaic forms spaCy can't tag reliably (selfe, thereof, whereof, …)
#   • edge cases where POS is ambiguous but the term is clearly not a concept
GENERIC_NOISE_TERMS = {
    # original entries
    "thing", "things", "one", "many", "every", "first", "second", "third",
    "without", "much", "others", "another", "make", "made", "well", "yet",
    "therefore",
    # archaic relational adverbs / reflexive — spaCy unreliable on these
    "selfe", "thereof", "whereof", "wherein", "therein", "hereof",
    "thereupon", "whereupon", "hereafter", "hereby", "thereby",
}


@dataclass
class Example:
    filename: str
    snippet: str
    definitional: bool = False


@dataclass
class TermStats:
    n: int
    freq_total: int = 0
    doc_freq: int = 0
    definitional_hits: int = 0
    chapters: set[int] = field(default_factory=set)
    examples: List[Example] = field(default_factory=list)
    idf_like: float = 0.0
    specificity: float = 0.0
    length_weight: float = 0.0
    score: float = 0.0


@dataclass
class LLMDecision:
    term: str
    label: str
    short_rationale: str
    evidence: List[str]
    canonical_term: str = ""


def _term_quality_key(rec: Dict[str, object]) -> Tuple[int, int, float, float, int, int]:
    """Deterministic ordering for canonical representative selection."""
    term = str(rec.get("term", ""))
    tokens = term.split()
    edge_bad = 0
    if tokens and (tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS):
        edge_bad = 1
    if tokens and (tokens[0] in MARKER_TOKENS or tokens[-1] in MARKER_TOKENS):
        edge_bad = 1
    return (
        -edge_bad,
        int(rec.get("definitional_hits", 0)),
        float(rec.get("idf_like", 0.0)),
        float(rec.get("score", 0.0)),
        -int(rec.get("n", 1)),
        -len(term),
    )


def parse_chapter(filename: str) -> Optional[int]:
    m = CHAPTER_RE.search(filename)
    if not m:
        return None
    return int(m.group(1))


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize text into (canonical_tokens, char_spans).

    Archaic forms (signifieth, hath, …) are normalized to their modern
    equivalents so that all downstream logic operates on consistent tokens.
    Spans always reference the original character positions for snippet
    extraction — they are unaffected by token normalization.
    """
    lower = text.lower()
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in TOKEN_RE.finditer(lower):
        tok = ARCHAIC_FORMS.get(m.group(0), m.group(0))
        tokens.append(tok)
        spans.append(m.span())
    return tokens, spans


def marker_centers(tokens: Sequence[str]) -> List[int]:
    out: List[int] = []
    n = len(tokens)
    for i in range(n):
        for marker in MARKERS:
            ln = len(marker)
            if i + ln <= n and tuple(tokens[i : i + ln]) == marker:
                out.append(i + (ln // 2))
        # "by ... i mean" wildcard variant.
        if tokens[i] == "by":
            hi = min(n - 1, i + 6)
            for j in range(i + 1, hi + 1):
                if tokens[j] == "i" and (j + 1 < n and tokens[j + 1] == "mean"):
                    out.append((i + j + 1) // 2)
                    break
    return out


def is_valid_candidate(ngram: Sequence[str]) -> bool:
    if not ngram:
        return False
    # Keep 1-2 character unigrams out explicitly; they are almost always function words/noise.
    if len(ngram) == 1 and len(ngram[0]) <= 2:
        return False
    if all(tok in STOPWORDS for tok in ngram):
        return False
    if ngram[0] in STOPWORDS or ngram[-1] in STOPWORDS:
        return False
    # Keep deterministic non-lexical filtering only.
    for tok in ngram:
        if any(ch.isdigit() for ch in tok):
            return False
    return True


def make_snippet(text: str, start_char: int, end_char: int, max_len: int = 160) -> str:
    left = max(0, start_char - 70)
    right = min(len(text), end_char + 70)
    chunk = re.sub(r"\s+", " ", text[left:right]).strip()
    if len(chunk) <= max_len:
        return chunk
    return (chunk[: max_len - 1] + "…").strip()


def add_example(stats: TermStats, filename: str, snippet: str, definitional: bool) -> None:
    if not snippet:
        return
    for ex in stats.examples:
        if ex.filename == filename and ex.snippet == snippet:
            return

    if definitional:
        # Insert definitional examples first.
        stats.examples.insert(0, Example(filename=filename, snippet=snippet, definitional=True))
    else:
        stats.examples.append(Example(filename=filename, snippet=snippet, definitional=False))

    # Keep short per-term memory.
    if len(stats.examples) > 6:
        stats.examples = stats.examples[:6]


def collect_txt_files(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.rglob("*.txt") if p.is_file()]
    files.sort()
    return files


def _assign_sentence_ids(text: str, spans: List[Tuple[int, int]]) -> List[int]:
    """Return a sentence ID for each token span.

    Two tokens with the same ID are guaranteed to be in the same sentence.
    Sentence breaks are detected conservatively (period/!/? + whitespace +
    uppercase, or blank line) to avoid splitting on abbreviations or
    mid-sentence punctuation common in 17th-century prose.
    Spans must be sorted by start position (as produced by tokenize_with_spans).
    """
    break_positions = [m.start() for m in SENTENCE_BREAK_RE.finditer(text)]
    sent_ids: List[int] = []
    sent_id = 0
    bp_idx = 0
    for start, _end in spans:
        while bp_idx < len(break_positions) and break_positions[bp_idx] < start:
            sent_id += 1
            bp_idx += 1
        sent_ids.append(sent_id)
    return sent_ids


def _spacy_lemmatize(tokens: List[str], nlp: Any) -> List[str]:
    """Apply spaCy lemmatization to a pre-tokenized list.

    Joins tokens into a single string, parses with spaCy, then extracts
    lemmas. Falls back to the original tokens if the token count doesn't
    align (e.g. spaCy splits a contraction differently).
    """
    if not tokens:
        return tokens
    try:
        doc = nlp(" ".join(tokens))
        if len(doc) != len(tokens):
            return tokens
        return [tok.lemma_.lower() for tok in doc]
    except Exception:
        return tokens


def _load_nlp(enabled: bool) -> Optional[Any]:
    """Load spaCy en_core_web_sm when enabled. Returns None if disabled or unavailable."""
    if not enabled:
        return None
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        print(f"⚠️  spaCy unavailable ({exc}) — POS filter / lemmatization disabled")
        return None


def _build_pos_map(terms: List[str], nlp: Any) -> Dict[str, List[str]]:
    """Batch-POS-tag terms as full phrases.

    Tags each term as a complete phrase so that ambiguous tokens like "call"
    or "like" get resolved with their neighbours' context.

    Returns term → list of POS tags (one per token in the phrase).
    Uses nlp.pipe for efficiency; falls back to empty dict on failure.
    """
    unique = list(dict.fromkeys(terms))  # dedupe, preserve order
    if not unique:
        return {}
    try:
        docs = list(nlp.pipe(unique, batch_size=256, disable=["ner", "parser"]))
    except Exception:
        return {}
    return {term: [tok.pos_ for tok in doc] for term, doc in zip(unique, docs)}


def extract_terms_from_files(
    files: Sequence[Path],
    *,
    max_ngram: int = 3,
    lemmatize: bool = False,
    nlp: Optional[Any] = None,
) -> Tuple[Dict[Tuple[str, ...], TermStats], int, int]:
    # If no nlp was provided externally but lemmatize was requested, load now.
    if nlp is None and lemmatize:
        nlp = _load_nlp(True)

    term_stats: Dict[Tuple[str, ...], TermStats] = {}
    chapters_seen: set[int] = set()

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        filename = path.name
        chapter = parse_chapter(filename)
        if chapter is not None:
            chapters_seen.add(chapter)

        # surface_tokens: archaic-normalized, used for marker detection.
        # index_tokens:   additionally spaCy-lemmatized (wars→war, men→man),
        #                 used as n-gram keys. Same length as surface_tokens.
        # spans:          original char positions, used for snippet extraction.
        surface_tokens, spans = tokenize_with_spans(text)
        if not surface_tokens:
            continue

        index_tokens = _spacy_lemmatize(surface_tokens, nlp) if nlp is not None else surface_tokens
        # Sentence ID per token: two tokens with the same ID are in the same
        # sentence. Used to prevent cross-sentence definitional false positives.
        sentence_ids = _assign_sentence_ids(text, spans)
        centers = marker_centers(surface_tokens)
        seen_this_doc: set[Tuple[str, ...]] = set()
        last_center_idx = 0

        for i in range(len(index_tokens)):
            for n in range(1, max(1, max_ngram) + 1):
                j = i + n
                if j > len(index_tokens):
                    break
                ng = tuple(index_tokens[i:j])
                if not is_valid_candidate(ng):
                    continue

                st = term_stats.get(ng)
                if st is None:
                    st = TermStats(n=n)
                    term_stats[ng] = st

                st.freq_total += 1
                if ng not in seen_this_doc:
                    st.doc_freq += 1
                    seen_this_doc.add(ng)
                if chapter is not None:
                    st.chapters.add(chapter)

                definitional = False
                if centers:
                    while last_center_idx < len(centers) and centers[last_center_idx] < i - DEFINITIONAL_WINDOW:
                        last_center_idx += 1
                    for cpos in centers[max(0, last_center_idx - 1) : last_center_idx + 5]:
                        if (
                            abs(cpos - i) <= DEFINITIONAL_WINDOW
                            and sentence_ids[cpos] == sentence_ids[i]
                        ):
                            definitional = True
                            break
                if definitional:
                    st.definitional_hits += 1

                start_char = spans[i][0]
                end_char = spans[j - 1][1]
                snippet = make_snippet(text, start_char, end_char)
                add_example(st, filename, snippet, definitional)

    return term_stats, len(files), len(chapters_seen)


def finalize_records(
    term_stats: Dict[Tuple[str, ...], TermStats],
    n_docs: int,
    n_chapters: int,
    top_k: int,
    *,
    w_freq: float = DEFAULT_SCORE_W_FREQ,
    w_idf: float = DEFAULT_SCORE_W_IDF,
    w_def: float = DEFAULT_SCORE_W_DEF,
    w_len: float = DEFAULT_SCORE_W_LEN,
    pos_filter: bool = False,
    nlp: Optional[Any] = None,
    min_freq: int = 0,
    min_def: int = 0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    # Normalise weights so any positive combination works from the CLI.
    _total = w_freq + w_idf + w_def + w_len
    if _total <= 0:
        raise ValueError("Score weights must sum to a positive value.")
    w_freq, w_idf, w_def, w_len = w_freq / _total, w_idf / _total, w_def / _total, w_len / _total

    # Build POS map: tag each term as a full phrase so ambiguous tokens like
    # "call" or "like" are resolved in context of their neighbours.
    pos_map: Dict[str, List[str]] = {}
    if pos_filter and nlp is not None:
        all_terms: List[str] = [" ".join(ng) for ng in term_stats]
        pos_map = _build_pos_map(all_terms, nlp)
        print(f"  POS-tagged {len(pos_map)} terms")

    scored: List[Dict[str, object]] = []
    for ng, st in term_stats.items():
        chapter_freq = len(st.chapters)
        idf_like = math.log((n_docs + 1) / (st.doc_freq + 1))
        specificity = 1.0 / chapter_freq if chapter_freq > 0 else 0.0
        length_weight = LENGTH_WEIGHT.get(st.n, 0.0)

        # Weighted mix (weights are normalised above):
        # - w_freq: frequency keeps stable, recurrent concepts
        # - w_idf:  IDF × specificity favours discriminative terms
        # - w_def:  definitional hits reward explicit definition language
        # - w_len:  length_weight nudges phrase-level terms above single words
        score = (
            w_freq * math.log1p(st.freq_total)
            + w_idf * (idf_like * (1.0 + 0.25 * specificity))
            + w_def * math.log1p(st.definitional_hits)
            + w_len * length_weight
        )

        ex = [
            {"filename": e.filename, "snippet": e.snippet[:160], "definitional": e.definitional}
            for e in st.examples[:3]
        ]
        defs = [e.snippet[:160] for e in st.examples if e.definitional][:3]

        scored.append(
            {
                "term": " ".join(ng),
                "n": st.n,
                "score": round(score, 6),
                "freq_total": st.freq_total,
                "doc_freq": st.doc_freq,
                "chapter_freq": chapter_freq,
                "definitional_hits": st.definitional_hits,
                "idf_like": round(idf_like, 6),
                "specificity": round(specificity, 6),
                "chapters": sorted(st.chapters),
                "examples": ex,
                "definitions": defs,
            }
        )

    scored.sort(
        key=lambda r: (
            float(r["score"]),
            int(r["definitional_hits"]),
            int(r["freq_total"]),
            int(r["n"]),
        ),
        reverse=True,
    )
    prelim_kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []

    def _is_short_unigram(rec: Dict[str, object]) -> bool:
        return int(rec["n"]) == 1 and len(str(rec["term"])) <= 3

    def _remove_reason(rec: Dict[str, object]) -> Optional[str]:
        term = str(rec["term"])
        tokens = term.split()
        n = int(rec["n"])
        freq_total = int(rec["freq_total"])
        doc_freq = int(rec["doc_freq"])
        chapter_freq = int(rec["chapter_freq"])
        definitional_hits = int(rec["definitional_hits"])
        idf_like = float(rec["idf_like"])

        doc_ratio = doc_freq / max(1, n_docs)
        chapter_ratio = chapter_freq / max(1, n_chapters)
        def_ratio = definitional_hits / max(1, freq_total)

        if term in MARKER_PHRASES:
            return "marker_phrase"
        if n == 1 and tokens and tokens[0] in MARKER_TOKENS:
            return "marker_token"
        if n == 1 and term in GENERIC_NOISE_TERMS:
            return "generic_noise"
        # For phrases: reject if any component token is a marker or noise word.
        # spaCy is unreliable on archaic tokens so we use the explicit lists.
        if n >= 2:
            for tok in tokens:
                if tok in MARKER_TOKENS:
                    return "phrase_contains_marker"
                if tok in GENERIC_NOISE_TERMS:
                    return "phrase_contains_noise"

        # spaCy POS filter — terms are tagged as full phrases so ambiguous
        # tokens ("call", "like") are resolved with neighbour context.
        if pos_map:
            # ADP covers archaic prepositional adverbs (wherein, therein…)
            # that spaCy tags as ADP rather than ADV.
            _ALWAYS_BAD = {"PRON", "DET", "CCONJ", "SCONJ", "PART", "AUX", "NUM", "ADP"}
            pos_tags: List[str] = pos_map.get(term, [])
            if n == 1:
                pos = pos_tags[0] if pos_tags else "X"
                if pos in _ALWAYS_BAD:
                    return "pos_filter_grammatical"
                if pos == "ADV" and definitional_hits < 3:
                    return "pos_filter_adv_low_def"
                if pos == "VERB" and definitional_hits < 5:
                    return "pos_filter_verb_low_def"
            else:
                # For phrases: reject if any token is a function word or a
                # verb/adverb without strong definitional anchoring.
                # X (spaCy-unknown, typically Latin/archaic) is also rejected
                # in phrases unless definitional signal is strong.
                for pos in pos_tags:
                    if pos in _ALWAYS_BAD:
                        return "pos_filter_phrase_grammatical"
                    if pos in {"VERB", "ADV"} and definitional_hits < 5:
                        return "pos_filter_phrase_verb_adv"
                    if pos == "X" and definitional_hits < 5:
                        return "pos_filter_phrase_unknown"

        # Phrases (bigrams/trigrams) with no corpus signal are almost always noise.
        # A phrase needs at least one of: repeated occurrence OR definitional anchor.
        if n >= 2 and freq_total < 2 and definitional_hits == 0:
            return "phrase_no_signal"

        # Short unigrams are high-risk noise unless supported by corpus signals.
        if _is_short_unigram(rec) and definitional_hits == 0 and idf_like < 1.2:
            return "short_unigram_low_signal"
        if _is_short_unigram(rec) and doc_ratio > 0.25 and chapter_ratio > 0.35 and def_ratio < 0.20:
            return "short_unigram_too_spread"

        # Adaptive genericity filter for unigrams:
        # very spread terms with weak definitional signal are removed.
        if n == 1 and doc_ratio >= 0.45 and chapter_ratio >= 0.60 and def_ratio < 0.22:
            return "too_generic_unigram"

        # Keep low-information unigrams out unless they are strongly discriminative.
        if n == 1 and idf_like < 0.75 and definitional_hits == 0 and freq_total < 8:
            return "low_signal_unigram"

        # Minimum frequency / definitional signal filter.
        # A term must satisfy freq >= min_freq OR def >= min_def (if either is set).
        if (min_freq > 0 or min_def > 0) and freq_total < min_freq and definitional_hits < min_def:
            return "below_min_signal"

        return None

    for rec in scored:
        reason = _remove_reason(rec)
        if reason is None:
            prelim_kept.append(rec)
        else:
            dropped = dict(rec)
            dropped["remove_reason"] = reason
            removed.append(dropped)

    # Build strong phrase token set from preliminarily kept bigrams/trigrams,
    # then suppress short unigrams that are mostly better represented in those phrases.
    if prelim_kept:
        prelim_scores = [float(r["score"]) for r in prelim_kept]
        prelim_median = median(prelim_scores)
        strong_phrase_floor = prelim_median + SHORT_UNIGRAM_SHADOW_MARGIN
    else:
        strong_phrase_floor = SHORT_UNIGRAM_SHADOW_MARGIN

    strong_phrases_terms: set[str] = set()
    strong_phrases_token_freq: Dict[str, int] = {}
    for rec in prelim_kept:
        if int(rec["n"]) < 2:
            continue
        if float(rec["score"]) < strong_phrase_floor:
            continue
        tokens = str(rec["term"]).split()
        strong_phrases_terms.update(tokens)
        phrase_freq = int(rec["freq_total"])
        for tok in tokens:
            strong_phrases_token_freq[tok] = strong_phrases_token_freq.get(tok, 0) + phrase_freq

    kept: List[Dict[str, object]] = []
    for rec in prelim_kept:
        if _is_short_unigram(rec) and str(rec["term"]) in strong_phrases_terms:
            # Keep short unigrams that have their own strong signal, even if they also
            # appear in strong phrases. This avoids over-pruning concept words.
            definitional_hits = int(rec["definitional_hits"])
            idf_like = float(rec["idf_like"])
            freq_total = int(rec["freq_total"])
            def_ratio = definitional_hits / max(1, freq_total)
            if definitional_hits >= 5 and idf_like >= 1.2:
                kept.append(rec)
                continue

            token = str(rec["term"])
            phrase_support = strong_phrases_token_freq.get(token, 0)
            support_ratio = phrase_support / max(1, freq_total)
            if support_ratio < SHORT_UNIGRAM_SHADOW_MIN_COVERAGE or def_ratio >= 0.20:
                kept.append(rec)
                continue

            dropped = dict(rec)
            dropped["remove_reason"] = "short_unigram_shadowed"
            removed.append(dropped)
            continue
        kept.append(rec)

    if top_k > 0:
        kept = kept[:top_k]
    return kept, removed


def write_json(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False), encoding="utf-8")


def write_json_obj(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_base = [
        "term",
        "n",
        "score",
        "freq_total",
        "doc_freq",
        "chapter_freq",
        "definitional_hits",
        "idf_like",
        "specificity",
        "chapters",
        "examples",
        "definitions",
    ]
    extras: List[str] = []
    seen = set(fields_base)
    for r in records:
        for k in r.keys():
            if k not in seen:
                extras.append(k)
                seen.add(k)
    fields = fields_base + extras
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = dict(r)
            row["chapters"] = "|".join(str(c) for c in r.get("chapters", []))
            row["examples"] = json.dumps(r.get("examples", []), ensure_ascii=False)
            row["definitions"] = json.dumps(r.get("definitions", []), ensure_ascii=False)
            w.writerow(row)


def _rank_table(rows: Sequence[Dict[str, object]], limit: int) -> str:
    out = ["| # | term | n | score | freq | docs | ch | def_hits |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rows[:limit], 1):
        out.append(
            f"| {i} | {r['term']} | {r['n']} | {float(r['score']):.4f} | {r['freq_total']} | "
            f"{r['doc_freq']} | {r['chapter_freq']} | {r['definitional_hits']} |"
        )
    return "\n".join(out)


def _rank_removed_table(rows: Sequence[Dict[str, object]], limit: int) -> str:
    out = [
        "| # | term | n | score | freq | docs | ch | def_hits | remove_reason |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(rows[:limit], 1):
        out.append(
            f"| {i} | {r['term']} | {r['n']} | {float(r['score']):.4f} | {r['freq_total']} | "
            f"{r['doc_freq']} | {r['chapter_freq']} | {r['definitional_hits']} | {r.get('remove_reason', '')} |"
        )
    return "\n".join(out)


def write_term_list(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(r["term"]) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_overrides(path: Optional[Path]) -> Tuple[set[str], set[str]]:
    if not path:
        return set(), set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    force_keep = {str(t) for t in payload.get("force_keep", [])}
    force_reject = {str(t) for t in payload.get("force_reject", [])}
    overlap = force_keep & force_reject
    if overlap:
        raise SystemExit(f"overrides contains terms in both force_keep and force_reject: {sorted(overlap)[:10]}")
    return force_keep, force_reject


def _chunked(records: Sequence[Dict[str, object]], size: int) -> List[List[Dict[str, object]]]:
    if size <= 0:
        raise ValueError("batch size must be > 0")
    return [list(records[i : i + size]) for i in range(0, len(records), size)]


def _extract_response_text(resp: Any) -> str:
    txt = getattr(resp, "output_text", "")
    if txt:
        return str(txt)
    chunks: List[str] = []
    for out_item in getattr(resp, "output", []) or []:
        for c in getattr(out_item, "content", []) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                chunks.append(t)
    if chunks:
        return "".join(chunks).strip()
    raise RuntimeError("Unable to read text content from Responses API result")


def _llm_batch_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["term", "label", "short_rationale", "evidence", "canonical_term"],
                    "properties": {
                        "term": {"type": "string"},
                        "label": {"type": "string", "enum": list(LLM_LABELS)},
                        "short_rationale": {"type": "string", "maxLength": 140},
                        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                        "canonical_term": {"type": "string", "maxLength": 120},
                    },
                },
            }
        },
    }


def _validate_decisions_payload(payload: Dict[str, object], batch_terms: Sequence[str]) -> Dict[str, LLMDecision]:
    if not isinstance(payload, dict) or "decisions" not in payload:
        raise ValueError("missing decisions")
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")
    batch_set = set(batch_terms)
    out: Dict[str, LLMDecision] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("decision item must be an object")
        term = str(item.get("term", ""))
        if term not in batch_set:
            raise ValueError(f"term outside batch: {term}")
        label = str(item.get("label", ""))
        if label not in LLM_LABELS:
            raise ValueError(f"invalid label: {label}")
        rationale = str(item.get("short_rationale", "")).strip()[:140]
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        evidence_out = [str(e) for e in evidence[:2]]
        canonical_term = str(item.get("canonical_term", "")).strip().lower()[:120]
        # Canonical target is only meaningful for orthography-variant rejections.
        if label != "REJECT_ORTHOGRAPHY_VARIANT":
            canonical_term = ""
        elif canonical_term == term:
            canonical_term = ""
        out[term] = LLMDecision(
            term=term,
            label=label,
            short_rationale=rationale,
            evidence=evidence_out,
            canonical_term=canonical_term,
        )
    if set(out.keys()) != batch_set:
        missing = batch_set - set(out.keys())
        extra = set(out.keys()) - batch_set
        raise ValueError(f"batch mismatch missing={len(missing)} extra={len(extra)}")
    return out


def _safe_unsure(batch: Sequence[Dict[str, object]], reason: str) -> Dict[str, LLMDecision]:
    return {
        str(r["term"]): LLMDecision(
            term=str(r["term"]),
            label="UNSURE",
            short_rationale=reason[:140],
            evidence=[],
            canonical_term="",
        )
        for r in batch
    }


def _write_checkpoint(path: Path, decisions: Dict[str, LLMDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        term: {
            "term": d.term,
            "label": d.label,
            "short_rationale": d.short_rationale,
            "evidence": d.evidence,
            "canonical_term": d.canonical_term,
        }
        for term, d in decisions.items()
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint(path: Path) -> Dict[str, LLMDecision]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, LLMDecision] = {}
        for term, d in raw.items():
            if not isinstance(d, dict):
                continue
            label = str(d.get("label", "UNSURE"))
            if label not in LLM_LABELS:
                label = "UNSURE"
            out[term] = LLMDecision(
                term=str(d.get("term", term)),
                label=label,
                short_rationale=str(d.get("short_rationale", "")),
                evidence=list(d.get("evidence", [])),
                canonical_term=str(d.get("canonical_term", "")),
            )
        print(f"   checkpoint: {len(out)} decisions already loaded from {path}")
        return out
    except Exception as exc:
        print(f"⚠️  Failed to load checkpoint from {path}: {exc} — starting fresh")
        return {}


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True
    name = exc.__class__.__name__.lower()
    return any(tok in name for tok in ("rate", "timeout", "connection"))


def _build_llm_prompt(batch_payload: Sequence[Dict[str, object]]) -> str:
    categories = "\n".join(f"- {c}" for c in LLM_LABELS)
    return (
        "You are reviewing retrieval anchors for an editorial clean-up pass.\n"
        "Classify each provided term with one label.\n"
        "Rules:\n"
        "1) Do not add or remove terms; classify exactly the provided terms.\n"
        "2) short_rationale must be concise (<= 140 chars).\n"
        "3) evidence must contain 0-2 snippets copied only from the provided examples.\n"
        "4) If ambiguous, use UNSURE.\n"
        "5) canonical_term: fill ONLY for REJECT_ORTHOGRAPHY_VARIANT.\n"
        "   - Put the normalized canonical term string (e.g., war).\n"
        "   - If unknown, set canonical_term to empty string.\n"
        "   - For all non-orthography labels, canonical_term must be empty string.\n"
        f"Allowed labels:\n{categories}\n\n"
        f"Input batch JSON:\n{json.dumps(batch_payload, ensure_ascii=False)}"
    )


def _review_batch_with_split(
    batch: List[Dict[str, object]],
    *,
    client: Any,
    model: str,
    max_retries: int,
) -> Dict[str, LLMDecision]:
    """Call the LLM for a batch, splitting recursively on validation failure.

    Failure modes:
    - Rate-limit / server error  → retry with exponential backoff at the same size.
    - Schema / validation error  → split in half and recurse into each sub-batch.
    - Single-term failure (size 1) → fall back to UNSURE for that term.

    Always returns a complete dict for every term in the batch — never raises.
    """
    if not batch:
        return {}

    batch_terms = [str(r["term"]) for r in batch]
    batch_payload = [
        {
            "term": str(r["term"]),
            "n": int(r["n"]),
            "score": float(r["score"]),
            "definitional_hits": int(r["definitional_hits"]),
            "idf_like": float(r["idf_like"]),
            "doc_freq": int(r["doc_freq"]),
            "chapter_freq": int(r["chapter_freq"]),
            "examples": list(r.get("examples", []))[:3],
        }
        for r in batch
    ]
    schema = _llm_batch_schema()
    prompt = _build_llm_prompt(batch_payload)

    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a strict term-review classifier."},
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "term_review_batch",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            text = _extract_response_text(resp)
            parsed = json.loads(text)
            return _validate_decisions_payload(parsed, batch_terms)
        except Exception as exc:  # noqa: BLE001
            if _is_retryable_openai_error(exc):
                if attempt <= max_retries:
                    sleep_s = min(20.0, (1.7 ** (attempt - 1)) + random.uniform(0, 0.6))
                    time.sleep(sleep_s)
                    continue
                # Rate-limit retries exhausted.
                return _safe_unsure(batch, f"rate_limit_exhausted: {exc.__class__.__name__}")
            # Validation / schema failure: split if possible, otherwise give up.
            if len(batch) > 1:
                mid = len(batch) // 2
                left = _review_batch_with_split(
                    batch[:mid], client=client, model=model, max_retries=max_retries
                )
                right = _review_batch_with_split(
                    batch[mid:], client=client, model=model, max_retries=max_retries
                )
                return {**left, **right}
            return _safe_unsure(batch, f"validation_failure_atomic: {exc.__class__.__name__}")

    return _safe_unsure(batch, "max_retries_exhausted")


def run_llm_review(
    records: Sequence[Dict[str, object]],
    *,
    model: str,
    batch_size: int,
    start_index: int,
    max_items: int,
    timeout_s: int,
    max_retries: int,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, LLMDecision]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package is required for --llm-review") from exc

    client = OpenAI(timeout=timeout_s)
    safe_start = max(0, start_index)
    if max_items > 0:
        target_records = list(records[safe_start : safe_start + max_items])
    else:
        target_records = list(records[safe_start:])

    decisions: Dict[str, LLMDecision] = _load_checkpoint(checkpoint_path) if checkpoint_path else {}

    batches = list(_chunked(target_records, batch_size))
    total_batches = len(batches)
    width = len(str(total_batches))

    for bi, batch in enumerate(batches, 1):
        # Skip terms already processed in a previous run.
        batch = [r for r in batch if str(r["term"]) not in decisions]
        if not batch:
            continue
        if verbose:
            print(f"  batch {bi:{width}}/{total_batches}  ({len(batch)} terms)...", end="", flush=True)
        result = _review_batch_with_split(
            batch,
            client=client,
            model=model,
            max_retries=max_retries,
        )
        decisions.update(result)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, decisions)
        if verbose:
            print(" done")
    return decisions


def apply_llm_review(
    kept_records: Sequence[Dict[str, object]],
    decisions: Dict[str, LLMDecision],
    *,
    force_keep: set[str],
    force_reject: set[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    final_kept: List[Dict[str, object]] = []
    final_rejected: List[Dict[str, object]] = []
    for rec in kept_records:
        term = str(rec["term"])
        out = dict(rec)
        decision = decisions.get(term)
        if decision is None:
            out["llm_label"] = "UNSURE"
            out["llm_short_rationale"] = "Not reviewed in this run"
            out["llm_evidence"] = []
            out["llm_canonical_term"] = ""
        else:
            out["llm_label"] = decision.label
            out["llm_short_rationale"] = decision.short_rationale
            out["llm_evidence"] = decision.evidence
            out["llm_canonical_term"] = decision.canonical_term

        if term in force_keep:
            out["review_outcome"] = "KEPT_OVERRIDE_FORCE_KEEP"
            final_kept.append(out)
            continue
        if term in force_reject:
            out["review_outcome"] = "REJECTED_OVERRIDE_FORCE_REJECT"
            final_rejected.append(out)
            continue

        label = str(out["llm_label"])
        if label in REJECT_LABELS:
            out["review_outcome"] = "REJECTED_BY_LLM"
            final_rejected.append(out)
        else:
            # KEEP and UNSURE both stay in final kept.
            out["review_outcome"] = "KEPT_BY_POLICY"
            final_kept.append(out)
    return final_kept, final_rejected


def _dedupe_examples(
    left: Sequence[Dict[str, object]],
    right: Sequence[Dict[str, object]],
    *,
    max_items: int = 6,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    for seq in (left, right):
        for it in seq:
            filename = str(it.get("filename", ""))
            snippet = str(it.get("snippet", ""))
            key = (filename, snippet)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "filename": filename,
                "snippet": snippet[:160],
                "definitional": bool(it.get("definitional", False)),
            })
            if len(out) >= max_items:
                return out
    return out


def _dedupe_definitions(
    left: Sequence[str],
    right: Sequence[str],
    *,
    max_items: int = 3,
) -> List[str]:
    """Merge two lists of definitional snippet strings, keeping unique entries."""
    seen: set[str] = set()
    out: List[str] = []
    for s in (*left, *right):
        s = str(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) >= max_items:
                break
    return out


def _candidate_canonical_terms(term: str) -> List[str]:
    tokens = term.split()
    cands: List[str] = []
    if not tokens:
        return cands

    # Strip noisy left/right edge tokens first.
    if len(tokens) >= 2 and (len(tokens[0]) == 1 or tokens[0] in STOPWORDS or tokens[0] in MARKER_TOKENS):
        cands.append(" ".join(tokens[1:]))
    if len(tokens) >= 2 and (len(tokens[-1]) == 1 or tokens[-1] in STOPWORDS or tokens[-1] in MARKER_TOKENS):
        cands.append(" ".join(tokens[:-1]))

    # Generic discourse prefixes.
    if term.startswith("called "):
        cands.append(term[len("called ") :].strip())
    if term.startswith("that is "):
        cands.append(term[len("that is ") :].strip())
    if term.startswith("that is to say "):
        cands.append(term[len("that is to say ") :].strip())

    # Fallback subphrases.
    if len(tokens) >= 2:
        cands.append(" ".join(tokens[1:]))
        cands.append(" ".join(tokens[:-1]))

    out: List[str] = []
    seen: set[str] = set()
    for c in cands:
        cc = c.strip()
        if not cc or cc == term or cc in seen:
            continue
        seen.add(cc)
        out.append(cc)
    return out


def merge_llm_rejected_into_kept(
    final_kept: Sequence[Dict[str, object]],
    final_rejected: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Merge selected LLM rejected terms into canonical kept entries.

    This preserves auditability while preventing signal loss:
    - rejected term is tracked under aliases / llm_absorbed_from
    - frequencies and definitional hits are aggregated
    - chapter/doc spread is updated deterministically (doc spread estimated)
    """
    kept_map: Dict[str, Dict[str, object]] = {str(r["term"]): dict(r) for r in final_kept}
    kept_terms = set(kept_map.keys())
    unmerged: List[Dict[str, object]] = []
    merge_events: List[Dict[str, object]] = []

    for rej in final_rejected:
        label = str(rej.get("llm_label", ""))
        term = str(rej.get("term", ""))
        if label not in MERGEABLE_REJECT_LABELS:
            unmerged.append(dict(rej))
            continue

        candidates: List[str] = []
        llm_target = str(rej.get("llm_canonical_term", "")).strip().lower()
        # LLM-provided canonical mapping has priority for orthography variants.
        if label == "REJECT_ORTHOGRAPHY_VARIANT" and llm_target and llm_target in kept_terms and llm_target != term:
            candidates.append(llm_target)
        candidates.extend([c for c in _candidate_canonical_terms(term) if c in kept_terms and c not in candidates])
        if not candidates:
            if label == "REJECT_ORTHOGRAPHY_VARIANT":
                # Canonical modern spelling not present in corpus (Hobbes uses archaic
                # forms exclusively).  Re-admit the original term so it isn't silently
                # dropped — it has real corpus frequency and should stay in the glossary.
                readmitted = dict(rej)
                readmitted["merge_mode"] = "readmitted_no_canon"
                readmitted["llm_label"] = "KEEP"
                kept_map[term] = readmitted
                kept_terms.add(term)
            else:
                unmerged.append(dict(rej))
            continue
        # Deterministic canonical target preference.
        target = max(candidates, key=lambda t: _term_quality_key(kept_map[t]))
        canon = kept_map[target]

        canon_aliases = list(canon.get("aliases", [])) if isinstance(canon.get("aliases", []), list) else []
        if term not in canon_aliases and term != target:
            canon_aliases.append(term)
        canon["aliases"] = canon_aliases

        absorbed = list(canon.get("llm_absorbed_from", [])) if isinstance(canon.get("llm_absorbed_from", []), list) else []
        absorbed.append(
            {
                "term": term,
                "label": label,
                "short_rationale": str(rej.get("llm_short_rationale", "")),
            }
        )
        canon["llm_absorbed_from"] = absorbed

        # Aggregate count-like signals.
        canon["freq_total"] = int(canon.get("freq_total", 0)) + int(rej.get("freq_total", 0))
        canon["definitional_hits"] = int(canon.get("definitional_hits", 0)) + int(rej.get("definitional_hits", 0))

        # Chapter union is exact (we have chapter ids).
        canon_ch = set(int(c) for c in canon.get("chapters", []) if isinstance(c, int) or str(c).isdigit())
        rej_ch = set(int(c) for c in rej.get("chapters", []) if isinstance(c, int) or str(c).isdigit())
        ch_union = canon_ch | rej_ch
        ch_inter = canon_ch & rej_ch
        canon["chapters"] = sorted(ch_union)
        canon["chapter_freq"] = len(ch_union)

        # Doc union is estimated from chapter overlap to avoid naive overcount.
        canon_doc = int(canon.get("doc_freq", 0))
        rej_doc = int(rej.get("doc_freq", 0))
        overlap_ratio = len(ch_inter) / max(1, len(ch_union))
        overlap_est = int(round(min(canon_doc, rej_doc) * overlap_ratio))
        canon["doc_freq"] = max(canon_doc, canon_doc + rej_doc - overlap_est)

        canon["examples"] = _dedupe_examples(
            canon.get("examples", []) if isinstance(canon.get("examples", []), list) else [],
            rej.get("examples", []) if isinstance(rej.get("examples", []), list) else [],
        )
        canon["definitions"] = _dedupe_definitions(
            canon.get("definitions", []) if isinstance(canon.get("definitions", []), list) else [],
            rej.get("definitions", []) if isinstance(rej.get("definitions", []), list) else [],
        )
        canon["merge_mode"] = "llm_alias_aggregation_estimated_doc_union"

        merge_events.append(
            {
                "from": term,
                "to": target,
                "label": label,
                "llm_canonical_term": llm_target,
            }
        )

    merged_kept = list(kept_map.values())
    merged_kept.sort(
        key=lambda r: (
            float(r.get("score", 0.0)),
            int(r.get("definitional_hits", 0)),
            int(r.get("freq_total", 0)),
            int(r.get("n", 1)),
        ),
        reverse=True,
    )
    summary = {
        "kept_input": len(final_kept),
        "rejected_input": len(final_rejected),
        "merged_aliases": len(merge_events),
        "kept_output": len(merged_kept),
        "rejected_unmerged_output": len(unmerged),
        "mergeable_labels": sorted(MERGEABLE_REJECT_LABELS),
        "events": merge_events,
    }
    return merged_kept, unmerged, summary


def _llm_summary(
    decisions: Dict[str, LLMDecision],
    final_kept: Sequence[Dict[str, object]],
    final_rejected: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    counts: Dict[str, int] = {label: 0 for label in LLM_LABELS}
    for d in decisions.values():
        counts[d.label] = counts.get(d.label, 0) + 1
    rejected_by_category = {k: counts.get(k, 0) for k in LLM_LABELS if k.startswith("REJECT_")}
    top_rejected = sorted(
        [r for r in final_rejected if str(r.get("llm_label", "")).startswith("REJECT_")],
        key=lambda r: float(r.get("score", 0.0)),
        reverse=True,
    )[:20]
    return {
        "reviewed_terms": len(decisions),
        "kept": counts.get("KEEP", 0),
        "rejected": sum(v for k, v in rejected_by_category.items()),
        "unsure": counts.get("UNSURE", 0),
        "rejected_by_category": rejected_by_category,
        "top_rejected": [
            {
                "term": str(r["term"]),
                "label": str(r.get("llm_label", "")),
                "short_rationale": str(r.get("llm_short_rationale", "")),
                "score": float(r.get("score", 0.0)),
            }
            for r in top_rejected
        ],
        "final_kept_count": len(final_kept),
        "final_rejected_count": len(final_rejected),
    }


def _embed_terms_with_cache(
    records: Sequence[Dict[str, object]],
    *,
    model: str,
    cache_path: Path,
    batch_size: int,
    timeout_s: int,
    max_retries: int,
) -> "Any":
    """Return embeddings aligned to records, using on-disk cache when possible."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for --matrix-cleanup (pip install numpy)") from exc

    terms = [str(r["term"]) for r in records]
    # Stable cache key: SHA-256 of the *sorted* term list — order-insensitive.
    cache_key = hashlib.sha256(
        json.dumps(sorted(terms), ensure_ascii=False).encode()
    ).hexdigest()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            data = np.load(str(cache_path), allow_pickle=True)
            if str(data["cache_key"][0]) == cache_key:
                # Reorder stored embeddings to match the current terms ordering.
                term_to_idx: Dict[str, int] = {
                    t: i for i, t in enumerate(list(data["terms"]))
                }
                rows = [term_to_idx[t] for t in terms]
                return data["embeddings"][rows]
        except Exception:
            pass

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai package is required for --matrix-cleanup") from exc

    client = OpenAI(timeout=timeout_s)
    embs: List[List[float]] = []
    for i in range(0, len(terms), batch_size):
        batch = terms[i : i + batch_size]
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = client.embeddings.create(model=model, input=batch)
                embs.extend([d.embedding for d in resp.data])
                break
            except Exception as exc:  # noqa: BLE001
                retryable = _is_retryable_openai_error(exc)
                if attempt <= max_retries and retryable:
                    sleep_s = min(20.0, (1.7 ** (attempt - 1)) + random.uniform(0, 0.5))
                    time.sleep(sleep_s)
                    continue
                raise

    out = np.array(embs, dtype=np.float32)
    np.savez(
        str(cache_path),
        terms=np.array(terms, dtype=object),
        embeddings=out,
        cache_key=np.array([cache_key]),
    )
    return out


def _build_connected_components(
    normed: "Any",  # np.ndarray (n, dim), L2-normalised
    sim_threshold: float,
) -> Tuple[Dict[int, List[int]], Optional["Any"], str]:
    """Union-find clustering of term embeddings by cosine similarity.

    Tries FAISS (O(n·k)) first for scalability; falls back to a full numpy
    cosine-matrix (O(n²)) if FAISS is not installed.

    Returns:
        groups      – {root_idx: [member_indices]}
        sim_matrix  – full cosine matrix (numpy path only) or None (FAISS path)
        method      – "faiss" | "numpy"
    """
    import numpy as np

    n = len(normed)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sim_matrix: Optional["Any"] = None
    method = "numpy"

    try:
        import faiss  # type: ignore[import]

        # k=64 neighbours per term is enough for threshold ≥ 0.90.
        # For very dense corpora increase k or use range_search.
        k = min(n, 64)
        index = faiss.IndexFlatIP(normed.shape[1])
        index.add(normed)
        D, I = index.search(normed, k)
        for i in range(n):
            for j_pos in range(k):
                j = int(I[i, j_pos])
                if j != i and float(D[i, j_pos]) >= sim_threshold:
                    union(i, j)
        method = "faiss"
    except ImportError:
        # Numpy fallback: compute full n×n cosine matrix.
        sim_matrix = normed @ normed.T
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim_matrix[i, j]) >= sim_threshold:
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups, sim_matrix, method


def run_matrix_cleanup(
    records: Sequence[Dict[str, object]],
    *,
    embed_model: str,
    sim_threshold: float,
    cache_npz: Path,
    batch_size: int,
    timeout_s: int,
    max_retries: int,
    matrix_out_npz: Optional[Path] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Deterministic cleanup via term-term cosine similarity clustering.

    - Build term embeddings
    - Cluster terms above threshold (FAISS if available, numpy otherwise)
    - Keep one canonical term per cluster, move others to rejected as aliases
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for --matrix-cleanup (pip install numpy)") from exc

    if not records:
        return [], [], {"clusters": 0, "duplicates_removed": 0}

    embeddings = _embed_terms_with_cache(
        records,
        model=embed_model,
        cache_path=cache_npz,
        batch_size=max(1, batch_size),
        timeout_s=max(1, timeout_s),
        max_retries=max(0, max_retries),
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    groups, sim_matrix, cluster_method = _build_connected_components(normed, sim_threshold)

    if matrix_out_npz:
        if sim_matrix is not None:
            matrix_out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(matrix_out_npz),
                terms=np.array([r["term"] for r in records], dtype=object),
                sim=sim_matrix,
            )
        else:
            print("⚠️  --matrix-out-npz skipped: full matrix not computed when FAISS is used")

    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []
    duplicates_removed = 0
    for idxs in groups.values():
        if len(idxs) == 1:
            kept.append(dict(records[idxs[0]]))
            continue
        ranked = sorted(idxs, key=lambda i: _term_quality_key(records[i]), reverse=True)
        canon_idx = ranked[0]
        canon_term = str(records[canon_idx]["term"])
        alias_terms = [str(records[i]["term"]) for i in ranked[1:]]

        canon = dict(records[canon_idx])
        canon["aliases"] = alias_terms
        canon["matrix_cluster_size"] = len(idxs)
        kept.append(canon)

        for i in ranked[1:]:
            dropped = dict(records[i])
            dropped["remove_reason"] = "matrix_duplicate_alias"
            dropped["matrix_canonical"] = canon_term
            dropped["matrix_similarity_to_canonical"] = round(float(normed[canon_idx] @ normed[i]), 6)
            removed.append(dropped)
            duplicates_removed += 1

    kept.sort(
        key=lambda r: (
            float(r.get("score", 0.0)),
            int(r.get("definitional_hits", 0)),
            int(r.get("freq_total", 0)),
            int(r.get("n", 1)),
        ),
        reverse=True,
    )
    summary = {
        "input_terms": len(records),
        "output_terms": len(kept),
        "duplicates_removed": duplicates_removed,
        "clusters": len(groups),
        "sim_threshold": sim_threshold,
        "embed_model": embed_model,
        "cache_npz": str(cache_npz),
        "matrix_npz": str(matrix_out_npz) if matrix_out_npz else None,
        "cluster_method": cluster_method,
    }
    return kept, removed, summary


def write_llm_review_html(
    path: Path,
    *,
    llm_summary: Dict[str, object],
    final_kept: Sequence[Dict[str, object]],
    final_rejected: Sequence[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected_by_cat = llm_summary.get("rejected_by_category", {})
    cat_items = "".join(
        f"<li><code>{html.escape(str(k))}</code>: <strong>{v}</strong></li>"
        for k, v in sorted(dict(rejected_by_cat).items())
    )
    rejected_rows = sorted(final_rejected, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:200]
    kept_rows = sorted(final_kept, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:200]

    def _review_rows(rows: Sequence[Dict[str, object]]) -> str:
        out = [
            "<table><thead><tr>"
            "<th>#</th><th>term</th><th>label</th><th>rationale</th><th>score</th><th>freq</th><th>docs</th>"
            "</tr></thead><tbody>"
        ]
        for i, r in enumerate(rows, 1):
            out.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td>{html.escape(str(r.get('term', '')))}</td>"
                f"<td><code>{html.escape(str(r.get('llm_label', '')))}</code></td>"
                f"<td>{html.escape(str(r.get('llm_short_rationale', '')))}</td>"
                f"<td>{float(r.get('score', 0.0)):.4f}</td>"
                f"<td>{int(r.get('freq_total', 0))}</td>"
                f"<td>{int(r.get('doc_freq', 0))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    kept_table = _review_rows(kept_rows)
    rejected_table = _review_rows(rejected_rows)
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Review Results</title>
  <style>
    body {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7fb; color: #1f2937; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 22px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; margin-bottom: 14px; }}
    .meta {{ display:grid; gap:8px; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); }}
    .chip {{ border:1px solid #e5e7eb; border-radius:8px; background:#f9fafb; padding:8px; }}
    table {{ width:100%; border-collapse: collapse; font-size:13px; }}
    th, td {{ border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; }}
    thead th {{ background:#eff8f9; position:sticky; top:0; }}
    .table-wrap {{ overflow-x:auto; }}
    code {{ background:#eef2ff; border:1px solid #e5e7eb; border-radius:6px; padding:1px 4px; }}
    input {{ width:100%; max-width:420px; border:1px solid #e5e7eb; border-radius:8px; padding:8px 10px; margin:6px 0 10px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>LLM Review Results</h1>
      <div class="meta">
        <div class="chip">reviewed_terms<br /><strong>{llm_summary.get("reviewed_terms", 0)}</strong></div>
        <div class="chip">label KEEP<br /><strong>{llm_summary.get("kept", 0)}</strong></div>
        <div class="chip">label REJECT_*<br /><strong>{llm_summary.get("rejected", 0)}</strong></div>
        <div class="chip">label UNSURE<br /><strong>{llm_summary.get("unsure", 0)}</strong></div>
        <div class="chip">final kept<br /><strong>{llm_summary.get("final_kept_count", 0)}</strong></div>
        <div class="chip">final rejected<br /><strong>{llm_summary.get("final_rejected_count", 0)}</strong></div>
      </div>
      <h3 style="margin-top:12px;">Rejected By Category</h3>
      <ul>{cat_items}</ul>
    </section>
    <section class="card">
      <h2>Final Rejected (Top 200)</h2>
      <input id="rejSearch" type="search" placeholder="Filter rejected terms..." />
      <div class="table-wrap" id="rejTable">{rejected_table}</div>
    </section>
    <section class="card">
      <h2>Final Kept (Top 200)</h2>
      <input id="keptSearch" type="search" placeholder="Filter kept terms..." />
      <div class="table-wrap" id="keptTable">{kept_table}</div>
    </section>
  </div>
  <script>
    function attachFilter(inputId, tableId) {{
      const input = document.getElementById(inputId);
      const rows = Array.from(document.querySelectorAll(`#${{tableId}} tbody tr`));
      input?.addEventListener('input', () => {{
        const q = input.value.trim().toLowerCase();
        for (const row of rows) {{
          row.style.display = !q || row.textContent.toLowerCase().includes(q) ? '' : 'none';
        }}
      }});
    }}
    attachFilter('rejSearch', 'rejTable');
    attachFilter('keptSearch', 'keptTable');
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_report(
    path: Path,
    kept_records: Sequence[Dict[str, object]],
    removed_records: Sequence[Dict[str, object]],
    *,
    input_dir: Path,
    n_docs: int,
    n_chapters: int,
    matrix_summary: Optional[Dict[str, object]] = None,
    llm_summary: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kept_records:
        scores = [float(r["score"]) for r in kept_records]
        mn = min(scores)
        md = median(scores)
        mx = max(scores)
    else:
        mn = md = mx = 0.0

    by_def_hits = sorted(
        [r for r in kept_records if int(r["definitional_hits"]) > 0],
        key=lambda r: (int(r["definitional_hits"]), float(r["score"])),
        reverse=True,
    )

    score_floor = md
    chapter_specific = sorted(
        [r for r in kept_records if int(r["chapter_freq"]) <= 2 and float(r["score"]) >= score_floor],
        key=lambda r: (float(r["score"]), -int(r["chapter_freq"])),
        reverse=True,
    )
    kept_short_unigrams = sorted(
        [r for r in kept_records if int(r["n"]) == 1 and len(str(r["term"])) <= 3],
        key=lambda r: (float(r["score"]), int(r["definitional_hits"]), int(r["freq_total"])),
        reverse=True,
    )
    removed_short_unigrams = sorted(
        [r for r in removed_records if int(r["n"]) == 1 and len(str(r["term"])) <= 3],
        key=lambda r: (float(r["score"]), int(r["definitional_hits"]), int(r["freq_total"])),
        reverse=True,
    )

    removed_by_reason: Dict[str, int] = {}
    for r in removed_records:
        key = str(r.get("remove_reason", "unknown"))
        removed_by_reason[key] = removed_by_reason.get(key, 0) + 1
    reason_lines = (
        [f"- `{k}`: **{v}**" for k, v in sorted(removed_by_reason.items(), key=lambda x: x[1], reverse=True)]
        if removed_by_reason
        else ["- none"]
    )

    report = [
        "# Term Extraction Report",
        "",
        "## Corpus",
        f"- input_dir: `{input_dir}`",
        f"- files: **{n_docs}**",
        f"- chapters: **{n_chapters}**",
        f"- candidates kept: **{len(kept_records)}**",
        f"- candidates removed: **{len(removed_records)}**",
        "",
        "## Removal Diagnostics",
        *reason_lines,
        "",
        "## Score Distribution",
        f"- min: `{mn:.4f}`",
        f"- median: `{md:.4f}`",
        f"- max: `{mx:.4f}`",
        "",
        "## Top 50 Terms (Overall)",
        _rank_table(kept_records, 50),
        "",
        "## Top 20 Definitional-Heavy Terms",
        _rank_table(by_def_hits, 20),
        "",
        "## Top 20 Chapter-Specific Terms",
        _rank_table(chapter_specific, 20),
        "",
        "## Short Unigrams",
        "### Top 20 Kept Short Unigrams",
        _rank_table(kept_short_unigrams, 20),
        "",
        "### Top 20 Removed Short Unigrams",
        _rank_removed_table(removed_short_unigrams, 20),
        "",
    ]
    if llm_summary:
        rejected_by_cat = llm_summary.get("rejected_by_category", {})
        by_cat_lines = [f"- `{k}`: **{v}**" for k, v in sorted(dict(rejected_by_cat).items())] or ["- none"]
        top_rejected = llm_summary.get("top_rejected", [])
        top_rej_lines = [
            "| # | term | label | rationale |",
            "|---:|---|---|---|",
        ]
        for i, row in enumerate(list(top_rejected)[:20], 1):
            top_rej_lines.append(
                f"| {i} | {row.get('term','')} | {row.get('label','')} | {row.get('short_rationale','')} |"
            )
        report.extend(
            [
                "## LLM Review Summary",
                f"- reviewed_terms: **{llm_summary.get('reviewed_terms', 0)}**",
                f"- kept: **{llm_summary.get('kept', 0)}**",
                f"- rejected: **{llm_summary.get('rejected', 0)}**",
                f"- unsure: **{llm_summary.get('unsure', 0)}**",
                "- rejected_by_category:",
                *by_cat_lines,
                "",
                "### Top 20 Rejected Terms (LLM)",
                *top_rej_lines,
                "",
            ]
        )
    if matrix_summary:
        report.extend(
            [
                "## Matrix Cleanup Summary",
                f"- input_terms: **{matrix_summary.get('input_terms', 0)}**",
                f"- output_terms: **{matrix_summary.get('output_terms', 0)}**",
                f"- duplicates_removed: **{matrix_summary.get('duplicates_removed', 0)}**",
                f"- clusters: **{matrix_summary.get('clusters', 0)}**",
                f"- sim_threshold: `{matrix_summary.get('sim_threshold', 0.0)}`",
                f"- embed_model: `{matrix_summary.get('embed_model', '')}`",
                f"- cache_npz: `{matrix_summary.get('cache_npz', '')}`",
                "",
            ]
        )
    path.write_text("\n".join(report), encoding="utf-8")


def _render_html_table(rows: Sequence[Dict[str, object]], limit: int, include_reason: bool = False) -> str:
    headers = ["#", "term", "n", "score", "freq", "docs", "ch", "def_hits"]
    if include_reason:
        headers.append("remove_reason")
    head = "".join(f"<th>{h}</th>" for h in headers)
    body_rows: List[str] = []
    for i, r in enumerate(rows[:limit], 1):
        cols = [
            str(i),
            html.escape(str(r["term"])),
            str(r["n"]),
            f"{float(r['score']):.4f}",
            str(r["freq_total"]),
            str(r["doc_freq"]),
            str(r["chapter_freq"]),
            str(r["definitional_hits"]),
        ]
        if include_reason:
            cols.append(html.escape(str(r.get("remove_reason", ""))))
        tds = "".join(f"<td>{c}</td>" for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html_report(
    path: Path,
    kept_records: Sequence[Dict[str, object]],
    removed_records: Sequence[Dict[str, object]],
    *,
    input_dir: Path,
    n_docs: int,
    n_chapters: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kept_records:
        scores = [float(r["score"]) for r in kept_records]
        mn = min(scores)
        md = median(scores)
        mx = max(scores)
    else:
        mn = md = mx = 0.0

    by_def_hits = sorted(
        [r for r in kept_records if int(r["definitional_hits"]) > 0],
        key=lambda r: (int(r["definitional_hits"]), float(r["score"])),
        reverse=True,
    )
    chapter_specific = sorted(
        [r for r in kept_records if int(r["chapter_freq"]) <= 2 and float(r["score"]) >= md],
        key=lambda r: (float(r["score"]), -int(r["chapter_freq"])),
        reverse=True,
    )
    kept_short_unigrams = sorted(
        [r for r in kept_records if int(r["n"]) == 1 and len(str(r["term"])) <= 3],
        key=lambda r: (float(r["score"]), int(r["definitional_hits"]), int(r["freq_total"])),
        reverse=True,
    )
    removed_short_unigrams = sorted(
        [r for r in removed_records if int(r["n"]) == 1 and len(str(r["term"])) <= 3],
        key=lambda r: (float(r["score"]), int(r["definitional_hits"]), int(r["freq_total"])),
        reverse=True,
    )

    removed_by_reason: Dict[str, int] = {}
    for r in removed_records:
        key = str(r.get("remove_reason", "unknown"))
        removed_by_reason[key] = removed_by_reason.get(key, 0) + 1
    reason_items = "".join(
        f"<li><code>{html.escape(k)}</code>: <strong>{v}</strong></li>"
        for k, v in sorted(removed_by_reason.items(), key=lambda x: x[1], reverse=True)
    ) or "<li>none</li>"

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Term Extraction Report</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef6f7 0%, var(--bg) 50%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; color: #0b5561; }}
    h3 {{ font-size: 16px; color: #0b5561; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      background: #f9fafb;
    }}
    .muted {{ color: var(--muted); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f0fdfa;
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 8px;
    }}
    td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
    }}
    tr:hover td {{ background: #f8fafc; }}
    code {{
      background: #ecfeff;
      border: 1px solid #ccfbf1;
      border-radius: 6px;
      padding: 1px 5px;
    }}
    input[type="search"] {{
      width: 100%;
      max-width: 420px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      margin-bottom: 10px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Term Extraction Report</h1>
      <div class="meta">
        <div class="chip"><span class="muted">input_dir</span><br /><code>{html.escape(str(input_dir))}</code></div>
        <div class="chip"><span class="muted">files</span><br /><strong>{n_docs}</strong></div>
        <div class="chip"><span class="muted">chapters</span><br /><strong>{n_chapters}</strong></div>
        <div class="chip"><span class="muted">kept</span><br /><strong>{len(kept_records)}</strong></div>
        <div class="chip"><span class="muted">removed</span><br /><strong>{len(removed_records)}</strong></div>
        <div class="chip"><span class="muted">score min/median/max</span><br /><strong>{mn:.4f} / {md:.4f} / {mx:.4f}</strong></div>
      </div>
    </section>

    <section class="card">
      <h2>Removal Diagnostics</h2>
      <ul>{reason_items}</ul>
    </section>

    <section class="card">
      <h2>Top 50 Terms (Overall)</h2>
      <input id="overallSearch" type="search" placeholder="Filter terms..." />
      <div class="table-wrap" id="overallTable">{_render_html_table(kept_records, 50)}</div>
    </section>

    <section class="card">
      <h2>Top 20 Definitional-Heavy Terms</h2>
      <div class="table-wrap">{_render_html_table(by_def_hits, 20)}</div>
    </section>

    <section class="card">
      <h2>Top 20 Chapter-Specific Terms</h2>
      <div class="table-wrap">{_render_html_table(chapter_specific, 20)}</div>
    </section>

    <section class="card">
      <h2>Short Unigrams</h2>
      <h3>Top 20 Kept Short Unigrams</h3>
      <div class="table-wrap">{_render_html_table(kept_short_unigrams, 20)}</div>
      <h3 style="margin-top:14px;">Top 20 Removed Short Unigrams</h3>
      <div class="table-wrap">{_render_html_table(removed_short_unigrams, 20, include_reason=True)}</div>
    </section>
  </div>
  <script>
    const input = document.getElementById('overallSearch');
    const rows = Array.from(document.querySelectorAll('#overallTable tbody tr'));
    input?.addEventListener('input', () => {{
      const q = input.value.trim().toLowerCase();
      for (const row of rows) {{
        const txt = row.textContent.toLowerCase();
        row.style.display = !q || txt.includes(q) ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_glossary_html(
    path: Path,
    kept_records: Sequence[Dict[str, object]],
    matrix_removed: Sequence[Dict[str, object]],
    extract_removed: Sequence[Dict[str, object]],
    *,
    n_docs: int,
    n_chapters: int,
    n_candidates: int,
    sim_threshold: float,
    llm_summary: Optional[Dict[str, object]] = None,
    llm_rejected: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    """Rich HTML glossary: final terms · matrix removed · extraction removed · LLM rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def esc(v: object) -> str:
        return html.escape(str(v))

    # ── chapter range for dot display ────────────────────────────────────────
    all_chs: List[int] = []
    for r in kept_records:
        chs = r.get("chapters")
        if isinstance(chs, list):
            all_chs.extend(int(c) for c in chs if isinstance(c, (int, float)))
    max_ch = max(all_chs) if all_chs else n_chapters

    def _chapter_dots(chapters: object) -> str:
        chs = set(int(c) for c in (chapters if isinstance(chapters, list) else []))
        dots = []
        for i in range(1, max_ch + 1):
            if i in chs:
                dots.append(f"<span class='dot dot-on' title='ch.{i}'>{i}</span>")
            else:
                dots.append(f"<span class='dot dot-off' title='ch.{i}'>{i}</span>")
        return "<span class='dots'>" + "".join(dots) + "</span>"

    def _def_badge(hits: object) -> str:
        h = int(hits) if isinstance(hits, (int, float)) else 0
        cls = "db-none" if h == 0 else "db-low" if h < 3 else "db-mid" if h < 8 else "db-high"
        return f"<span class='def-badge {cls}'>{h}</span>"

    max_freq = max((int(r.get("freq_total", 0)) for r in kept_records), default=1) or 1

    def _freq_bar(freq: object) -> str:
        f = int(freq) if isinstance(freq, (int, float)) else 0
        pct = min(100, int(f / max_freq * 100))
        return f"<div class='freq-bar'><div class='freq-fill' style='width:{pct}%'></div><span class='freq-n'>{f}</span></div>"

    # ── reason metadata for extraction-removed buckets ────────────────────────
    REASON_META: Dict[str, tuple] = {
        "below_min_signal":       ("#fee2e2", "#991b1b", "Sous le seuil (freq ET def)"),
        "phrase_no_signal":       ("#fef3c7", "#92400e", "Phrase sans signal"),
        "phrase_contains_noise":  ("#fce7f3", "#9d174d", "Phrase avec mot bruit"),
        "phrase_contains_marker": ("#ede9fe", "#5b21b6", "Phrase avec marqueur déf."),
        "generic_noise":          ("#f0fdf4", "#166534", "Bruit générique"),
        "marker_token":           ("#f0f9ff", "#075985", "Token marqueur déf."),
        "marker_phrase":          ("#f0f9ff", "#075985", "Phrase marqueur déf."),
        "short_unigram_low_signal":("#fff7ed", "#9a3412", "Unigramme court bas signal"),
        "short_unigram_too_spread":("#fff7ed", "#9a3412", "Unigramme court trop répandu"),
        "short_unigram_shadowed": ("#fff7ed", "#9a3412", "Unigramme court dans phrase"),
        "too_generic_unigram":    ("#fef3c7", "#92400e", "Unigramme trop générique"),
        "low_signal_unigram":     ("#fef3c7", "#92400e", "Unigramme bas signal"),
        "pos_filter_grammatical": ("#f5f3ff", "#4c1d95", "POS: mot grammatical"),
        "pos_filter_adv_low_def": ("#f5f3ff", "#4c1d95", "POS: adverbe faible déf."),
        "pos_filter_verb_low_def":("#f5f3ff", "#4c1d95", "POS: verbe faible déf."),
        "pos_filter_phrase_grammatical": ("#f5f3ff", "#4c1d95", "POS phrase: mot grammatical"),
        "pos_filter_phrase_verb_adv":    ("#f5f3ff", "#4c1d95", "POS phrase: verbe/adverbe"),
        "pos_filter_phrase_unknown":     ("#f5f3ff", "#4c1d95", "POS phrase: token inconnu"),
    }

    def _reason_style(reason: str) -> tuple:
        return REASON_META.get(reason, ("#f3f4f6", "#374151", reason))

    # group extract_removed by reason
    from collections import defaultdict as _defaultdict
    erem_buckets: Dict[str, List[Dict[str, object]]] = _defaultdict(list)
    for r in extract_removed:
        erem_buckets[str(r.get("remove_reason", "other"))].append(r)

    def _erem_reason_cards() -> str:
        parts = []
        for reason, items in sorted(erem_buckets.items(), key=lambda x: -len(x[1])):
            bg, fg, label = _reason_style(reason)
            parts.append(
                f"<div class='reason-card' style='background:{bg};border-left:4px solid {fg}' "
                f"onclick=\"jumpToReason('{reason}')\">"
                f"<span class='rc-count' style='color:{fg}'>{len(items)}</span>"
                f"<span class='rc-label'>{label}</span></div>"
            )
        return "\n".join(parts)

    def _erem_details() -> str:
        parts = []
        for reason, items in sorted(erem_buckets.items(), key=lambda x: -len(x[1])):
            bg, fg, label = _reason_style(reason)
            rows: List[str] = []
            for r in items[:200]:
                defs = r.get("definitions", [])
                snip = esc(str(defs[0])[:120]) if defs else ""
                rows.append(
                    f"<tr><td><strong>{esc(r['term'])}</strong></td>"
                    f"<td>{r['n']}-gram</td><td>{r['freq_total']}</td>"
                    f"<td>{r['definitional_hits']}</td>"
                    f"<td class='def-snip'>{snip}</td></tr>"
                )
            more = (f"<p class='more-note'>… et {len(items)-200} autres</p>"
                    if len(items) > 200 else "")
            parts.append(
                f"<details id='reason-{reason}' class='reason-bucket'>"
                f"<summary style='border-left:4px solid {fg};background:{bg}20'>"
                f"<span class='rb-label' style='color:{fg}'>{label}</span>"
                f"<span class='rb-count'>{len(items)} termes</span></summary>"
                f"<div class='rb-body'><table class='rb-table'><thead><tr>"
                f"<th>Terme</th><th>n</th><th>freq</th><th>def</th><th>Snippet</th>"
                f"</tr></thead><tbody>{''.join(rows)}</tbody></table>{more}</div></details>"
            )
        return "\n".join(parts)

    # ── kept rows ─────────────────────────────────────────────────────────────
    def _krow(r: Dict[str, object]) -> str:
        defs = r.get("definitions", [])
        def_snip = esc(str(defs[0])[:130]) if defs else ""
        n = int(r["n"])
        tag_cls = "tag-bi" if n == 2 else "tag-tri" if n == 3 else "tag-uni"
        tag_lbl = f"{n}-gram"
        llm = esc(str(r.get("llm_label", "")))
        llm_td = f"<td><span class='llm-tag'>{llm}</span></td>" if llm else "<td></td>"
        return (
            f"<tr data-n='{n}' data-term='{esc(r['term'])}' data-def='{1 if int(r.get('definitional_hits',0))>0 else 0}'>"
            f"<td><strong>{esc(r['term'])}</strong></td>"
            f"<td><span class='ngram-tag {tag_cls}'>{tag_lbl}</span></td>"
            f"<td>{_freq_bar(r['freq_total'])}</td>"
            f"<td>{_def_badge(r['definitional_hits'])}</td>"
            f"<td><span class='idf-val'>{float(r['idf_like']):.2f}</span></td>"
            f"<td>{_chapter_dots(r.get('chapters', []))}</td>"
            f"<td class='def-snip'>{def_snip}</td>"
            f"{llm_td}</tr>"
        )

    # ── matrix-removed rows & buckets ────────────────────────────────────────
    thresh_floor = round(math.floor(sim_threshold / 0.05) * 0.05, 10)
    _raw_edges: List[float] = []
    _e = thresh_floor
    while _e < 1.0:
        _raw_edges.append(round(_e, 10))
        _e = round(_e + 0.05, 10)
    _raw_edges.append(1.01)
    BUCKET_LOW_EDGES: List[float] = [_raw_edges[i] for i in range(len(_raw_edges) - 1)][::-1]
    BUCKET_LABELS: List[str] = [
        f"{low:.2f} – {min(low + 0.05, 1.0):.2f}" for low in BUCKET_LOW_EDGES
    ]

    def _sim_to_label_idx(sim: float) -> int:
        for i, low in enumerate(BUCKET_LOW_EDGES):
            if sim >= low:
                return i
        return len(BUCKET_LABELS) - 1

    bucketed: Dict[int, List[Dict[str, object]]] = {}
    for r in matrix_removed:
        sim = r.get("matrix_similarity_to_canonical")
        if not isinstance(sim, float):
            try:
                sim = float(str(sim))
            except (ValueError, TypeError):
                sim = 1.0
        bucketed.setdefault(_sim_to_label_idx(sim), []).append(r)

    def _mrow(r: Dict[str, object]) -> str:
        sim = r.get("matrix_similarity_to_canonical", "—")
        sim_s = f"{float(sim):.3f}" if isinstance(sim, (int, float)) else str(sim)
        defs = r.get("definitions", [])
        def_snip = esc(str(defs[0])[:110]) if defs else "—"
        return (
            f"<tr><td><strong>{esc(r['term'])}</strong></td>"
            f"<td>{r['n']}</td><td>{float(r['score']):.3f}</td>"
            f"<td>{r['freq_total']}</td><td>{r['definitional_hits']}</td>"
            f"<td>{esc(str(r.get('matrix_canonical','')))} <small style='color:#6b7280'>({sim_s})</small></td>"
            f"<td class='def-snip'>{def_snip}</td></tr>"
        )

    bucket_sections = []
    for label_idx, label in enumerate(BUCKET_LABELS):
        entries = bucketed.get(label_idx, [])
        is_lowest = label_idx == len(BUCKET_LABELS) - 1
        badge = " <span class='badge-thresh'>seuil utilisé</span>" if is_lowest else ""
        rows_html = "\n".join(_mrow(r) for r in entries) if entries else (
            "<tr><td colspan='7' style='color:#9ca3af;text-align:center;padding:14px'>"
            "Aucun terme dans cette tranche</td></tr>"
        )
        bucket_sections.append(
            f"<details {'open' if entries else ''} class='bucket'>"
            f"<summary><span class='bucket-label'>{label}</span>"
            f"<span class='bucket-count'>{len(entries)} terme{'s' if len(entries)!=1 else ''}</span>"
            f"{badge}</summary>"
            f"<div class='tbl-wrap'><table><thead><tr>"
            f"<th>Terme</th><th>n</th><th>score</th><th>freq</th><th>def_hits</th>"
            f"<th>→ canonique (sim)</th><th>définition</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table></div></details>"
        )
    mrem_html = "\n".join(bucket_sections)

    # ── LLM rejected tab ─────────────────────────────────────────────────────
    def _lrow(r: Dict[str, object]) -> str:
        label = str(r.get("llm_label", ""))
        tag_style = ("background:#dbeafe;color:#1e40af" if label == "REJECT_ORTHOGRAPHY_VARIANT"
                     else "background:#fee2e2;color:#991b1b")
        canon = esc(str(r.get("llm_canonical_term") or r.get("canonical_term") or ""))
        canon_td = f"<td><em>{canon}</em></td>" if canon else "<td></td>"
        rationale = esc(str(r.get("llm_short_rationale", "")))
        return (
            f"<tr><td><strong>{esc(r['term'])}</strong></td>"
            f"<td>{r['n']}</td><td>{float(r['score']):.3f}</td>"
            f"<td>{r['freq_total']}</td><td>{r['definitional_hits']}</td>"
            f"<td><span class='llm-tag' style='{tag_style}'>{esc(label)}</span></td>"
            f"<td class='def-snip'>{rationale}</td>{canon_td}</tr>"
        )

    lrej_rows_html = "\n".join(_lrow(r) for r in (llm_rejected or []))
    if llm_rejected is not None:
        n_ortho = sum(1 for r in llm_rejected if r.get("llm_label") == "REJECT_ORTHOGRAPHY_VARIANT")
        n_other = sum(1 for r in llm_rejected if r.get("llm_label") == "REJECT_OTHER")
        lrej_count = len(llm_rejected)
        lrej_tab_html = f'<div class="tab" onclick="switchTab(\'lrej\',this)">LLM rejected ({lrej_count})</div>'
        lrej_pane_html = (
            f'<div id="pane-lrej" class="pane">'
            f'<div class="stat-row" style="margin-bottom:14px">'
            f'<div class="lrej-stat"><strong>{n_ortho}</strong>orthography variant</div>'
            f'<div class="lrej-stat"><strong>{n_other}</strong>reject other</div></div>'
            f'<div class="toolbar">'
            f'<input type="search" id="lS" placeholder="Filtrer…" oninput="ftable(\'lB\',this.value,\'lC\')"/>'
            f'<select onchange="llabelfilter(this.value)">'
            f'<option value="">label : tous</option>'
            f'<option value="REJECT_ORTHOGRAPHY_VARIANT">REJECT_ORTHOGRAPHY_VARIANT</option>'
            f'<option value="REJECT_OTHER">REJECT_OTHER</option></select>'
            f'<span id="lC" class="count-label">{lrej_count} termes</span></div>'
            f'<div class="tbl-wrap"><table><thead><tr>'
            f'<th>Terme</th><th>n</th><th>score</th><th>freq</th><th>def_hits</th>'
            f'<th>label</th><th>rationale</th><th>canonique</th>'
            f'</tr></thead><tbody id="lB">{lrej_rows_html}</tbody></table></div></div>'
        )
    else:
        lrej_tab_html = ""
        lrej_pane_html = ""

    # ── header stats ─────────────────────────────────────────────────────────
    n_with_defs = sum(1 for r in kept_records if r.get("definitions"))
    n_kept_uni = sum(1 for r in kept_records if int(r.get("n", 1)) == 1)
    n_kept_bi  = sum(1 for r in kept_records if int(r.get("n", 1)) >= 2)
    has_llm = any(r.get("llm_label") for r in kept_records)
    llm_th = "<th>LLM</th>" if has_llm else ""
    llm_stat = ""
    if llm_summary:
        llm_stat = (
            f'<div class="stat hi"><div class="n">{llm_summary.get("kept","?")}</div><div class="l">LLM kept</div></div>'
            f'<div class="stat wa"><div class="n">{llm_summary.get("rejected","?")}</div><div class="l">LLM rejected</div></div>'
        )

    kept_rows_html = "\n".join(_krow(r) for r in kept_records)

    doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Glossaire — Leviathan</title>
  <style>
    :root{{--bg:#f4f6fb;--card:#fff;--text:#1e293b;--muted:#64748b;--line:#e2e8f0;--acc:#0f766e;--acc2:#134e4a;}}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:ui-sans-serif,-apple-system,sans-serif;background:var(--bg);color:var(--text);font-size:13px;}}
    header{{background:linear-gradient(135deg,#0f766e,#134e4a);color:#fff;padding:28px 36px 20px;}}
    header h1{{font-size:24px;font-weight:700;letter-spacing:-.3px;margin-bottom:4px;}}
    header .sub{{opacity:.7;font-size:12px;margin-bottom:18px;}}
    .stat-row{{display:flex;flex-wrap:wrap;gap:10px;}}
    .stat{{background:rgba(255,255,255,.15);border-radius:12px;padding:10px 18px;min-width:90px;}}
    .stat .n{{font-size:26px;font-weight:700;line-height:1;}}
    .stat .l{{font-size:10px;opacity:.65;text-transform:uppercase;letter-spacing:.6px;margin-top:3px;}}
    .stat.hi .n{{color:#86efac;}}.stat.wa .n{{color:#fbbf24;}}
    .lrej-stat{{background:#f9fafb;border:1px solid var(--line);border-radius:8px;padding:6px 14px;font-size:12px;}}
    .lrej-stat strong{{font-size:18px;display:block;}}
    .wrap{{max-width:1500px;margin:0 auto;padding:24px 28px;}}
    .tabs{{display:flex;gap:0;border-bottom:2px solid var(--line);margin-bottom:18px;}}
    .tab{{padding:10px 22px;cursor:pointer;border-bottom:3px solid transparent;font-weight:600;color:var(--muted);font-size:13px;transition:color .15s;}}
    .tab:hover{{color:var(--text);}}.tab.active{{color:var(--acc);border-bottom-color:var(--acc);}}
    .pane{{display:none;}}.pane.active{{display:block;}}
    .toolbar{{display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap;}}
    input[type=search],select{{border:1px solid var(--line);border-radius:8px;padding:7px 12px;font-size:12px;background:#fff;color:var(--text);outline:none;}}
    input[type=search]{{width:260px;}}
    input[type=search]:focus,select:focus{{border-color:var(--acc);}}
    .count-label{{color:var(--muted);font-size:12px;margin-left:4px;}}
    .tbl-wrap{{overflow-x:auto;max-height:62vh;overflow-y:auto;border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.07);}}
    table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px;}}
    thead th{{background:#f0fdfa;border-bottom:2px solid var(--line);padding:9px 10px;text-align:left;white-space:nowrap;position:sticky;top:0;z-index:2;}}
    thead th.sortable{{cursor:pointer;user-select:none;}}
    thead th.sortable:hover{{background:#ccfbf1;}}
    td{{padding:7px 10px;border-bottom:1px solid #f1f5f9;vertical-align:middle;}}
    tr:hover td{{background:#f8fffe;}}tr.hidden{{display:none;}}
    .ngram-tag{{display:inline-block;border-radius:5px;padding:1px 7px;font-size:10px;font-weight:600;}}
    .tag-uni{{background:#dbeafe;color:#1e40af;}}.tag-bi{{background:#fef9c3;color:#854d0e;}}.tag-tri{{background:#fce7f3;color:#9d174d;}}
    .freq-bar{{display:flex;align-items:center;gap:6px;}}
    .freq-fill{{height:8px;border-radius:4px;background:linear-gradient(90deg,#5eead4,#0f766e);min-width:2px;}}
    .freq-n{{color:var(--muted);font-size:11px;min-width:28px;}}
    .def-badge{{display:inline-block;border-radius:6px;padding:2px 7px;font-size:11px;font-weight:600;}}
    .db-none{{background:#f1f5f9;color:#94a3b8;}}.db-low{{background:#fef9c3;color:#a16207;}}
    .db-mid{{background:#d1fae5;color:#065f46;}}.db-high{{background:#0f766e;color:#fff;}}
    .idf-val{{color:var(--muted);font-size:11px;}}
    .dots{{display:flex;flex-wrap:wrap;gap:2px;max-width:200px;}}
    .dot{{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:4px;font-size:8px;font-weight:700;}}
    .dot-on{{background:#0f766e;color:#fff;}}.dot-off{{background:#e2e8f0;color:#94a3b8;}}
    .def-snip{{color:#374151;font-style:italic;max-width:380px;white-space:normal;line-height:1.4;}}
    .llm-tag{{display:inline-block;border-radius:5px;padding:1px 6px;font-size:10px;font-weight:600;background:#fef3c7;color:#92400e;}}
    .reason-cards{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px;}}
    .reason-card{{border-radius:10px;padding:12px 18px;cursor:pointer;min-width:150px;transition:opacity .15s;}}
    .reason-card:hover{{opacity:.82;}}
    .rc-count{{display:block;font-size:26px;font-weight:700;line-height:1;}}
    .rc-label{{display:block;font-size:11px;margin-top:4px;opacity:.8;}}
    .reason-bucket{{margin-bottom:10px;border-radius:10px;overflow:hidden;border:1px solid var(--line);}}
    .reason-bucket summary{{padding:12px 18px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;list-style:none;}}
    .reason-bucket summary::-webkit-details-marker{{display:none;}}
    .reason-bucket[open] summary{{border-bottom:1px solid var(--line);}}
    .rb-label{{font-weight:600;font-size:13px;}}.rb-count{{color:var(--muted);font-size:12px;}}
    .rb-body{{padding:12px;background:#fff;}}
    .rb-table{{width:100%;border-collapse:collapse;font-size:12px;}}
    .rb-table th{{background:#f8fafc;padding:7px 9px;text-align:left;border-bottom:2px solid var(--line);}}
    .rb-table td{{padding:6px 9px;border-bottom:1px solid #f1f5f9;}}
    .more-note{{color:var(--muted);font-size:11px;padding:8px 0 4px;text-align:center;}}
    details.bucket{{margin-bottom:10px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;}}
    details.bucket summary{{padding:11px 16px;cursor:pointer;display:flex;align-items:center;gap:10px;font-weight:500;list-style:none;background:#f9fafb;}}
    details.bucket summary::-webkit-details-marker{{display:none;}}
    details.bucket[open] summary{{background:#f0fdfa;border-bottom:1px solid var(--line);}}
    .bucket-label{{font-size:14px;font-weight:600;color:#0b5561;min-width:110px;}}
    .bucket-count{{color:var(--muted);font-size:12px;}}
    .badge-thresh{{background:#d1fae5;color:#065f46;border-radius:5px;padding:2px 7px;font-size:10px;font-weight:600;}}
  </style>
</head>
<body>
<header>
  <h1>Hobbes · Leviathan &nbsp;·&nbsp; Glossaire</h1>
  <p class="sub">docs: {n_docs} · chapters: {n_chapters} · candidates: {n_candidates} · sim_threshold: {sim_threshold}</p>
  <div class="stat-row">
    <div class="stat"><div class="n">{n_candidates}</div><div class="l">candidats</div></div>
    <div class="stat hi"><div class="n">{len(kept_records)}</div><div class="l">termes retenus</div></div>
    <div class="stat"><div class="n">{n_kept_uni}</div><div class="l">unigrammes</div></div>
    <div class="stat"><div class="n">{n_kept_bi}</div><div class="l">bi/trigrammes</div></div>
    <div class="stat"><div class="n">{n_with_defs}</div><div class="l">avec déf. hits</div></div>
    <div class="stat wa"><div class="n">{len(extract_removed)}</div><div class="l">rejetés extract.</div></div>
    <div class="stat wa"><div class="n">{len(matrix_removed)}</div><div class="l">rejetés matrix</div></div>
    {llm_stat}
  </div>
</header>
<div class="wrap">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('kept',this)">Termes retenus ({len(kept_records)})</div>
    <div class="tab" onclick="switchTab('erem',this)">Rejetés extraction ({len(extract_removed)})</div>
    <div class="tab" onclick="switchTab('mrem',this)">Rejetés matrix ({len(matrix_removed)})</div>
    {lrej_tab_html}
  </div>

  <!-- KEPT TERMS -->
  <div id="pane-kept" class="pane active">
    <div class="toolbar">
      <input type="search" id="kSearch" placeholder="Rechercher…" oninput="filterKept()"/>
      <select id="kNgram" onchange="filterKept()">
        <option value="">Tous les n-grams</option>
        <option value="1">Unigrammes ({n_kept_uni})</option>
        <option value="2">Bi/trigrammes ({n_kept_bi})</option>
      </select>
      <select id="kDef" onchange="filterKept()">
        <option value="">Tous (def hits)</option>
        <option value="1">Avec def. hits</option>
        <option value="0">Sans def. hits</option>
      </select>
      <span id="kCount" class="count-label">{len(kept_records)} termes</span>
    </div>
    <div class="tbl-wrap">
      <table><thead><tr>
        <th class="sortable" onclick="sortKept(0)">Terme</th>
        <th>Type</th>
        <th class="sortable" onclick="sortKept(2)">Fréquence</th>
        <th class="sortable" onclick="sortKept(3)">Def. hits</th>
        <th class="sortable" onclick="sortKept(4)">IDF</th>
        <th>Chapitres</th>
        <th>Snippet définitionnel</th>
        {llm_th}
      </tr></thead>
      <tbody id="kBody">{kept_rows_html}</tbody></table>
    </div>
  </div>

  <!-- EXTRACTION REMOVED (grouped by reason) -->
  <div id="pane-erem" class="pane">
    <div class="reason-cards">{_erem_reason_cards()}</div>
    {_erem_details()}
  </div>

  <!-- MATRIX REMOVED (bucketed by similarity) -->
  <div id="pane-mrem" class="pane">
    <div class="toolbar">
      <input type="search" id="mS" placeholder="Filtrer dans tous les buckets…" oninput="mfilter(this.value)"/>
      <span id="mC" class="count-label">{len(matrix_removed)} termes</span>
    </div>
    {mrem_html}
  </div>

  <!-- LLM REJECTED -->
  {lrej_pane_html}
</div>
<script>
function switchTab(id,el){{
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('pane-'+id).classList.add('active');
  el.classList.add('active');
}}
function filterKept(){{
  const q=document.getElementById('kSearch').value.toLowerCase();
  const ng=document.getElementById('kNgram').value;
  const def=document.getElementById('kDef').value;
  const rows=document.querySelectorAll('#kBody tr');
  let v=0;
  rows.forEach(r=>{{
    const matchQ=!q||r.dataset.term.includes(q);
    const matchNg=!ng||(ng==='2'?parseInt(r.dataset.n)>=2:r.dataset.n===ng);
    const matchDef=!def||(def===r.dataset.def);
    if(matchQ&&matchNg&&matchDef){{r.classList.remove('hidden');v++;}}
    else r.classList.add('hidden');
  }});
  document.getElementById('kCount').textContent=v+' termes';
}}
let _sd=[-1,-1,-1,-1,-1];
function sortKept(col){{
  _sd[col]*=-1;
  const body=document.getElementById('kBody');
  const rows=Array.from(body.querySelectorAll('tr'));
  rows.sort((a,b)=>{{
    if(col===0)return _sd[col]*a.dataset.term.localeCompare(b.dataset.term);
    if(col===2)return _sd[col]*(parseInt(a.querySelector('.freq-n')?.textContent||'0')-parseInt(b.querySelector('.freq-n')?.textContent||'0'));
    if(col===3)return _sd[col]*(parseInt(a.querySelector('.def-badge')?.textContent||'0')-parseInt(b.querySelector('.def-badge')?.textContent||'0'));
    if(col===4)return _sd[col]*(parseFloat(a.querySelector('.idf-val')?.textContent||'0')-parseFloat(b.querySelector('.idf-val')?.textContent||'0'));
    return 0;
  }});
  rows.forEach(r=>body.appendChild(r));
}}
function jumpToReason(reason){{
  switchTab('erem',document.querySelectorAll('.tab')[1]);
  const el=document.getElementById('reason-'+reason);
  if(el){{el.open=true;el.scrollIntoView({{behavior:'smooth',block:'start'}});}}
}}
function ftable(tbodyId,q,countId){{
  const rows=Array.from(document.getElementById(tbodyId).rows);
  const lq=q.trim().toLowerCase();
  let n=0;
  rows.forEach(r=>{{const show=!lq||r.textContent.toLowerCase().includes(lq);r.style.display=show?'':'none';if(show)n++;}});
  document.getElementById(countId).textContent=n+' termes';
}}
function llabelfilter(val){{
  const rows=Array.from(document.getElementById('lB').rows);
  let n=0;
  rows.forEach(r=>{{const show=!val||r.cells[5]?.textContent.trim()===val;r.style.display=show?'':'none';if(show)n++;}});
  document.getElementById('lC').textContent=n+' termes';
}}
function mfilter(q){{
  const lq=q.trim().toLowerCase();
  let n=0;
  document.querySelectorAll('#pane-mrem tbody tr').forEach(r=>{{
    const show=!lq||r.textContent.toLowerCase().includes(lq);
    r.style.display=show?'':'none';if(show)n++;
  }});
  document.getElementById('mC').textContent=n+' termes';
}}
</script>
</body>
</html>"""
    path.write_text(doc, encoding="utf-8")


def _resolve_versioned_output_dir(input_dir: Path) -> Path:
    # By convention, if input is "<book>/output", write outputs under "<book>/terms_extracted_vN".
    book_dir = input_dir.parent if input_dir.name == "output" else input_dir
    versions: List[int] = []
    if book_dir.exists():
        for entry in book_dir.iterdir():
            if not entry.is_dir():
                continue
            m = TERMS_OUTPUT_DIR_RE.match(entry.name)
            if m:
                versions.append(int(m.group(1)))
    next_version = (max(versions) + 1) if versions else 1
    out_dir = book_dir / f"terms_extracted_v{next_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Hobbesian terms/phrases from split corpus files.")
    p.add_argument("--input-dir", required=True, help="Directory containing .txt files (recursive).")
    p.add_argument("--out-json", default=None, help="Output JSON file.")
    p.add_argument("--out-csv", default=None, help="Output CSV file.")
    p.add_argument("--out-report", default=None, help="Output markdown report.")
    p.add_argument(
        "--out-removed-json",
        default=None,
        help="Optional JSON file containing removed candidates with remove_reason.",
    )
    p.add_argument(
        "--out-removed-csv",
        default=None,
        help="Optional CSV file containing removed candidates with remove_reason.",
    )
    p.add_argument("--top-k", type=int, default=0, help="Keep top-k terms after scoring (default: 0 = disabled).")
    p.add_argument("--min-freq", type=int, default=5, help="Minimum corpus frequency for a term to be kept (default: 5). A term passes if freq >= min-freq OR def >= min-def.")
    p.add_argument("--min-def", type=int, default=2, help="Minimum definitional hits for a term to be kept (default: 2). A term passes if freq >= min-freq OR def >= min-def.")
    p.add_argument("--score-w-freq", type=float, default=DEFAULT_SCORE_W_FREQ,
                   help=f"Score weight for term frequency (default: {DEFAULT_SCORE_W_FREQ}). Weights are normalised.")
    p.add_argument("--score-w-idf", type=float, default=DEFAULT_SCORE_W_IDF,
                   help=f"Score weight for IDF × specificity (default: {DEFAULT_SCORE_W_IDF}).")
    p.add_argument("--score-w-def", type=float, default=DEFAULT_SCORE_W_DEF,
                   help=f"Score weight for definitional hits (default: {DEFAULT_SCORE_W_DEF}).")
    p.add_argument("--score-w-len", type=float, default=DEFAULT_SCORE_W_LEN,
                   help=f"Score weight for n-gram length bonus (default: {DEFAULT_SCORE_W_LEN}).")
    p.add_argument("--max-ngram", type=int, default=2, choices=[1, 2, 3], help="Maximum n-gram length to extract.")
    p.add_argument(
        "--lemmatize",
        action="store_true",
        help="Apply spaCy lemmatization on top of archaic-forms normalization (requires: pip install spacy && python -m spacy download en_core_web_sm).",
    )
    p.add_argument(
        "--pos-filter",
        action="store_true",
        help=(
            "Apply spaCy POS-based filtering to unigrams: removes pronouns, determiners, "
            "conjunctions, particles, auxiliaries, numerals, low-def adverbs and verbs. "
            "Requires spaCy (pip install spacy && python -m spacy download en_core_web_sm)."
        ),
    )
    p.add_argument("--matrix-cleanup", action="store_true", help="Run deterministic cosine-matrix cleanup on kept terms.")
    p.add_argument(
        "--matrix-embed-model",
        default=DEFAULT_MATRIX_EMBED_MODEL,
        help=f"Embedding model for matrix cleanup (default: {DEFAULT_MATRIX_EMBED_MODEL}).",
    )
    p.add_argument("--matrix-sim-threshold", type=float, default=0.75, help="Cosine threshold for duplicate clustering.")
    p.add_argument("--matrix-batch-size", type=int, default=200, help="Embedding batch size for matrix cleanup.")
    p.add_argument("--matrix-timeout-s", type=int, default=60, help="Timeout for embedding calls in matrix cleanup.")
    p.add_argument("--matrix-max-retries", type=int, default=4, help="Max retries for embedding calls in matrix cleanup.")
    p.add_argument("--matrix-cache-npz", default=None, help="Optional path for embeddings cache npz.")
    p.add_argument("--matrix-out-npz", default=None, help="Optional path to persist cosine matrix npz.")
    p.add_argument("--llm-review", action="store_true", help="Run optional LLM editorial review on kept records.")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help=f"Model for LLM review (default: {DEFAULT_LLM_MODEL}).")
    p.add_argument("--llm-batch-size", type=int, default=25, help="Number of kept records per LLM batch.")
    p.add_argument("--llm-start-index", type=int, default=0, help="Start offset in kept records for LLM review.")
    p.add_argument("--llm-max-items", type=int, default=0, help="Optional cap of kept records reviewed (0 = all).")
    p.add_argument(
        "--llm-checkpoint",
        default=None,
        help="Path to checkpoint JSON for resumable LLM review. Auto-set if not provided.",
    )
    p.add_argument(
        "--llm-merge-rejected",
        action="store_true",
        help="Merge selected rejected LLM terms into canonical kept entries (aliases + aggregated stats).",
    )
    p.add_argument("--overrides-json", default=None, help="Optional overrides JSON with force_keep / force_reject.")
    p.add_argument("--openai-timeout-s", type=int, default=60, help="OpenAI client timeout in seconds.")
    p.add_argument("--openai-max-retries", type=int, default=6, help="Max retries on 429/5xx errors.")
    return p.parse_args()


def _step(label: str) -> None:
    """Print a section header line."""
    print(f"\n── {label} {'─' * max(2, 50 - len(label))}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a directory: {input_dir}")

    files = collect_txt_files(input_dir)
    if not files:
        raise SystemExit(f"No .txt files found under: {input_dir}")

    _step("extract")
    print(f"  input : {input_dir}  ({len(files)} files)")
    pos_filter: bool = args.pos_filter
    nlp: Optional[Any] = _load_nlp(args.lemmatize or pos_filter)
    if pos_filter and nlp is not None:
        print("  POS filter : active (spaCy en_core_web_sm)")
    elif pos_filter:
        print("  POS filter : requested but spaCy unavailable — skipping")
    stats, n_docs, n_chapters = extract_terms_from_files(
        files,
        max_ngram=max(1, args.max_ngram),
        nlp=nlp,
    )
    kept_records, removed_records = finalize_records(
        stats,
        n_docs=n_docs,
        n_chapters=n_chapters,
        top_k=max(0, args.top_k),
        w_freq=args.score_w_freq,
        w_idf=args.score_w_idf,
        w_def=args.score_w_def,
        w_len=args.score_w_len,
        pos_filter=pos_filter,
        nlp=nlp,
        min_freq=args.min_freq,
        min_def=args.min_def,
    )
    print(f"  docs: {n_docs}   chapters: {n_chapters}   candidates: {len(stats)}")
    print(f"  kept: {len(kept_records)}   removed: {len(removed_records)}")

    manual_core_outputs = [args.out_json, args.out_csv, args.out_report]
    manual_count = sum(1 for v in manual_core_outputs if v)
    if manual_count not in (0, 3):
        raise SystemExit("Provide either all of --out-json/--out-csv/--out-report, or none for auto versioned output.")

    if manual_count == 3:
        out_json = Path(args.out_json)
        out_csv = Path(args.out_csv)
        out_report = Path(args.out_report)
    else:
        auto_dir = _resolve_versioned_output_dir(input_dir)
        out_json = auto_dir / "terms.json"
        out_csv = auto_dir / "terms.csv"
        out_report = auto_dir / "terms_report.md"

    out_removed_json = Path(args.out_removed_json) if args.out_removed_json else out_json.with_name(f"{out_json.stem}_removed.json")
    out_removed_csv = Path(args.out_removed_csv) if args.out_removed_csv else out_csv.with_name(f"{out_csv.stem}_removed.csv")
    out_kept_terms = out_json.with_name(f"{out_json.stem}_kept_terms.txt")
    out_removed_terms = out_json.with_name(f"{out_json.stem}_removed_terms.txt")
    out_llm_review_json = out_json.with_name(f"{out_json.stem}_llm_review.json")
    out_final_json = out_json.with_name(f"{out_json.stem}_final.json")
    out_final_csv = out_json.with_name(f"{out_json.stem}_final.csv")
    out_final_terms = out_json.with_name(f"{out_json.stem}_final_terms.txt")
    out_final_rejected_json = out_json.with_name(f"{out_json.stem}_final_rejected.json")
    out_final_rejected_csv = out_json.with_name(f"{out_json.stem}_final_rejected.csv")
    out_final_rejected_terms = out_json.with_name(f"{out_json.stem}_final_rejected_terms.txt")
    out_final_merged_json = out_json.with_name(f"{out_json.stem}_final_merged.json")
    out_final_merged_csv = out_json.with_name(f"{out_json.stem}_final_merged.csv")
    out_final_merged_terms = out_json.with_name(f"{out_json.stem}_final_merged_terms.txt")
    out_final_rejected_unmerged_json = out_json.with_name(f"{out_json.stem}_final_rejected_unmerged.json")
    out_final_rejected_unmerged_csv = out_json.with_name(f"{out_json.stem}_final_rejected_unmerged.csv")
    out_final_rejected_unmerged_terms = out_json.with_name(f"{out_json.stem}_final_rejected_unmerged_terms.txt")
    out_llm_merge_json = out_json.with_name(f"{out_json.stem}_llm_merge_summary.json")
    out_llm_review_html = out_json.with_name(f"{out_json.stem}_llm_review.html")
    out_matrix_kept_json = out_json.with_name(f"{out_json.stem}_matrix_cleaned.json")
    out_matrix_kept_csv = out_json.with_name(f"{out_json.stem}_matrix_cleaned.csv")
    out_matrix_kept_terms = out_json.with_name(f"{out_json.stem}_matrix_cleaned_terms.txt")
    out_matrix_removed_json = out_json.with_name(f"{out_json.stem}_matrix_removed.json")
    out_matrix_removed_csv = out_json.with_name(f"{out_json.stem}_matrix_removed.csv")
    out_matrix_removed_terms = out_json.with_name(f"{out_json.stem}_matrix_removed_terms.txt")

    write_json(out_json, kept_records)
    write_csv(out_csv, kept_records)
    write_json(out_removed_json, removed_records)
    write_csv(out_removed_csv, removed_records)
    write_term_list(out_kept_terms, kept_records)
    write_term_list(out_removed_terms, removed_records)

    matrix_summary: Optional[Dict[str, object]] = None
    review_input_records: Sequence[Dict[str, object]] = kept_records
    if args.matrix_cleanup:
        _step("matrix cleanup")
        matrix_cache_npz = (
            Path(args.matrix_cache_npz)
            if args.matrix_cache_npz
            else out_json.with_name(f"{out_json.stem}_matrix_embeddings.npz")
        )
        matrix_out_npz = (
            Path(args.matrix_out_npz)
            if args.matrix_out_npz
            else out_json.with_name(f"{out_json.stem}_matrix_cosine.npz")
        )
        matrix_kept, matrix_removed, matrix_summary = run_matrix_cleanup(
            kept_records,
            embed_model=args.matrix_embed_model,
            sim_threshold=float(args.matrix_sim_threshold),
            cache_npz=matrix_cache_npz,
            batch_size=max(1, args.matrix_batch_size),
            timeout_s=max(1, args.matrix_timeout_s),
            max_retries=max(0, args.matrix_max_retries),
            matrix_out_npz=matrix_out_npz,
        )
        print(
            f"  method: {matrix_summary.get('cluster_method', '?')}   "
            f"threshold: {args.matrix_sim_threshold}   "
            f"input: {matrix_summary.get('input_terms', '?')}   "
            f"clusters: {matrix_summary.get('clusters', '?')}   "
            f"removed: {matrix_summary.get('duplicates_removed', '?')}"
        )
        write_json(out_matrix_kept_json, matrix_kept)
        write_csv(out_matrix_kept_csv, matrix_kept)
        write_term_list(out_matrix_kept_terms, matrix_kept)
        write_json(out_matrix_removed_json, matrix_removed)
        write_csv(out_matrix_removed_csv, matrix_removed)
        write_term_list(out_matrix_removed_terms, matrix_removed)
        review_input_records = matrix_kept

    llm_summary: Optional[Dict[str, object]] = None
    final_rejected: Optional[List[Dict[str, object]]] = None
    if args.llm_review:
        _step(f"llm review · {args.llm_model}")
        force_keep, force_reject = _load_overrides(Path(args.overrides_json) if args.overrides_json else None)
        out_llm_checkpoint = (
            Path(args.llm_checkpoint)
            if args.llm_checkpoint
            else out_json.with_name(f"{out_json.stem}_llm_checkpoint.json")
        )
        decisions = run_llm_review(
            review_input_records,
            model=args.llm_model,
            batch_size=max(1, args.llm_batch_size),
            start_index=max(0, args.llm_start_index),
            max_items=max(0, args.llm_max_items),
            timeout_s=max(1, args.openai_timeout_s),
            max_retries=max(0, args.openai_max_retries),
            checkpoint_path=out_llm_checkpoint,
            verbose=True,
        )
        final_kept, final_rejected = apply_llm_review(
            review_input_records,
            decisions,
            force_keep=force_keep,
            force_reject=force_reject,
        )
        llm_summary = _llm_summary(decisions, final_kept, final_rejected)
        print(
            f"  reviewed: {llm_summary.get('reviewed_terms', '?')}   "
            f"kept: {llm_summary.get('kept', '?')}   "
            f"rejected: {llm_summary.get('rejected', '?')}   "
            f"unsure: {llm_summary.get('unsure', '?')}"
        )
        write_json_obj(
            out_llm_review_json,
            {
                "model": args.llm_model,
                "reviewed_terms": len(decisions),
                "decisions": [
                    {
                        "term": d.term,
                        "label": d.label,
                        "short_rationale": d.short_rationale,
                        "evidence": d.evidence,
                        "canonical_term": d.canonical_term,
                    }
                    for d in decisions.values()
                ],
            },
        )
        write_json(out_final_json, final_kept)
        write_csv(out_final_csv, final_kept)
        write_term_list(out_final_terms, final_kept)
        write_json(out_final_rejected_json, final_rejected)
        write_csv(out_final_rejected_csv, final_rejected)
        write_term_list(out_final_rejected_terms, final_rejected)

        if args.llm_merge_rejected:
            merged_kept, rejected_unmerged, merge_summary = merge_llm_rejected_into_kept(final_kept, final_rejected)
            write_json(out_final_merged_json, merged_kept)
            write_csv(out_final_merged_csv, merged_kept)
            write_term_list(out_final_merged_terms, merged_kept)
            write_json(out_final_rejected_unmerged_json, rejected_unmerged)
            write_csv(out_final_rejected_unmerged_csv, rejected_unmerged)
            write_term_list(out_final_rejected_unmerged_terms, rejected_unmerged)
            write_json_obj(out_llm_merge_json, merge_summary)

        write_llm_review_html(
            out_llm_review_html,
            llm_summary=llm_summary,
            final_kept=final_kept,
            final_rejected=final_rejected,
        )

    write_report(
        out_report,
        kept_records,
        removed_records,
        input_dir=input_dir,
        n_docs=n_docs,
        n_chapters=n_chapters,
        matrix_summary=matrix_summary,
        llm_summary=llm_summary,
    )
    out_report_html = out_report.with_suffix(".html")
    write_html_report(
        out_report_html,
        kept_records,
        removed_records,
        input_dir=input_dir,
        n_docs=n_docs,
        n_chapters=n_chapters,
    )

    # ── Glossary HTML (rich, 3-tab) ────────────────────────────────────────
    # Final kept = best available: post-merge > post-llm > post-matrix > post-extract
    if args.llm_review and args.llm_merge_rejected:
        glossary_kept: Sequence[Dict[str, object]] = merged_kept
    elif args.llm_review:
        glossary_kept = final_kept
    elif args.matrix_cleanup:
        glossary_kept = matrix_kept
    else:
        glossary_kept = kept_records

    glossary_mrem: Sequence[Dict[str, object]] = matrix_removed if args.matrix_cleanup else []
    out_glossary_html = out_json.with_name(f"{out_json.stem}_glossary.html")
    write_glossary_html(
        out_glossary_html,
        glossary_kept,
        glossary_mrem,
        removed_records,
        n_docs=n_docs,
        n_chapters=n_chapters,
        n_candidates=len(stats),
        sim_threshold=float(args.matrix_sim_threshold) if args.matrix_cleanup else 0.0,
        llm_summary=llm_summary,
        llm_rejected=final_rejected if args.llm_review else None,
    )

    out_dir = out_json.parent

    _step("done ✓")

    # ── files written ─────────────────────────────────
    print(f"\n  {out_dir}/")
    print(f"    {'terms.json':<38}  {len(kept_records)} terms")
    print(f"    {'terms_removed.json':<38}  {len(removed_records)} terms")
    print(f"    terms_report.md / .html")

    if args.matrix_cleanup and matrix_summary:
        n_mk = len(matrix_kept)
        n_mr = len(matrix_removed)
        print(f"\n  [matrix]")
        print(f"    {'terms_matrix_cleaned.json':<38}  {n_mk} terms")
        print(f"    {'terms_matrix_removed.json':<38}  {n_mr} terms")

    if args.llm_review and llm_summary:
        n_fk = int(llm_summary.get("final_kept_count", 0))
        n_fr = int(llm_summary.get("final_rejected_count", 0))
        print(f"\n  [llm]")
        print(f"    {'terms_final.json':<38}  {n_fk} terms")
        print(f"    {'terms_final_rejected.json':<38}  {n_fr} terms")
        if args.llm_merge_rejected:
            print(f"    {'terms_final_merged.json':<38}  (merged aliases)")
        print(f"    llm_review.html")

    print(f"\n    {'terms_glossary.html':<38}  {len(glossary_kept)} terms (3-tab)")


if __name__ == "__main__":
    main()
