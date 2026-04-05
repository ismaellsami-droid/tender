from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import spacy


_NLP = None

_GENERIC_WORDS = {
    "meaning",
    "mean",
    "understanding",
    "understand",
    "role",
    "kind",
    "type",
    "way",
    "explain",
    "explanation",
    "definition",
    "idea",
    "conception",
    "view",
    "thought",
    "think",
    "characterize",
    "characterization",
}

_FRAME_WORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "must",
    "will",
    "shall",
    "according",
    "mean",
    "understanding",
    "understand",
    "explain",
    "distinguish",
    "difference",
    "definition",
    "role",
    "hobbes",
}

_CONTENT_POS = {"NOUN", "PROPN", "ADJ"}
_GOOD_DEPS = {"dobj", "pobj", "attr", "nsubj", "nsubjpass", "conj", "appos", "oprd"}
_GENERIC_VERBS = {
    "be",
    "do",
    "have",
    "mean",
    "think",
    "say",
    "make",
    "use",
    "build",
    "describe",
    "define",
    "explain",
    "distinguish",
    "characterize",
    "know",
    "drive",
}


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass
class _Candidate:
    text: str
    score: float
    start: int
    end: int
    source: str


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _trim_chunk_tokens(tokens):
    trimmed = [t for t in tokens if not t.is_space and not t.is_punct]
    while trimmed and (trimmed[0].is_stop or trimmed[0].lower_ in {"'s", "’s"}):
        trimmed = trimmed[1:]
    while trimmed and (trimmed[-1].is_stop or trimmed[-1].lower_ in {"'s", "’s"}):
        trimmed = trimmed[:-1]
    return trimmed


def _candidate_text(tokens) -> str:
    return _normalize_text(" ".join(t.text for t in tokens))


def _score_chunk(tokens, root) -> float:
    content = [t for t in tokens if t.pos_ in _CONTENT_POS and not t.is_stop]
    if not content:
        return -100.0

    score = 0.0
    score += min(len(content), 3) * 1.2

    if len(content) >= 2:
        score += 1.5
    if any(t.dep_ in {"compound", "amod"} for t in content):
        score += 1.0
    if root.dep_ in _GOOD_DEPS:
        score += 1.0
    if root.pos_ in {"NOUN", "PROPN"}:
        score += 1.0

    root_lemma = root.lemma_.lower()
    if root_lemma in _GENERIC_WORDS:
        score -= 4.0

    if len(content) == 1 and content[0].lemma_.lower() in _FRAME_WORDS:
        score -= 5.0
    if len(content) == 1 and content[0].pos_ == "PROPN":
        score -= 4.0

    if all(t.lemma_.lower() in _FRAME_WORDS or t.is_stop for t in tokens):
        score -= 6.0

    return score


def _is_covered_by_larger(candidate: _Candidate, kept: Iterable[_Candidate]) -> bool:
    candidate_tokens = candidate.text.split()
    for other in kept:
        if other.text == candidate.text:
            continue
        other_tokens = other.text.split()
        if len(other_tokens) <= len(candidate_tokens):
            continue
        if all(token in other_tokens for token in candidate_tokens):
            return True
    return False


def _score_verb(token) -> float:
    if token.pos_ != "VERB":
        return -100.0
    lemma = token.lemma_.lower()
    if lemma in _GENERIC_VERBS or lemma in _FRAME_WORDS or token.is_stop:
        return -100.0
    if token.dep_ not in {"ROOT", "conj", "ccomp", "xcomp"}:
        return -100.0

    score = 1.8
    if token.dep_ == "ROOT":
        score += 0.8
    elif token.dep_ in {"ccomp", "xcomp"}:
        score += 0.5
    if any(child.dep_ in {"nsubj", "dobj", "obj"} for child in token.children):
        score += 0.6
    return score


def _score_contrast_pair(left, right) -> float:
    score = 3.6
    if left.pos_ == right.pos_:
        score += 0.4
    if left.pos_ in {"ADJ", "NOUN", "PROPN"}:
        score += 0.4
    return score


def _extract_contrast_pairs(doc) -> list[_Candidate]:
    pairs: list[_Candidate] = []
    seen: set[str] = set()
    for token in doc:
        if token.dep_ != "conj":
            continue
        head = token.head
        if head.i > token.i:
            continue
        if head.pos_ not in {"ADJ", "NOUN", "PROPN"} or token.pos_ not in {"ADJ", "NOUN", "PROPN"}:
            continue
        has_cc = any(child.dep_ == "cc" and child.lower_ in {"and", "or"} for child in head.children)
        if not has_cc:
            continue
        span = doc[head.i : token.i + 1]
        text = _candidate_text(_trim_chunk_tokens(list(span)))
        if not text or text in seen:
            continue
        pairs.append(_Candidate(text=text, score=_score_contrast_pair(head, token), start=head.i, end=token.i, source="contrast"))
        seen.add(text)
    return pairs


def _score_of_phrase(head, complement) -> float:
    score = 3.8
    if head.pos_ == "NOUN":
        score += 0.6
    if complement.pos_ == "NOUN":
        score += 0.4
    return score


def _extract_of_phrases(doc) -> list[_Candidate]:
    phrases: list[_Candidate] = []
    seen: set[str] = set()
    for token in doc:
        if token.dep_ != "pobj":
            continue
        prep = token.head
        if prep.dep_ != "prep" or prep.lower_ != "of":
            continue
        head = prep.head
        if head.pos_ not in {"NOUN", "PROPN"} or token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if head.lemma_.lower() in _GENERIC_WORDS or head.lemma_.lower() in _FRAME_WORDS:
            continue
        span = doc[head.i : token.i + 1]
        text = _candidate_text(_trim_chunk_tokens(list(span)))
        if not text or text in seen:
            continue
        phrases.append(
            _Candidate(
                text=text,
                score=_score_of_phrase(head, token),
                start=head.i,
                end=token.i,
                source="of_phrase",
            )
        )
        seen.add(text)
    return phrases


def _is_attached_complement(candidate: _Candidate, kept: Iterable[_Candidate], doc) -> bool:
    token = doc[candidate.end]
    if token.dep_ != "pobj" or token.head.dep_ != "prep":
        return False
    governor = token.head.head
    return any(other.start <= governor.i <= other.end for other in kept if other.text != candidate.text)


def extract_key_concepts(question: str) -> list[str]:
    """Extract a short ordered list of concept spans from a question.

    Heuristic pipeline:
    1. Parse with spaCy.
    2. Extract noun chunks.
    3. Filter/score chunks to prefer short, meaningful conceptual spans.
    4. Fallback to isolated nouns/proper nouns/adjectives when no good chunk exists.
    5. Deduplicate and remove terms subsumed by a stronger longer expression.
    """
    doc = _get_nlp()(question)

    candidates: list[_Candidate] = []
    seen: set[str] = set()

    for chunk in doc.noun_chunks:
        tokens = _trim_chunk_tokens(list(chunk))
        if not tokens:
            continue
        text = _candidate_text(tokens)
        if not text or text in seen:
            continue
        score = _score_chunk(tokens, tokens[-1].head if tokens[-1].dep_ == "poss" else chunk.root)
        if score <= -1.0:
            continue
        candidates.append(_Candidate(text=text, score=score, start=tokens[0].i, end=tokens[-1].i, source="chunk"))
        seen.add(text)

    for candidate in _extract_contrast_pairs(doc):
        if candidate.text in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate.text)

    for candidate in _extract_of_phrases(doc):
        if candidate.text in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate.text)

    for token in doc:
        score = _score_verb(token)
        if score <= -1.0:
            continue
        text = _normalize_text(token.lemma_ or token.text)
        if not text or text in seen:
            continue
        candidates.append(_Candidate(text=text, score=score, start=token.i, end=token.i, source="verb"))
        seen.add(text)

    if not candidates:
        for token in doc:
            if token.is_space or token.is_punct or token.is_stop:
                continue
            if token.pos_ not in _CONTENT_POS:
                continue
            lemma = token.lemma_.lower()
            if lemma in _FRAME_WORDS or lemma in _GENERIC_WORDS:
                continue
            text = _normalize_text(token.text)
            if not text or text in seen:
                continue
            score = 1.5
            if token.dep_ in _GOOD_DEPS:
                score += 0.5
            candidates.append(_Candidate(text=text, score=score, start=token.i, end=token.i, source="fallback"))
            seen.add(text)

    ranked = sorted(candidates, key=lambda c: (-c.score, c.start, -(c.end - c.start)))
    kept: list[_Candidate] = []
    for candidate in ranked:
        if _is_covered_by_larger(candidate, kept):
            continue
        if _is_attached_complement(candidate, kept, doc):
            continue
        kept.append(candidate)

    kept.sort(key=lambda c: (-c.score, c.start))
    return [item.text for item in kept]


if __name__ == "__main__":
    examples = [
        "What is Hobbes understanding of personal interest?",
        "What does Hobbes mean by logic?",
        "How does Hobbes distinguish prudence from science?",
        "What is the role of fear in the social contract?",
        "According to Hobbes, what characterizes human life without government?",
    ]
    for question in examples:
        print(question)
        print(" ->", extract_key_concepts(question))
        print()
