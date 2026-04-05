#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from eval.suite_loader import load_suite, resolve_suite_path
from reports.generate_exploration_mock import generate_exploration_mock
from tender.engine.anchor_detection import (
    GlossaryLookup,
    OpenAIEmbeddingBackend,
    OpenAIExplorationAdvisor,
)
from tender.engine.anchor_types import QuestionExplorationResult, QuestionReformulationResult
from tender.engine.scoring import extract_keywords


DEFAULT_OUTPUT_DIR = ROOT / "runs"

SPELL_CORRECTION_MODEL = "gpt-4o-mini"
SPELL_CORRECTION_PROMPT = (
    "Correct only spelling mistakes in the following question. "
    "Do not rephrase, add, or remove any words. "
    "If there are no spelling mistakes, return the question unchanged. "
    "Return only the corrected question, nothing else."
)

REFORMULATION_MODEL = "gpt-4o-mini"
REFORMULATION_PROMPT = (
    "You are judging local reformulation candidates for a retrieval system.\n"
    "For each candidate:\n"
    "1. Apply only minimal grammatical or stylistic corrections.\n"
    "2. Do not introduce new concepts.\n"
    "3. Do not change the intended substitution.\n"
    "Then choose the best corrected candidate.\n"
    "Rules:\n"
    "- You must choose only from the provided candidates after minimal correction.\n"
    "- Do not invent a completely new reformulation.\n"
    "- Select a candidate only if it is a faithful local substitution of the original question.\n"
    "- Reject candidates that remain unnatural, conceptually off, too broad, too narrow, or more exploratory than reformulative.\n"
    "- Prefer the candidate that stays closest to the original wording while making the key concept more canonical or retrieval-friendly.\n"
    "- If none of the candidates is clearly good, return NONE.\n"
    "- Return JSON only with keys: choice, corrected_candidates, reason.\n"
)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _correct_spelling(client: OpenAI, question: str) -> str:
    response = client.chat.completions.create(
        model=SPELL_CORRECTION_MODEL,
        messages=[
            {"role": "system", "content": SPELL_CORRECTION_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=256,
    )
    corrected = (response.choices[0].message.content or "").strip()
    return corrected or question


_REFORMULATION_STOPWORDS = {
    "a", "an", "the", "what", "does", "do", "did", "is", "are", "was", "were", "mean", "by",
    "how", "why", "when", "where", "who", "whom", "which", "about", "according", "to", "of",
    "and", "or", "in", "on", "for", "from", "with", "without", "between", "into", "than",
    "this", "that", "these", "those", "be", "being", "capable", "think", "we", "our", "their",
    "his", "her", "its", "it", "as", "at", "if", "then", "so", "much", "according", "question",
}

_TOKEN_NLP = None


def _get_token_nlp():
    global _TOKEN_NLP
    if _TOKEN_NLP is None:
        import spacy

        _TOKEN_NLP = spacy.load("en_core_web_sm")
    return _TOKEN_NLP


def _content_tokens(text: str) -> set[str]:
    doc = _get_token_nlp()(text)
    tokens: set[str] = set()
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        surface = token.text.lower()
        lemma = token.lemma_.lower().strip()
        if surface in _REFORMULATION_STOPWORDS:
            continue
        for normalized in (surface, lemma):
            if not normalized:
                continue
            if normalized in _REFORMULATION_STOPWORDS:
                continue
            if re.fullmatch(r"[a-z][a-z'-]*", normalized):
                tokens.add(normalized)
    return tokens


def _validate_reformulation(question: str, candidate_sentences: List[str], reformulated: str) -> tuple[str | None, str]:
    candidate = reformulated.strip()
    if not candidate:
        return None, "empty_response"
    if candidate.upper() == "NONE":
        return None, "llm_returned_none"
    if "?" not in candidate:
        return None, "missing_question_mark"

    original_tokens = _content_tokens(question)
    candidate_tokens = _content_tokens(candidate)
    if not candidate_tokens:
        return None, "no_content_tokens"
    if candidate_tokens == original_tokens:
        return None, "no_meaningful_change_from_original"
    normalized_candidates = {_normalize_sentence(text) for text in candidate_sentences if text.strip()}
    if normalized_candidates and _normalize_sentence(candidate) not in normalized_candidates:
        return None, "choice_not_in_corrected_candidates"
    return candidate, "accepted"


def _normalize_sentence(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _stage_enabled(suite_payload: Dict[str, Any], test: Dict[str, Any], key: str) -> bool:
    test_value = test.get(key)
    if isinstance(test_value, bool):
        return test_value
    suite_value = suite_payload.get(key)
    if isinstance(suite_value, bool):
        return suite_value
    return True


def _extract_json_payload(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Salvage common truncated JSON cases: keep fields we can still recover.
    choice_match = re.search(r'"choice"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    corrected_match = re.search(r'"corrected_candidates"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)

    if not choice_match and not corrected_match:
        return None

    corrected_candidates: list[str] = []
    if corrected_match:
        raw_items = corrected_match.group(1)
        for item in re.findall(r'"((?:[^"\\]|\\.)*)"', raw_items, flags=re.DOTALL):
            cleaned = bytes(item, "utf-8").decode("unicode_escape").strip()
            if cleaned:
                corrected_candidates.append(cleaned)

    payload: dict[str, Any] = {
        "choice": bytes(choice_match.group(1), "utf-8").decode("unicode_escape").strip() if choice_match else "",
        "corrected_candidates": corrected_candidates,
        "reason": bytes(reason_match.group(1), "utf-8").decode("unicode_escape").strip() if reason_match else "",
    }
    return payload


def _propose_reformulated_question(
    client: OpenAI,
    question: str,
    focus_spans: List[str],
    candidates: List[str],
) -> Dict[str, Any]:
    payload = {
        "question": question,
        "focus_spans": focus_spans,
        "candidates": candidates,
    }
    response = client.chat.completions.create(
        model=REFORMULATION_MODEL,
        messages=[
            {"role": "system", "content": REFORMULATION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=256,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return {
            "raw_response": None,
            "parsed_reformulated_question": None,
            "accepted_question": None,
            "postprocess_status": "filtered",
            "postprocess_reason": "empty_llm_response",
        }
    parsed = _extract_json_payload(content)
    if parsed is None:
        return {
            "raw_response": content,
            "parsed_reformulated_question": None,
            "accepted_question": None,
            "postprocess_status": "filtered",
            "postprocess_reason": "invalid_json",
        }
    corrected_candidates = [
        str(item).strip()
        for item in (parsed.get("corrected_candidates") or [])
        if isinstance(item, str) and str(item).strip()
    ]
    choice = str(parsed.get("choice") or "").strip()
    accepted, reason = _validate_reformulation(question, corrected_candidates, choice)
    return {
        "raw_response": content,
        "parsed_reformulated_question": choice or None,
        "corrected_candidates": corrected_candidates,
        "accepted_question": accepted,
        "postprocess_status": "accepted" if accepted else "filtered",
        "postprocess_reason": reason,
    }


class HashingTextEmbeddingBackend:
    """Local deterministic embedding fallback using hashed character and token features."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dim
        normalized = text.lower().strip()
        if not normalized:
            return vector

        padded = f"  {normalized}  "
        for i in range(len(padded) - 2):
            trigram = padded[i:i + 3]
            index = hash(("tri", trigram)) % self._dim
            vector[index] += 1.0

        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", normalized):
            index = hash(("tok", token)) % self._dim
            vector[index] += 2.0

        return vector


def _offline_correct_spelling(question: str, glossary_terms: List[str]) -> str:
    import difflib

    common_words = {
        "what", "does", "hobbes", "mean", "by", "state", "of", "nature", "is", "the",
        "difference", "between", "and", "are", "animals", "capable", "how", "do", "we",
        "know", "fair", "according", "think", "about", "sense", "time", "why", "humans",
        "fight", "so", "much", "spirituality", "meditation", "inner", "peace", "life",
        "without", "government", "human",
    }
    vocabulary = sorted(set(glossary_terms) | common_words)

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*|[^A-Za-z]+", question)
    corrected_tokens: List[str] = []
    for token in tokens:
        if not token.isalpha():
            corrected_tokens.append(token)
            continue
        lowered = token.lower()
        if lowered in vocabulary:
            corrected_tokens.append(token)
            continue
        matches = difflib.get_close_matches(lowered, vocabulary, n=1, cutoff=0.88)
        if not matches:
            corrected_tokens.append(token)
            continue
        replacement = matches[0]
        if token[0].isupper():
            replacement = replacement.capitalize()
        corrected_tokens.append(replacement)
    return "".join(corrected_tokens)


def _fallback_keywords(question: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", question.lower())
    stopwords = {
        "what", "does", "how", "why", "is", "the", "and", "of", "by",
        "to", "do", "we", "about", "according", "are", "or",
    }
    return [token for token in tokens if token not in stopwords]


def _extract_keywords(question: str) -> List[str]:
    result = extract_keywords(question)
    kept = result.get("kept") or []
    if kept:
        return kept
    return _fallback_keywords(question)


def _build_glossary_lookup(
    *,
    glossary_path: Path,
    graph_path: Path,
    embedding_backend: Any,
    exploration_advisor: Any = None,
) -> Dict[str, Any]:
    glossary_entries = json.loads(glossary_path.read_text(encoding="utf-8"))
    graph_nodes = json.loads(graph_path.read_text(encoding="utf-8"))

    glossary_by_term: Dict[str, Dict[str, Any]] = {}
    for entry in glossary_entries:
        term = entry.get("term")
        if isinstance(term, str) and term.strip():
            glossary_by_term[term.strip().lower()] = entry

    lookup = GlossaryLookup(
        glossary_entries=glossary_entries,
        graph_nodes=graph_nodes,
        embedding_backend=embedding_backend,
        glossary_by_term=glossary_by_term,
        exploration_advisor=exploration_advisor,
    )
    return {"lookup": lookup, "glossary_by_term": glossary_by_term}


def _serialize_lookup_result(result: Any, glossary_by_term: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "keyword": result.keyword,
        "matched_step": result.matched_step,
        "canonical_term": result.canonical_term,
        "satellite_term": result.satellite_term,
        "best_score": result.best_score,
        "graph_neighbors": result.graph_neighbors,
    }
    if result.canonical_term:
        entry = glossary_by_term.get(result.canonical_term, {})
        payload["anchor_entry"] = {
            "term": result.canonical_term,
            "importance": entry.get("importance"),
            "quote": entry.get("quote"),
            "frequency": entry.get("frequency"),
        }
    else:
        payload["anchor_entry"] = None
    trigger_neighbor = None
    if result.satellite_term:
        trigger_neighbor = next(
            (neighbor for neighbor in result.graph_neighbors if neighbor.get("term") == result.satellite_term),
            None,
        )
    payload["trigger_neighbor"] = trigger_neighbor
    return payload


def _serialize_question_exploration(
    status: str,
    source_mode: str,
    initial_candidates: List[Dict[str, Any]],
    extracted_concepts: List[str],
    span_pools: List[Dict[str, Any]],
    quote_pools: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    advice: Dict[str, Any],
    glossary_by_term: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    suggested_terms = []
    for term in advice.get("suggested_terms", []) or []:
        entry = glossary_by_term.get(str(term).strip().lower(), {})
        suggested_terms.append(
            {
                "term": str(term).strip().lower(),
                "importance": entry.get("importance"),
                "quote": entry.get("quote"),
            }
        )
    return {
        "status": status,
        "source_mode": source_mode,
        "initial_question_topk": initial_candidates,
        "extracted_concepts": extracted_concepts,
        "span_pools": span_pools,
        "quote_pools": quote_pools,
        "raw_candidates": candidates,
        "reason": advice.get("reason"),
        "suggested_terms": suggested_terms,
    }


def _serialize_question_reformulation(
    status: str,
    extracted_concepts: List[str],
    initial_candidates: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    span_term_matches: List[Dict[str, Any]],
    generated_candidates: List[Dict[str, Any]],
    proposed_question: str | None,
    llm_debug: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": status,
        "extracted_concepts": extracted_concepts,
        "initial_candidates": initial_candidates,
        "candidates": candidates,
        "span_term_matches": span_term_matches,
        "generated_candidates": generated_candidates,
        "proposed_question": proposed_question,
        "llm_debug": llm_debug,
    }


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Run glossary/anchor diagnostics for a book test suite.")
    parser.add_argument("--book-id", required=True, help="Book/corpus id")
    parser.add_argument("--suite", default=None, help="Suite name under books/<book_id>/tests/")
    parser.add_argument("--test-file", default=None, help="Optional explicit path to a suite JSON file")
    parser.add_argument("--out-root", default="runs", help="Root output dir for diagnostic runs")
    return parser.parse_args()


def generate_report(*, book_id: str, suite: Optional[str], test_file: Optional[str], out_root: str) -> Path:
    _load_env(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    online_mode = bool(api_key)
    client = OpenAI(api_key=api_key) if api_key else None
    suite_path = resolve_suite_path(book_id=book_id, suite=suite, test_file=test_file)
    suite_payload = load_suite(suite_path)
    glossary_path = ROOT / "books" / book_id / "glossary_terms.json"
    graph_path = ROOT / "books" / book_id / "glossary_graph.json"

    if client is not None:
        try:
            embedding_backend = OpenAIEmbeddingBackend(client)
            embedding_backend(["probe"])
        except Exception:
            online_mode = False
            embedding_backend = HashingTextEmbeddingBackend()
    else:
        embedding_backend = HashingTextEmbeddingBackend()

    resources = _build_glossary_lookup(
        glossary_path=glossary_path,
        graph_path=graph_path,
        embedding_backend=embedding_backend,
        exploration_advisor=OpenAIExplorationAdvisor(client) if client is not None else None,
    )
    lookup: GlossaryLookup = resources["lookup"]
    glossary_terms = lookup._canonical_terms

    tests = suite_payload["tests"]
    suite_id = suite_payload.get("suite_id", suite_path.stem)

    rows: List[Dict[str, Any]] = []
    for test in tests:
        question = test["question"]
        if online_mode and client is not None:
            try:
                corrected_question = _correct_spelling(client, question)
            except Exception:
                corrected_question = _offline_correct_spelling(question, glossary_terms)
                online_mode = False
        else:
            corrected_question = _offline_correct_spelling(question, glossary_terms)

        keywords = _extract_keywords(corrected_question)
        lookup_results = lookup.lookup_keywords(keywords)
        run_reformulation = _stage_enabled(suite_payload, test, "run_reformulation")
        run_exploration = _stage_enabled(suite_payload, test, "run_exploration")

        proposed_reformulation: str | None = None
        reformulation_status = "completed" if run_reformulation else "skipped"
        if run_reformulation:
            question_reformulation = lookup.suggest_question_reformulation(corrected_question, keywords)
            reformulation_debug: Dict[str, Any] = {
                "raw_response": None,
                "parsed_reformulated_question": None,
                "accepted_question": None,
                "postprocess_status": "not_run",
                "postprocess_reason": "llm_not_called",
            }
            if online_mode and client is not None:
                try:
                    generated_candidate_sentences = [
                        str(item.get("text", "")).strip()
                        for item in question_reformulation.generated_candidates
                        if str(item.get("text", "")).strip()
                    ]
                    if generated_candidate_sentences:
                        reformulation_debug = _propose_reformulated_question(
                            client,
                            corrected_question,
                            question_reformulation.extracted_concepts,
                            generated_candidate_sentences,
                        )
                    else:
                        reformulation_debug = {
                            "raw_response": None,
                            "parsed_reformulated_question": None,
                            "corrected_candidates": [],
                            "accepted_question": None,
                            "postprocess_status": "filtered",
                            "postprocess_reason": "no_generated_candidates",
                        }
                    proposed_reformulation = reformulation_debug.get("accepted_question")
                except Exception:
                    proposed_reformulation = None
                    reformulation_debug = {
                        "raw_response": None,
                        "parsed_reformulated_question": None,
                        "corrected_candidates": [],
                        "accepted_question": None,
                        "postprocess_status": "filtered",
                        "postprocess_reason": "llm_call_failed",
                    }
        else:
            question_reformulation = QuestionReformulationResult(question=corrected_question, keywords=keywords)
            reformulation_debug = {
                "raw_response": None,
                "parsed_reformulated_question": None,
                "accepted_question": None,
                "corrected_candidates": [],
                "postprocess_status": "skipped",
                "postprocess_reason": "stage_disabled",
            }

        exploration_status = "completed" if run_exploration else "skipped"
        if run_exploration:
            question_exploration = lookup.suggest_question_exploration(
                corrected_question,
                answer_context=test.get("answer_context"),
            )
        else:
            question_exploration = QuestionExplorationResult(
                question=corrected_question,
                source_mode="skipped",
                exploration_advice={"suggested_terms": [], "reason": "stage_disabled"},
            )

        rows.append({
            "id": test["id"],
            "category": test.get("category"),
            "question": question,
            "corrected_question": corrected_question,
            "spelling_changed": corrected_question != question,
            "keywords": keywords,
            "anchors": [
                _serialize_lookup_result(result, resources["glossary_by_term"])
                for result in lookup_results
                if result.matched_step != "no_match"
            ],
            "question_reformulation": _serialize_question_reformulation(
                reformulation_status,
                question_reformulation.extracted_concepts,
                question_reformulation.initial_candidates,
                question_reformulation.candidates,
                question_reformulation.span_term_matches,
                question_reformulation.generated_candidates,
                proposed_reformulation,
                reformulation_debug,
            ),
            "question_exploration": _serialize_question_exploration(
                exploration_status,
                question_exploration.source_mode,
                question_exploration.initial_candidates,
                question_exploration.extracted_concepts,
                question_exploration.span_pools,
                question_exploration.quote_pools,
                question_exploration.candidates,
                question_exploration.exploration_advice,
                resources["glossary_by_term"],
            ),
        })

    changed_count = sum(1 for row in rows if row["spelling_changed"])
    anchored_keywords = sum(
        1 for row in rows
        for r in row["anchors"]
    )
    total_keywords = sum(len(row["keywords"]) for row in rows)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(out_root) / book_id / "anchors" / f"{ts}_{suite_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.json"
    data = {
        "timestamp": ts,
        "run_type": "anchors",
        "book_id": book_id,
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "glossary_path": str(glossary_path),
        "graph_path": str(graph_path),
        "mode": "online_openai" if online_mode else "offline_local_fallback",
        "questions_count": len(rows),
        "spelling_changed_count": changed_count,
        "anchored_keywords": anchored_keywords,
        "total_keywords": total_keywords,
        "questions": rows,
    }
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "timestamp": ts,
        "run_type": "anchors",
        "book_id": book_id,
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "glossary_path": str(glossary_path),
        "graph_path": str(graph_path),
        "mode": "online_openai" if online_mode else "offline_local_fallback",
        "questions": len(rows),
        "spelling_changed": changed_count,
        "anchored_keywords": anchored_keywords,
        "total_keywords": total_keywords,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_exploration_mock(output_path)
    return output_path


if __name__ == "__main__":
    args = parse_args()
    path = generate_report(
        book_id=args.book_id,
        suite=args.suite,
        test_file=args.test_file,
        out_root=args.out_root,
    )
    print(path)
