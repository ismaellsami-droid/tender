#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
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
from tender.engine.anchor_detection import (
    GlossaryLookup,
    LookupResult,
    OpenAIEmbeddingBackend,
)
from tender.engine.scoring import extract_keywords


DEFAULT_OUTPUT_DIR = ROOT / "runs"

SPELL_CORRECTION_MODEL = "gpt-4o-mini"
SPELL_CORRECTION_PROMPT = (
    "Correct only spelling mistakes in the following question. "
    "Do not rephrase, add, or remove any words. "
    "If there are no spelling mistakes, return the question unchanged. "
    "Return only the corrected question, nothing else."
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
    )
    return {"lookup": lookup, "glossary_by_term": glossary_by_term}


def _render_anchor_block(result: LookupResult, glossary_by_term: Dict[str, Dict[str, Any]]) -> str:
    if result.matched_step == "no_match" or result.canonical_term is None:
        return "<div class='no-anchor'>no match</div>"

    step_label = result.matched_step
    if result.satellite_term:
        step_label = f"satellite ({result.satellite_term})"

    anchor_entry = glossary_by_term.get(result.canonical_term, {})
    anchor_importance = anchor_entry.get("importance") or "—"

    neighbor_html = "<em>none</em>"
    if result.graph_neighbors:
        rows = []
        for neighbor in result.graph_neighbors[:12]:
            strength = neighbor.get("strength")
            strength_str = (
                f"{strength:.3f}" if isinstance(strength, (int, float))
                else ("—" if strength is None else html.escape(str(strength)))
            )
            importance = neighbor.get("importance") or "—"
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(neighbor['term'])}</code></td>"
                f"<td>{html.escape(neighbor['link_type'])}</td>"
                f"<td>{html.escape(importance)}</td>"
                f"<td style='text-align:right;'>{strength_str}</td>"
                "</tr>"
            )
        neighbor_html = (
            "<table class='neighbors-table'>"
            "<thead><tr>"
            "<th>Term</th>"
            "<th>Link type</th>"
            "<th>Glossary word important</th>"
            "<th>Link strength</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    score_str = f"{result.best_score:.4f}" if result.best_score is not None else "—"
    return (
        f"<div><b>anchor</b> <code>{html.escape(result.canonical_term)}</code>"
        f" <span class='badge'>{html.escape(step_label)}</span></div>"
        f"<div><b>best_score</b> {html.escape(score_str)}</div>"
        f"<div><b>glossary word important</b> {html.escape(str(anchor_importance))}</div>"
        f"<div><b>graph neighbors</b>{neighbor_html}</div>"
    )


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

        rows.append({
            "id": test["id"],
            "category": test.get("category"),
            "question": question,
            "corrected_question": corrected_question,
            "spelling_changed": corrected_question != question,
            "keywords": keywords,
            "lookup_results": lookup_results,
        })

    changed_count = sum(1 for row in rows if row["spelling_changed"])
    anchored_keywords = sum(
        1 for row in rows
        for r in row["lookup_results"]
        if r.matched_step != "no_match"
    )
    total_keywords = sum(len(row["lookup_results"]) for row in rows)

    cards: List[str] = []
    for row in rows:
        keyword_cards: List[str] = []
        for result in row["lookup_results"]:
            keyword_cards.append(
                f"""
                <div class="keyword-card">
                  <div><b>keyword</b> <code>{html.escape(result.keyword)}</code></div>
                  {_render_anchor_block(result, resources["glossary_by_term"])}
                </div>
                """
            )

        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(row['id'])}</h2>
              <div><b>category</b> {html.escape(str(row['category']))}</div>
              <div><b>question</b> {html.escape(row['question'])}</div>
              <div><b>corrected</b> {html.escape(row['corrected_question'])}</div>
              <div><b>spelling_changed</b> {str(row['spelling_changed']).lower()}</div>
              <div><b>keywords</b> {", ".join(f"<code>{html.escape(k)}</code>" for k in row['keywords']) or "<em>none</em>"}</div>
              {''.join(keyword_cards) or '<p><em>no keywords</em></p>'}
            </section>
            """
        )

    html_output = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Anchor Detection Report</title>
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px auto; max-width: 1200px; color: #1b1b1b; }}
          h1, h2 {{ margin-bottom: 8px; }}
          .summary {{ background: #f4f1ea; border: 1px solid #d8cfbf; padding: 16px; border-radius: 10px; margin-bottom: 24px; }}
          .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 20px; background: #fff; }}
          .keyword-card {{ border-top: 1px solid #eee; margin-top: 12px; padding-top: 12px; }}
          code {{ background: #f6f6f6; padding: 1px 5px; border-radius: 4px; }}
          .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e8eefc; color: #294a9b; font-size: 12px; margin-left: 8px; }}
          .no-anchor {{ color: #8a1c1c; font-style: italic; margin-top: 4px; }}
          .neighbors-table {{ border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 0.92em; }}
          .neighbors-table th, .neighbors-table td {{ border-top: 1px solid #eee; padding: 4px 8px; text-align: left; }}
          .neighbors-table th {{ background: #f7f7f7; }}
        </style>
      </head>
      <body>
        <h1>Anchor Detection Report</h1>
        <div class="summary">
          <div><b>book</b> {html.escape(book_id)}</div>
          <div><b>suite file</b> {html.escape(str(suite_path.relative_to(ROOT)))}</div>
          <div><b>generated</b> {html.escape(datetime.now().isoformat(timespec='seconds'))}</div>
          <div><b>mode</b> {"online_openai" if online_mode else "offline_local_fallback"}</div>
          <div><b>questions</b> {len(rows)}</div>
          <div><b>spelling changed</b> {changed_count}</div>
          <div><b>anchored keywords</b> {anchored_keywords} / {total_keywords}</div>
        </div>
        {''.join(cards)}
      </body>
    </html>
    """

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(out_root) / book_id / "anchors" / f"{ts}_{suite_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.html"
    output_path.write_text(html_output, encoding="utf-8")
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
