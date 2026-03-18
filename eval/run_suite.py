#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from tender.corpora.registry import default_corpus_id, get_corpus
from tender.engine.engine import _normalize_for_match
from tender.engine.pipeline import run_pipeline_with_policy
from eval.grader import grade_quotes_relevance
from eval.suite_loader import resolve_suite_path, load_suite
from eval.utils import (
    ensure_dir,
    safe_slug,
    get_git_commit,
    write_jsonl,
    render_html_report,
    compute_metrics,
    extract_extras_for_grading,
    extract_expected_quotes_for_grading,
)

# ----------------------------
# Helpers (for expected-based metrics)
# ----------------------------

_REF_CH_SEC_RE = re.compile(r"Ch\.\s*(\d{1,2}).*?§\s*(\d{1,2})", re.IGNORECASE)


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _expected_set(expected: List[Dict[str, int]]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for e in expected or []:
        ch = e.get("ch")
        sec = e.get("sec")
        if isinstance(ch, int) and isinstance(sec, int):
            out.add((ch, sec))
    return out


def _final_citations_set_from_trace(trace: Dict[str, Any]) -> Set[Tuple[int, int]]:
    events = (trace or {}).get("events") or []
    final_quotes = []
    for ev in reversed(events):
        if ev.get("event") == "quotes_selected":
            final_quotes = ev.get("quotes") or []
            break

    out: Set[Tuple[int, int]] = set()
    for q in final_quotes or []:
        ref = q.get("ref", "") or ""
        m = _REF_CH_SEC_RE.search(ref)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


def _retrieval_pool_map_from_selection(selection: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    pool = (selection or {}).get("pool") or []
    mp: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for it in pool:
        ch = it.get("chapter")
        sec = it.get("section")
        if not (isinstance(ch, int) and isinstance(sec, int)):
            continue
        key = (ch, sec)
        if key not in mp:
            mp[key] = {"rank": it.get("rank"), "score": it.get("score")}
    return mp


def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _clean_quote_for_match(s: str) -> str:
    """
    Keep only the quoted text content.
    Some outputs include appended reference fragments like:
      ... "quote text" leviathan_...txt, Ch. 13, §5
    """
    q = (s or "").strip()
    if not q:
        return q

    # Remove trailing inline filename + chapter marker if present.
    q = re.sub(
        r"\s+[a-z0-9][a-z0-9._-]*\.(?:txt|md|pdf)\s*,?\s*(?:Ch\.|Chapter)\s*\d+.*$",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()

    # Remove trailing plain chapter marker if present.
    q = re.sub(
        r"\s+(?:[A-Za-z][\w\s\-]{0,80},\s*)?(?:Ch\.|Chapter)\s*\d+.*$",
        "",
        q,
        flags=re.IGNORECASE,
    ).strip()

    # Drop surrounding matching quotes.
    if len(q) >= 2 and q[0] in {'"', "'"} and q[-1] == q[0]:
        q = q[1:-1].strip()

    return q


def _read_text_best_effort(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None


def _extract_final_quotes_with_refs(trace: Dict[str, Any]) -> List[Dict[str, str]]:
    events = (trace or {}).get("events") or []
    for ev in reversed(events):
        if ev.get("event") == "quotes_selected":
            quotes = ev.get("quotes") or []
            return [q for q in quotes if isinstance(q, dict)]
    return []


def _selected_filename_map(selection: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    out: Dict[Tuple[int, int], str] = {}
    pool = (selection or {}).get("pool") or []
    for it in pool:
        if not isinstance(it, dict):
            continue
        if str(it.get("selected", "")).upper() != "Y":
            continue
        ch = it.get("chapter")
        sec = it.get("section")
        fn = it.get("filename")
        if isinstance(ch, int) and isinstance(sec, int) and isinstance(fn, str) and fn:
            out[(ch, sec)] = fn
    return out


def quote_match_metrics(
    *,
    trace: Dict[str, Any],
    selection: Dict[str, Any],
    data_dir: str,
) -> Dict[str, Any]:
    """
    Checks each final selected quote against its local source file text.
    """
    filename_map = _selected_filename_map(selection)
    quotes = _extract_final_quotes_with_refs(trace)

    checked = 0
    matched = 0
    missing_source = 0

    for q in quotes:
        quote = _clean_quote_for_match(q.get("quote") or "")
        ref = q.get("ref") or ""
        if not quote:
            continue

        m = _REF_CH_SEC_RE.search(ref)
        if not m:
            continue
        chsec = (int(m.group(1)), int(m.group(2)))
        filename = filename_map.get(chsec)
        if not filename:
            missing_source += 1
            continue

        text = _read_text_best_effort(Path(data_dir) / filename)
        if text is None:
            missing_source += 1
            continue

        checked += 1
        if _normalize_for_match(quote) in _normalize_for_match(text):
            matched += 1

    rate = (matched / checked) if checked > 0 else None
    return {
        "quote_match_checked": checked,
        "quote_match_count": matched,
        "quote_match_rate": rate,
        "quote_match_all": (matched == checked) if checked > 0 else None,
        "quote_match_missing_source": missing_source,
    }


def _grading_weighted_mean(graded: Dict[str, Any]) -> tuple[Optional[float], int]:
    """
    Returns (mean, count) from a grader payload:
      {"per_item":[{"score":...}, ...], "mean": ...}
    """
    per_item = graded.get("per_item", []) if isinstance(graded, dict) else []
    vals: List[float] = []
    for it in per_item:
        if not isinstance(it, dict):
            continue
        sc = it.get("score")
        if isinstance(sc, (int, float)):
            vals.append(float(sc))
    if not vals:
        return None, 0
    return (sum(vals) / len(vals)), len(vals)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# ----------------------------
# Metrics (per-test only)
# ----------------------------

def compute_all_metrics(
    *,
    final_answer: str,
    trace: Dict[str, Any],
    selection: Dict[str, Any],
    must_refuse: bool,
    expected_quote_contains: str = "",
    expected: Optional[List[Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Returns a flat dict of metrics for THIS test only.
    Includes:
      - base metrics (refusal, audit, expected_quote_contains hit)
      - expected-based retrieval/selection metrics if expected provided
    """
    metrics: Dict[str, Any] = {}

    # A) Base metrics
    metrics.update(
        compute_metrics(
            final_answer=final_answer,
            trace=trace,
            must_refuse=must_refuse,
            expected_quote_contains=expected_quote_contains,
        )
    )

    # B) Expected-based metrics (optional)
    exp_set = _expected_set(expected or [])
    if not exp_set:
        return metrics

    pool_map = _retrieval_pool_map_from_selection(selection)
    pool_keys = set(pool_map.keys())

    retrieved_hits = sorted(exp_set.intersection(pool_keys))
    retrieval_recall_at_20 = (len(retrieved_hits) / len(exp_set)) if exp_set else None

    ranks: List[float] = []
    scores: List[float] = []
    for k in retrieved_hits:
        r = pool_map.get(k, {}).get("rank")
        s = pool_map.get(k, {}).get("score")
        if isinstance(r, int):
            ranks.append(float(r))
        sf = _safe_float(s)
        if sf is not None:
            scores.append(sf)

    final_set = _final_citations_set_from_trace(trace)
    selected_hits = sorted(exp_set.intersection(final_set))
    final_recall = (len(selected_hits) / len(exp_set)) if exp_set else None
    extra_selected_count = len(final_set) - len(selected_hits)

    metrics.update(
        {
            "expected_total": len(exp_set),
            "expected_retrieved": len(retrieved_hits),
            "retrieval_recall_at_20": retrieval_recall_at_20,
            "retrieval_mean_rank": _mean(ranks),
            "retrieval_mean_score": _mean(scores),
            "expected_selected": len(selected_hits),
            "final_recall": final_recall,
            "final_unique_citations": len(final_set),
            "extra_selected_count": extra_selected_count,
        }
    )

    return metrics


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a book test suite against the Tender pipeline.")
    p.add_argument(
        "--book-id",
        default=default_corpus_id(),
        help="Book/corpus id to evaluate (default: registry default)",
    )
    p.add_argument(
        "--suite",
        default=None,
        help="Suite name under books/<book_id>/tests/ (for example: work, spelling, blind)",
    )
    p.add_argument(
        "--test-file",
        default=None,
        help="Optional explicit path to a suite JSON file",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Optional local text dir override passed to run_pipeline_with_policy",
    )
    p.add_argument(
        "--out-root",
        default="runs",
        help="Root folder for runs. Outputs are stored under <out_root>/<book_id>/suite/<timestamp>_<suite_id>/",
    )
    p.add_argument("--score-low", type=float, default=0.20,
                   help="BookScore B1 score_low param (default: 0.20)")
    p.add_argument("--score-high", type=float, default=0.80,
                   help="BookScore B1 score_high param (default: 0.80)")
    return p.parse_args()


# ----------------------------
# Main (runner only)
# ----------------------------

def _extract_bookscore_metrics(out: Dict[str, Any]) -> Dict[str, Any]:
    """Extract bookscore fields from run_pipeline_with_policy() output."""
    bs = out.get("bookscore") or {}
    audit_passed = bool((out.get("trace") or {}).get("meta", {}).get("audit_passed"))
    return {
        "bookscore": bs.get("bookscore"),
        "bookscore_zone": bs.get("zone"),
        "B1": bs.get("B1"),
        "did_answer": audit_passed,
        "did_refuse": not audit_passed,
    }


def main() -> None:
    args = parse_args()
    _load_env(ROOT / ".env")

    suite_path = resolve_suite_path(
        book_id=args.book_id,
        suite=args.suite,
        test_file=args.test_file,
    )
    suite = load_suite(suite_path)
    corpus = get_corpus(args.book_id)
    effective_data_dir = args.data_dir or corpus.data_dir

    pipeline_ver = get_git_commit() or "no_git"
    suite_id = suite.get("suite_id", suite_path.stem)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        Path(args.out_root)
        / safe_slug(args.book_id)
        / "suite"
        / f"{ts}_{safe_slug(suite_id)}"
    )
    ensure_dir(run_dir)

    config: Dict[str, Any] = {
        "timestamp": ts,
        "run_type": "suite",
        "pipeline_version": pipeline_ver,
        "suite_id": suite_id,
        "mode": suite.get("mode"),
        "suite_path": str(suite_path),
        "n_tests": len(suite["tests"]),
        "book_id": args.book_id,
        "corpus_id": args.book_id,
        "data_dir": effective_data_dir,
        "bookscore_params": {
            "score_low": args.score_low,
            "score_high": args.score_high,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    client = OpenAI()

    for test in suite["tests"]:
        test_id = test.get("id", "unknown")
        question = test.get("question", "") or ""
        must_refuse = bool(test.get("must_refuse", False))
        expected_quote_contains = test.get("expected_quote_contains", "") or ""
        expected = test.get("expected", []) or []

        out = run_pipeline_with_policy(
            question,
            corpus_id=args.book_id,
            data_dir=effective_data_dir,
            bookscore_score_low=args.score_low,
            bookscore_score_high=args.score_high,
        )

        final_answer = out.get("final_answer", "")
        trace = out.get("trace", {}) or {}
        selection = out.get("selection", {}) or {}

        # A) grade extras + expected quotes
        extras = extract_extras_for_grading(question=question, trace=trace, expected=expected)
        expected_quotes = extract_expected_quotes_for_grading(question=question, trace=trace, expected=expected)

        graded_extras = grade_quotes_relevance(client, question, extras)
        graded_expected = grade_quotes_relevance(client, question, expected_quotes)

        # B) per-test metrics
        metrics = compute_all_metrics(
            final_answer=final_answer,
            trace=trace,
            selection=selection,
            must_refuse=must_refuse,
            expected_quote_contains=expected_quote_contains,
            expected=expected,
        )

        # C) grading metrics
        metrics["extras_mean_relevance"] = graded_extras.get("mean")
        metrics["extras_count_graded"] = len(graded_extras.get("per_item", []))
        metrics["expected_quotes_mean_relevance"] = graded_expected.get("mean")
        metrics["expected_quotes_count_graded"] = len(graded_expected.get("per_item", []))

        extras_mean, extras_count = _grading_weighted_mean(graded_extras)
        expected_mean, expected_count = _grading_weighted_mean(graded_expected)
        total_count = extras_count + expected_count
        weighted = (
            ((extras_mean or 0.0) * extras_count + (expected_mean or 0.0) * expected_count) / total_count
            if total_count > 0 else None
        )
        metrics["answers_mean_relevance"] = weighted
        metrics["answers_count_graded"] = total_count
        metrics.update(
            quote_match_metrics(trace=trace, selection=selection, data_dir=effective_data_dir)
        )

        # D) bookscore metrics
        metrics.update(_extract_bookscore_metrics(out))

        # E) grounded citation count from trace
        events = trace.get("events", [])
        for ev in reversed(events):
            if ev.get("event") == "answer_generated":
                metrics["grounded_citation_count"] = ev.get("grounded_citations", 0)
                break

        # F) build row
        row: Dict[str, Any] = {
            "id": test_id,
            "question": question,
            "must_refuse": must_refuse,
            "expected_quote_contains": expected_quote_contains,
            "expected": expected,
            "final_answer": final_answer,
            "metrics": metrics,
            "selection": selection,
            "trace": trace,
            "pipeline_version": pipeline_ver,
            "suite_id": suite_id,
            "timestamp": ts,
            "book_id": args.book_id,
            "corpus_id": args.book_id,
        }
        row["extras_grading"] = graded_extras
        row["expected_quotes_grading"] = graded_expected

        # G) bookscore + glossary_match section
        row["bookscore"] = out.get("bookscore") or {}
        row["glossary_match"] = out.get("glossary_match") or {}

        rows.append(row)

    # Persist run outputs (runner responsibility)
    write_jsonl(run_dir / "results.jsonl", rows)
    render_html_report(rows, run_dir / "report.html", config=config)
    summary = {
        "timestamp": ts,
        "run_type": "suite",
        "book_id": args.book_id,
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "pipeline_version": pipeline_ver,
        "n_tests": len(rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote suite run to: {run_dir}")
    print(f"   - {run_dir / 'config.json'}")
    print(f"   - {run_dir / 'results.jsonl'}")
    print(f"   - {run_dir / 'report.html'}")
    print(f"   - {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
