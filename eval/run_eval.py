#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from tender.engine.pipeline import run_pipeline
from eval.grader import grade_quotes_relevance
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
# Suite loading
# ----------------------------

def load_suite(path: Path) -> Dict[str, Any]:
    try:
        suite = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path} at line {e.lineno}, col {e.colno}: {e.msg}") from e

    if "tests" not in suite or not isinstance(suite["tests"], list):
        raise ValueError("Suite JSON must contain a top-level 'tests' list.")
    if "id" not in suite or not isinstance(suite["id"], str):
        suite["id"] = path.stem
    return suite


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluation suite against Tender pipeline.")
    p.add_argument(
        "suite",
        nargs="?",
        default="eval/tests/definitions_v0.json",
        help="Path to suite JSON (default: eval/tests/definitions_v0.json)",
    )
    p.add_argument(
        "--corpus",
        default="hobbes",
        help="Corpus id to evaluate (default: hobbes)",
    )
    p.add_argument(
        "--mode",
        default=None,
        help="Pipeline mode override (default: suite.mode or 'short')",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Local text dir passed to run_pipeline (default: data)",
    )
    p.add_argument(
        "--out-root",
        default="eval_runs",
        help="Root folder for runs (default: eval_runs)",
    )
    return p.parse_args()


# ----------------------------
# Main (runner only)
# ----------------------------

def main() -> None:
    args = parse_args()

    suite_path = Path(args.suite)
    suite = load_suite(suite_path)

    pipeline_ver = get_git_commit() or "no_git"
    suite_id = suite.get("id", suite_path.stem)

    mode = args.mode or suite.get("mode", "short")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        Path(args.out_root)
        / f"{ts}__{safe_slug(pipeline_ver)}__{safe_slug(suite_id)}__{safe_slug(args.corpus)}"
    )
    ensure_dir(run_dir)

    config = {
        "timestamp": ts,
        "pipeline_version": pipeline_ver,
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "n_tests": len(suite["tests"]),
        "mode": mode,
        "corpus_id": args.corpus,
        "data_dir": args.data_dir,
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

        out = run_pipeline(
            question,
            mode=mode,
            corpus_id=args.corpus,     # ✅ multi-corpus
            data_dir=args.data_dir,    # ✅ align with selection view
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

        # C) add grading metrics (per test)
        metrics["extras_mean_relevance"] = graded_extras.get("mean")
        metrics["extras_count_graded"] = len(graded_extras.get("per_item", []))

        metrics["expected_quotes_mean_relevance"] = graded_expected.get("mean")
        metrics["expected_quotes_count_graded"] = len(graded_expected.get("per_item", []))

        # D) build row
        row = {
            "id": test_id,
            "question": question,
            "must_refuse": must_refuse,
            "expected_quote_contains": expected_quote_contains,
            "expected": expected,
            "final_answer": final_answer,
            "metrics": metrics,
            "selection": selection,
            "trace": trace,
            # run metadata copied into each row (makes analysis easier)
            "pipeline_version": pipeline_ver,
            "suite_id": suite_id,
            "mode": mode,
            "timestamp": ts,
            "corpus_id": args.corpus,
        }

        row["extras_grading"] = graded_extras
        row["expected_quotes_grading"] = graded_expected

        rows.append(row)

    # Persist run outputs (runner responsibility)
    write_jsonl(run_dir / "results.jsonl", rows)
    render_html_report(rows, run_dir / "report.html", config=config)

    print(f"✅ Wrote eval run to: {run_dir}")
    print(f"   - {run_dir / 'config.json'}")
    print(f"   - {run_dir / 'results.jsonl'}")
    print(f"   - {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
