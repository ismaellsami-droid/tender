#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from tender.corpora.registry import default_corpus_id, get_corpus
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


def _normalize_for_match(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = " ".join(s.split())
    return s.lower()


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
        default="eval/tests/hobbes_leviathan_book01/pool_a_work.json",
        help="Path to suite JSON (default: eval/tests/hobbes_leviathan_book01/pool_a_work.json)",
    )
    p.add_argument(
        "--corpus",
        default=default_corpus_id(),
        help="Corpus id to evaluate (default: registry default)",
    )
    p.add_argument(
        "--mode",
        default=None,
        help="Pipeline mode override (default: suite.mode or 'short')",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Optional local text dir override passed to run_pipeline",
    )
    p.add_argument(
        "--out-root",
        default="eval_runs",
        help="Root folder for runs. Runs are stored under <out_root>/<corpus>/<suite_id>/ (default: eval_runs)",
    )
    return p.parse_args()


# ----------------------------
# Main (runner only)
# ----------------------------

def main() -> None:
    args = parse_args()

    suite_path = Path(args.suite)
    suite = load_suite(suite_path)
    corpus = get_corpus(args.corpus)
    effective_data_dir = args.data_dir or corpus.data_dir

    pipeline_ver = get_git_commit() or "no_git"
    suite_id = suite.get("id", suite_path.stem)

    mode = args.mode or suite.get("mode", "short")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_base = Path(args.out_root) / safe_slug(args.corpus) / safe_slug(suite_id)
    run_dir = (
        runs_base
        / f"{ts}__{safe_slug(pipeline_ver)}__{safe_slug(mode)}"
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
        "data_dir": effective_data_dir,
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
            data_dir=effective_data_dir,    # ✅ align with selection view
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

        extras_mean, extras_count = _grading_weighted_mean(graded_extras)
        expected_mean, expected_count = _grading_weighted_mean(graded_expected)
        total_count = extras_count + expected_count
        if total_count > 0:
            weighted = ((extras_mean or 0.0) * extras_count + (expected_mean or 0.0) * expected_count) / total_count
        else:
            weighted = None

        metrics["answers_mean_relevance"] = weighted
        metrics["answers_count_graded"] = total_count
        metrics.update(
            quote_match_metrics(
                trace=trace,
                selection=selection,
                data_dir=effective_data_dir,
            )
        )

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
