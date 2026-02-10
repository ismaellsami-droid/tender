import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


RUNS_DIR = Path("eval_runs")

FOCUS_METRICS = [
    "m__retrieval_recall_at_20",
    "m__retrieval_mean_rank",
    "m__retrieval_mean_score",
    "m__final_recall",
    "m__extra_selected_count",
    "m__extras_mean_relevance",
    "m__expected_quotes_mean_relevance",
]

# Metric formatting + "better direction"
# better: "up" -> higher is better; "down" -> lower is better
METRIC_CFG: Dict[str, Dict[str, Any]] = {
    "m__retrieval_recall_at_20": {"label": "retrieval_recall_at_20", "fmt": "pct0", "better": "up"},
    "m__retrieval_mean_rank": {"label": "retrieval_mean_rank", "fmt": "f1", "better": "down"},
    "m__retrieval_mean_score": {"label": "retrieval_mean_score", "fmt": "f2", "better": "up"},
    "m__final_recall": {"label": "final_recall", "fmt": "pct0", "better": "up"},
    "m__extra_selected_count": {"label": "extra_selected_count", "fmt": "f1", "better": "down"},
    "m__extras_mean_relevance": {"label": "extras_mean_relevance", "fmt": "f1", "better": "up"},
    "m__expected_quotes_mean_relevance": {"label": "expected_quotes_mean_relevance", "fmt": "f1", "better": "up"},
}


# =============================
# Loading helpers
# =============================

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return [
        d for d in sorted(runs_dir.iterdir())
        if d.is_dir()
        and (d / "config.json").exists()
        and (d / "results.jsonl").exists()
    ]


def flatten_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {f"m__{k}": v for k, v in (row.get("metrics") or {}).items()}


# =============================
# DataFrames
# =============================

def load_all_runs(runs_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rd in find_runs(runs_dir):
        cfg = read_json(rd / "config.json")
        results = read_jsonl(rd / "results.jsonl")

        for r in results:
            base = {
                "run_dir": rd.name,
                "timestamp": cfg.get("timestamp"),
                "pipeline_version": cfg.get("pipeline_version"),
                "suite_id": cfg.get("suite_id"),
                "mode": cfg.get("mode"),
                "n_tests": cfg.get("n_tests"),
                "test_id": r.get("id"),
                "question": r.get("question"),
            }
            base.update(flatten_metrics(r))
            rows.append(base)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for c in df.columns:
        if c.startswith("m__"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(by=["timestamp", "run_dir"], ascending=[False, False])


def summarize_runs(df_tests: pd.DataFrame) -> pd.DataFrame:
    if df_tests.empty:
        return df_tests

    metrics = [c for c in FOCUS_METRICS if c in df_tests.columns]
    gcols = ["timestamp", "run_dir", "pipeline_version", "suite_id", "mode", "n_tests"]

    df = (
        df_tests
        .groupby(gcols, dropna=False, as_index=False)
        .agg({c: "mean" for c in metrics})
        .sort_values(by=["timestamp", "run_dir"], ascending=[False, False])
    )
    return df


def compare_runs(df_tests: pd.DataFrame, run_a: str, run_b: str) -> pd.DataFrame:
    """
    Simple compare table (no special coloring here; it’s an auxiliary view).
    """
    a = df_tests[df_tests["run_dir"] == run_a]
    b = df_tests[df_tests["run_dir"] == run_b]

    keep = ["test_id", "question"] + [c for c in FOCUS_METRICS if c in df_tests.columns]

    a2 = a[keep].rename(columns={c: f"{c}__A" for c in keep if c not in ("test_id", "question")})
    b2 = b[keep].rename(columns={c: f"{c}__B" for c in keep if c not in ("test_id", "question")})

    df = a2.merge(b2, on=["test_id", "question"], how="outer")

    for c in FOCUS_METRICS:
        ca, cb = f"{c}__A", f"{c}__B"
        if ca in df.columns and cb in df.columns:
            df[f"delta__{c}"] = df[ca] - df[cb]

    return df.sort_values("test_id")


# =============================
# Formatting + coloring
# =============================

def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _fmt_decimal_comma(x: float, decimals: int) -> str:
    s = f"{x:.{decimals}f}"
    return s.replace(".", ",")


def format_metric_value(metric: str, val: Any) -> str:
    if _is_nan(val):
        return ""

    cfg = METRIC_CFG.get(metric, {"fmt": "raw"})
    kind = cfg.get("fmt", "raw")

    try:
        v = float(val)
    except Exception:
        return str(val)

    if kind == "pct0":
        # assume val is in [0,1] (recall); if it’s already [0,100], you can adjust here
        pct = round(v * 100.0)
        return f"{int(pct)}%"
    if kind == "f1":
        return _fmt_decimal_comma(v, 1)
    if kind == "f2":
        return _fmt_decimal_comma(v, 2)

    return str(val)


def classify_change(metric: str, prev: Any, cur: Any) -> str:
    """
    Returns: "good" | "bad" | "neutral"
    """
    if _is_nan(prev) or _is_nan(cur):
        return "neutral"

    try:
        dp = float(cur) - float(prev)
    except Exception:
        return "neutral"

    if dp == 0:
        return "neutral"

    better = METRIC_CFG.get(metric, {}).get("better", "up")
    if better == "up":
        return "good" if dp > 0 else "bad"
    else:
        return "good" if dp < 0 else "bad"


def escape_html(s: Any) -> str:
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# =============================
# HTML blocks (tables-only)
# =============================

def metrics_table_block(
    title_left: str,
    title_right: str,
    df_rows: pd.DataFrame,
    metric_cols: Sequence[str],
) -> str:
    """
    Table with:
      timestamp | run_dir | metrics...
    Each metric cell colored based on change vs previous run (chronological).
    No delta columns.
    """
    d = df_rows.copy()
    if d.empty:
        return f"""
<details class="qblock">
  <summary><span class="qid">{escape_html(title_left)}</span><span class="qtext">{escape_html(title_right)}</span></summary>
  <div class="qcontent"><p><em>No data.</em></p></div>
</details>
"""

    # chronological for diff
    d = d.sort_values(by="timestamp", ascending=True).reset_index(drop=True)

    # build header
    headers = ["timestamp", "run_dir"] + [METRIC_CFG.get(m, {}).get("label", m) for m in metric_cols]

    # build body
    body_rows: List[str] = []
    for i in range(len(d)):
        row = d.iloc[i]
        prev = d.iloc[i - 1] if i > 0 else None

        t = escape_html(row.get("timestamp"))
        rd = escape_html(row.get("run_dir"))

        cells = [f"<td>{t}</td>", f"<td><code>{rd}</code></td>"]

        for m in metric_cols:
            cur_val = row.get(m)
            prev_val = prev.get(m) if prev is not None else None
            cls = classify_change(m, prev_val, cur_val)
            txt = format_metric_value(m, cur_val)
            cells.append(f"<td class='num {cls}'>{escape_html(txt)}</td>")

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        "<div class='qtablewrap'>"
        "<table class='qtable'>"
        "<thead><tr>"
        + "".join([f"<th>{escape_html(h)}</th>" for h in headers])
        + "</tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )

    return f"""
<details class="qblock">
  <summary>
    <span class="qid">{escape_html(title_left)}</span>
    <span class="qtext">{escape_html(title_right)}</span>
  </summary>
  <div class="qcontent">
    {table}
  </div>
</details>
"""


def simple_table_block(title: str, df: pd.DataFrame, max_rows: int = 500) -> str:
    if df is None or df.empty:
        return f"<h4>{escape_html(title)}</h4><p><em>No data.</em></p>"
    return f"<h4>{escape_html(title)}</h4>" + df.head(max_rows).to_html(index=False, escape=True)


# =============================
# HTML report
# =============================

def render_report(
    out_path: Path,
    df_runs: pd.DataFrame,
    df_tests: pd.DataFrame,
    df_compare: Optional[pd.DataFrame],
    suite_filter: Optional[str],
    mode_filter: Optional[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = []

    # SUMMARY
    summary_metrics = [c for c in FOCUS_METRICS if c in df_runs.columns]
    if not df_runs.empty and summary_metrics:
        blocks.append(
            metrics_table_block(
                "Summary",
                "Run-level means",
                df_runs[["timestamp", "run_dir"] + summary_metrics],
                summary_metrics,
            )
        )
    else:
        blocks.append("<p><em>No summary data.</em></p>")

    # COMPARE (optional)
    if df_compare is not None:
        blocks.append(simple_table_block("Run comparison (per test)", df_compare))

    # QUESTIONS
    metric_cols = [c for c in FOCUS_METRICS if c in df_tests.columns]
    if not df_tests.empty and metric_cols:
        # Keep only needed cols once
        keep_cols = ["timestamp", "run_dir", "test_id", "question"] + metric_cols
        dft = df_tests[keep_cols].copy()

        for test_id in sorted(dft["test_id"].dropna().unique()):
            df_q = dft[dft["test_id"] == test_id].copy()
            if df_q.empty:
                continue
            q = df_q["question"].iloc[0] or ""
            blocks.append(
                metrics_table_block(
                    f"Test {test_id}",
                    q,
                    df_q[["timestamp", "run_dir"] + metric_cols],
                    metric_cols,
                )
            )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Eval analysis report</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
  th {{ background: #f7f7f7; position: sticky; top: 0; z-index: 2; }}
  tr:nth-child(even) {{ background: #fcfcfc; }}

  .qblock {{ border: 1px solid #e6e6e6; border-radius: 12px; padding: 10px 12px; margin: 10px 0; background: #fff; }}
  .qblock > summary {{ cursor: pointer; list-style: none; display:flex; gap:10px; align-items: baseline; }}
  .qblock > summary::-webkit-details-marker {{ display:none; }}
  .qid {{ font-weight: 700; white-space: nowrap; }}
  .qtext {{ color: #333; }}

  .qcontent {{ margin-top: 10px; }}
  .qtablewrap {{ overflow-x: auto; border-radius: 10px; }}
  .qtable td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  /* Pastel feedback colors */
  .good {{ background: #e9f7ef; }} /* pastel green */
  .bad  {{ background: #fdecea; }} /* pastel red */

  code {{ background: #f6f6f6; padding: 1px 4px; border-radius: 6px; }}
</style>
</head>
<body>
  <h2>Eval analysis report</h2>
  <p><strong>Filters</strong>: suite_id={escape_html(suite_filter or "ALL")}, mode={escape_html(mode_filter or "ALL")}</p>
  {''.join(blocks)}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="eval_runs")
    ap.add_argument("--suite")
    ap.add_argument("--mode")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--out", default="analysis/report.html")
    args = ap.parse_args()

    df_tests = load_all_runs(Path(args.runs_dir))
    if df_tests.empty:
        print("No runs found.")
        return

    if args.suite:
        df_tests = df_tests[df_tests["suite_id"] == args.suite]
    if args.mode:
        df_tests = df_tests[df_tests["mode"] == args.mode]

    df_runs = summarize_runs(df_tests)
    df_compare = compare_runs(df_tests, *args.compare) if args.compare else None

    out_path = Path(args.out)
    render_report(out_path, df_runs, df_tests, df_compare, args.suite, args.mode)

    print(f"✅ Wrote analysis report: {out_path}")


if __name__ == "__main__":
    main()
