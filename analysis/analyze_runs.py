import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


RUNS_DIR = Path("eval_runs")

FOCUS_METRICS = [
    "m__retrieval_recall_at_20",
    "m__retrieval_mean_rank",
    "m__retrieval_mean_score",
    "m__selected_pool_scores",
    "m__quote_match_rate",
    "m__final_recall",
    "m__extra_selected_count",
    "m__answers_mean_relevance",
    "m__extras_mean_relevance",
    "m__expected_quotes_mean_relevance",
]

# Formatting + "better direction"
METRIC_CFG: Dict[str, Dict[str, Any]] = {
    "m__retrieval_recall_at_20": {"label": "retrieval_recall_at_20", "fmt": "pct0", "better": "up"},
    "m__retrieval_mean_rank": {"label": "retrieval_mean_rank", "fmt": "f1", "better": "down"},
    "m__retrieval_mean_score": {"label": "retrieval_mean_score", "fmt": "f2", "better": "up"},
    "m__selected_pool_scores": {"label": "selected_pool_scores", "fmt": "text", "better": "neutral"},
    "m__quote_match_rate": {"label": "quote_match_rate", "fmt": "pct0", "better": "up"},
    "m__final_recall": {"label": "final_recall", "fmt": "pct0", "better": "up"},
    "m__extra_selected_count": {"label": "extra_selected_count", "fmt": "f1", "better": "down"},
    "m__answers_mean_relevance": {"label": "answers_mean_relevance", "fmt": "f1", "better": "up"},
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
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs: List[Path] = []
    for cfg in sorted(runs_dir.rglob("config.json")):
        d = cfg.parent
        if (d / "results.jsonl").exists():
            runs.append(d)
    return runs


def selected_pool_scores_joined(row: Dict[str, Any]) -> str:
    selection = row.get("selection") or {}
    pool = selection.get("pool") or []
    scores: List[str] = []

    for item in pool:
        if not isinstance(item, dict):
            continue
        if str(item.get("selected", "")).upper() != "Y":
            continue
        score = item.get("score")
        if isinstance(score, (int, float)):
            scores.append(f"{float(score):.6f}")

    return " - ".join(scores)


def flatten_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {f"m__{k}": v for k, v in (row.get("metrics") or {}).items()}
    out["m__selected_pool_scores"] = selected_pool_scores_joined(row)
    return out


# =============================
# DataFrames
# =============================

def load_all_runs(runs_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rd in find_runs(runs_dir):
        cfg = read_json(rd / "config.json")
        run_relpath = str(rd.relative_to(runs_dir))
        for r in read_jsonl(rd / "results.jsonl"):
            base = {
                "run_dir": run_relpath,
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
        if c.startswith("m__") and METRIC_CFG.get(c, {}).get("fmt") != "text":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values(by=["timestamp", "run_dir"], ascending=[False, False])


def summarize_runs(df_tests: pd.DataFrame) -> pd.DataFrame:
    if df_tests.empty:
        return df_tests

    metrics = [c for c in FOCUS_METRICS if c in df_tests.columns]
    gcols = ["timestamp", "run_dir", "pipeline_version", "suite_id", "mode", "n_tests"]
    agg_map = {c: ("first" if METRIC_CFG.get(c, {}).get("fmt") == "text" else "mean") for c in metrics}

    return (
        df_tests
        .groupby(gcols, dropna=False, as_index=False)
        .agg(agg_map)
        .sort_values(by=["timestamp", "run_dir"], ascending=[False, False])
    )


# =============================
# Formatting helpers
# =============================

def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _fmt_decimal_comma(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}".replace(".", ",")


def format_metric_value(metric: str, val: Any) -> str:
    if _is_nan(val):
        return ""

    kind = METRIC_CFG.get(metric, {}).get("fmt")
    if kind == "text":
        return str(val)

    v = float(val)

    if kind == "pct0":
        return f"{int(round(v * 100))}%"
    if kind == "f1":
        return _fmt_decimal_comma(v, 1)
    if kind == "f2":
        return _fmt_decimal_comma(v, 2)
    return str(v)


def classify_change(metric: str, prev: Any, cur: Any) -> str:
    kind = METRIC_CFG.get(metric, {}).get("fmt")
    if kind == "text":
        return "neutral"

    if _is_nan(prev) or _is_nan(cur):
        return "neutral"

    dp = float(cur) - float(prev)
    if dp == 0:
        return "neutral"

    better = METRIC_CFG.get(metric, {}).get("better", "up")
    return "good" if (dp > 0 if better == "up" else dp < 0) else "bad"


def escape_html(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run_report_link(run_relpath: Any) -> str:
    if not run_relpath:
        return ""
    rd = escape_html(run_relpath)
    return f'<a href="{rd}/report.html" target="_blank">open</a>'


# =============================
# HTML blocks
# =============================

def metrics_table_block(
    title_left: str,
    title_right: str,
    df_rows: pd.DataFrame,
    metric_cols: Sequence[str],
) -> str:
    d = df_rows.sort_values("timestamp").reset_index(drop=True)

    headers = (
        ["timestamp", "run_dir", "run_report"]
        + [METRIC_CFG[m]["label"] for m in metric_cols]
    )

    body: List[str] = []

    for i, row in d.iterrows():
        prev = d.iloc[i - 1] if i > 0 else None

        cells = [
            f"<td>{escape_html(row['timestamp'])}</td>",
            f"<td><code>{escape_html(row['run_dir'])}</code></td>",
            f"<td class='link'>{run_report_link(row['run_dir'])}</td>",
        ]

        for m in metric_cols:
            cls = classify_change(m, prev[m] if prev is not None else None, row[m])
            txt = format_metric_value(m, row[m])
            cells.append(f"<td class='num {cls}'>{escape_html(txt)}</td>")

        body.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        "<div class='qtablewrap'>"
        "<table class='qtable'>"
        "<thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead>"
        "<tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )

    return f"""
<details class="qblock">
  <summary>
    <span class="qid">{escape_html(title_left)}</span>
    <span class="qtext">{escape_html(title_right)}</span>
  </summary>
  <div class="qcontent">{table}</div>
</details>
"""


# =============================
# HTML report
# =============================

def render_report(
    out_path: Path,
    df_runs: pd.DataFrame,
    df_tests: pd.DataFrame,
    suite_filter: Optional[str],
    mode_filter: Optional[str],
) -> None:
    blocks: List[str] = []

    summary_metrics = [c for c in FOCUS_METRICS if c in df_runs.columns]
    if not df_runs.empty:
        blocks.append(
            metrics_table_block(
                "Summary",
                "Run-level means",
                df_runs[["timestamp", "run_dir"] + summary_metrics],
                summary_metrics,
            )
        )

    if not df_tests.empty and "test_id" in df_tests.columns:
        metric_cols = [c for c in FOCUS_METRICS if c in df_tests.columns]
        for test_id in sorted(df_tests["test_id"].dropna().unique()):
            df_q = df_tests[df_tests["test_id"] == test_id]
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

  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 13px; }}
  th {{ background: #f7f7f7; position: sticky; top: 0; }}

  .qblock {{ border: 1px solid #e6e6e6; border-radius: 12px; margin: 10px 0; padding: 10px; }}
  .qblock summary {{ cursor: pointer; display: flex; gap: 10px; }}
  .qid {{ font-weight: 700; white-space: nowrap; }}
  .qtext {{ color: #333; }}

  .qtablewrap {{ overflow-x: auto; }}
  .qtable th, .qtable td {{ width: 10%; }} /* uniform column width */

  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .good {{ background: #e9f7ef; }}
  .bad  {{ background: #fdecea; }}

  .link {{ text-align: center; }}
  .link a {{ color: #2563eb; text-decoration: none; font-weight: 600; }}
  .link a:hover {{ text-decoration: underline; }}

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
    ap.add_argument("--out")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out) if args.out else runs_dir / "report.html"

    df_tests = load_all_runs(runs_dir)
    if df_tests.empty:
        render_report(out_path, pd.DataFrame(), pd.DataFrame(), args.suite, args.mode)
        print(f"✅ Wrote analysis report: {out_path}")
        return

    if args.suite:
        df_tests = df_tests[df_tests["suite_id"] == args.suite]
    if args.mode:
        df_tests = df_tests[df_tests["mode"] == args.mode]

    df_runs = summarize_runs(df_tests)

    render_report(out_path, df_runs, df_tests, args.suite, args.mode)
    print(f"✅ Wrote analysis report: {out_path}")


if __name__ == "__main__":
    main()
