import json
import re
import subprocess
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:80]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return None


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_html_report(results, out_path: Path, config: dict):
    cards = []
    for r in results:
        status = "✅" if r["metrics"]["pass_all"] else "❌"
        cards.append(f"""
        <div style="border:1px solid #ddd; padding:12px; margin:12px 0; border-radius:8px;">
          <div><b>{status} {html_escape(r["id"])}</b></div>
          <div style="margin-top:6px;"><b>Q:</b> {html_escape(r["question"])}</div>
          <div style="margin-top:6px;"><b>Answer:</b>
            <pre style="white-space:pre-wrap;">{html_escape(r["final_answer"])}</pre>
          </div>
          <details open>
            <summary><b>Selection</b></summary>
            <pre>{html_escape(json.dumps(r.get("selection", {}), indent=2, ensure_ascii=False)[:200000])}</pre>
          </details>
          <details>
            <summary>Metrics</summary>
            <pre>{html_escape(json.dumps(r["metrics"], indent=2, ensure_ascii=False))}</pre>
          </details>
          <details>
            <summary>Trace</summary>
            <pre>{html_escape(json.dumps(r["trace"], indent=2, ensure_ascii=False)[:20000])}</pre>
          </details>
        </div>
        """)

    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Eval report</title>
      </head>
      <body style="font-family:Arial, sans-serif; max-width:1100px; margin:20px auto;">
        <h2>Eval report</h2>
        <pre>{html_escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre>
        {''.join(cards)}
      </body>
    </html>
    """
    out_path.write_text(html, encoding="utf-8")


def _normalize(s: str) -> str:
    return " ".join((s or "").split())


def compute_metrics(
    *,
    final_answer: str,
    trace: dict,
    must_refuse: bool,
    expected_quote_contains
):
    """
    expected_quote_contains:
      - "" or None
      - string
      - list of strings
    """

    if isinstance(expected_quote_contains, str):
        expected = [expected_quote_contains] if expected_quote_contains else []
    elif isinstance(expected_quote_contains, list):
        expected = expected_quote_contains
    else:
        expected = []

    refusal_text = "Je ne peux pas répondre à partir du corpus actuel."
    refusal = final_answer.strip() == refusal_text
    refusal_ok = (refusal == must_refuse)

    expected_quote_hit = True
    if expected:
        ans_norm = _normalize(final_answer)
        expected_quote_hit = any(_normalize(x) in ans_norm for x in expected)

    audit_passed = bool(trace.get("audit_passed", True))

    pass_all = refusal_ok and expected_quote_hit and audit_passed

    return {
        "refusal": refusal,
        "refusal_ok": refusal_ok,
        "expected_quote_hit": expected_quote_hit,
        "audit_passed": audit_passed,
        "pass_all": pass_all,
    }

import re
from typing import Any, Dict, List, Optional, Set, Tuple

_REF_CH_SEC_RE = re.compile(r"Ch\.\s*(\d{1,2}).*?§\s*(\d{1,2})", re.IGNORECASE)


def _expected_set(expected: List[Dict[str, int]]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for e in expected or []:
        ch = e.get("ch")
        sec = e.get("sec")
        if isinstance(ch, int) and isinstance(sec, int):
            out.add((ch, sec))
    return out


def _chsec_from_ref(ref: str) -> Optional[Tuple[int, int]]:
    if not ref:
        return None
    m = _REF_CH_SEC_RE.search(ref)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def extract_expected_quotes_for_grading(
    *,
    question: str,
    trace: Dict[str, Any],
    expected: List[Dict[str, int]],
) -> List[Dict[str, str]]:
    """
    Returns final citations that ARE in expected (by Ch/§).
    Each item: {quote, ref}
    """
    exp = _expected_set(expected)

    events = (trace or {}).get("events") or []
    final_quotes = []
    for ev in reversed(events):
        if ev.get("event") == "quotes_selected":
            final_quotes = ev.get("quotes") or []
            break

    picked: List[Dict[str, str]] = []
    for q in final_quotes:
        if not isinstance(q, dict):
            continue

        ref = q.get("ref", "") or ""
        quote = q.get("quote", "") or ""

        chsec = _chsec_from_ref(ref)
        is_expected = (chsec in exp) if chsec else False

        if is_expected:
            picked.append({"quote": quote, "ref": ref})

    return picked


def extract_extras_for_grading(
    *,
    question: str,
    trace: Dict[str, Any],
    expected: List[Dict[str, int]],
) -> List[Dict[str, str]]:
    """
    Returns final citations that are NOT in expected (by Ch/§).
    Each item: {quote, ref}
    """
    exp = _expected_set(expected)

    events = (trace or {}).get("events") or []
    final_quotes = []
    for ev in reversed(events):
        if ev.get("event") == "quotes_selected":
            final_quotes = ev.get("quotes") or []
            break

    extras: List[Dict[str, str]] = []
    for q in final_quotes:
        if not isinstance(q, dict):
            continue

        ref = q.get("ref", "") or ""
        quote = q.get("quote", "") or ""

        chsec = _chsec_from_ref(ref)
        is_expected = (chsec in exp) if chsec else False

        if not is_expected:
            extras.append({
                "quote": quote,
                "ref": ref,
            })

    return extras
