import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


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


def _render_glossary_match_html(gm: dict) -> str:
    if not gm:
        return "<em>No glossary match data.</em>"

    keywords = gm.get("keywords", [])
    unmatched = gm.get("unmatched_keywords", [])
    keyword_results = gm.get("keyword_results", [])
    has_matches = gm.get("has_matches", False)

    badge = "🟢" if has_matches else "🔴"

    IMPORTANCE_COLOR = {"High": "#1a7a1a", "Med": "#8a6000", "Low": "#a00000"}

    def _is_graph_row(origin: str) -> bool:
        return " of " in origin

    def _td(content, bold=False, color=None, align="left", italic=False,
            extra_style=""):
        style = f"padding:4px 8px;text-align:{align};{extra_style}"
        if color:
            style += f"color:{color};"
        inner = html_escape(str(content))
        if bold:
            inner = f"<b>{inner}</b>"
        elif italic:
            inner = f"<i>{inner}</i>"
        return f'<td style="{style}">{inner}</td>'

    def _td_empty(extra_style=""):
        return f'<td style="padding:4px 8px;{extra_style}"></td>'

    def _origin_cell(origin: str) -> str:
        if origin == "Close in Glossary":
            inner = f'<b style="color:#1a4a8a;">{html_escape(origin)}</b>'
        else:
            # graph expansion row — indent with ↳
            inner = f'<span style="color:#999;padding-left:16px;">↳ <i>{html_escape(origin)}</i></span>'
        return f'<td style="padding:4px 8px;">{inner}</td>'

    rows = []
    for kr in keyword_results:
        kw = kr.get("keyword", "")
        related = kr.get("related_terms", [])
        if not related:
            continue

        for i, item in enumerate(related):
            term = item.get("term", "")
            sim = item.get("sim")
            origin = item.get("origin", "")
            is_glossary = item.get("is_glossary", False)
            importance = item.get("importance") or ""
            frequency = item.get("frequency")
            strength = item.get("strength")
            link_type = item.get("link_type") or ""
            imp_color = IMPORTANCE_COLOR.get(importance, "#333")
            is_graph = _is_graph_row(origin)
            border = "2px solid #bbb" if i == 0 else "1px solid #eee"
            # graph rows: light lavender tint; non-glossary graph rows: slightly more tinted
            bg = "background:#f5f5fa;" if is_graph else ""

            sim_str = f"{sim:.3f}" if sim is not None else "—"
            freq_str = str(frequency) if frequency is not None else "—"
            strength_str = f"{strength:.3f}" if isinstance(strength, (int, float)) else ("—" if strength is None else str(strength))
            term_style = "color:#aaa;font-size:0.88em;" if is_graph else ""

            rows.append(f"""
            <tr style="border-top:{border};{bg}">
              {_td(kw, bold=True) if i == 0 else _td_empty()}
              {_origin_cell(origin)}
              {_td(term, bold=(is_glossary and not is_graph), italic=is_graph, extra_style=term_style)}
              {_td(sim_str, align="right", italic=is_graph, extra_style=term_style)}
              {_td(importance, color=imp_color if not is_graph else None, italic=is_graph, extra_style=term_style) if importance else _td_empty()}
              {_td(link_type, italic=is_graph, extra_style=term_style) if link_type else _td_empty()}
              {_td(strength_str, align="right", italic=is_graph, extra_style=term_style) if is_graph else _td_empty()}
              {_td(freq_str, align="right", italic=is_graph, extra_style=term_style) if is_glossary else _td_empty()}
            </tr>""")

    table = ""
    if rows:
        table = f"""
        <table style="border-collapse:collapse;width:100%;margin-top:8px;font-size:0.9em;">
          <thead>
            <tr style="background:#f0f0f0;">
              <th style="padding:4px 8px;text-align:left;">Keyword</th>
              <th style="padding:4px 8px;text-align:left;">Origin</th>
              <th style="padding:4px 8px;text-align:left;">Term</th>
              <th style="padding:4px 8px;text-align:right;">Sim</th>
              <th style="padding:4px 8px;text-align:left;">Glossary word important</th>
              <th style="padding:4px 8px;text-align:left;">Link type</th>
              <th style="padding:4px 8px;text-align:right;">Link strength</th>
              <th style="padding:4px 8px;text-align:right;">Freq</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>"""

    unmatched_html = ""
    if unmatched:
        unmatched_html = f"<div style='margin-top:6px;color:#a00;font-size:0.85em;'>Unmatched: {html_escape(', '.join(unmatched))}</div>"

    detail_json = html_escape(json.dumps(keyword_results, indent=2, ensure_ascii=False))
    return f"""
    <div>
      <div>{badge} keywords: <b>{html_escape(', '.join(keywords)) or '(none)'}</b></div>
      {table}
      {unmatched_html}
      <details style="margin-top:6px;">
        <summary style="font-size:0.85em;color:#555;cursor:pointer;">Raw JSON</summary>
        <pre style="font-size:0.8em;">{detail_json}</pre>
      </details>
    </div>"""


def _render_bookscore_html(bs: dict) -> str:
    if not bs:
        return "<em>No bookscore data.</em>"
    zone = bs.get("zone", "?")
    zone_color = {"high": "#1a7a1a", "med": "#8a6000", "low": "#a00000"}.get(zone, "#333")
    score = bs.get("bookscore", 0)
    b1 = bs.get("B1", 0)
    label = bs.get("label", zone)
    return f"""
    <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.9em;margin-top:4px;">
      <div><b>Score</b> <span style="font-size:1.1em;">{score:.3f}</span></div>
      <div><b>Zone</b> <span style="color:{zone_color};font-weight:bold;">{zone.upper()}</span>
           <span style="color:#666;font-size:0.85em;margin-left:4px;">({html_escape(label)})</span></div>
      <div><b>B1</b> {b1:.3f}</div>
    </div>"""


def render_html_report(results, out_path: Path, config: dict):
    cards = []
    for r in results:
        status = "✅" if r["metrics"]["pass_all"] else "❌"
        gm_html = _render_glossary_match_html(r.get("glossary_match", {}))
        bs_html = _render_bookscore_html(r.get("bookscore", {}))

        # Strip bookscore fields from metrics copy to avoid duplication
        metrics_display = {
            k: v for k, v in r["metrics"].items()
            if k not in ("bookscore", "bookscore_zone", "B1")
        }

        cards.append(f"""
        <div style="border:1px solid #ddd; padding:12px; margin:12px 0; border-radius:8px;">
          <div><b>{status} {html_escape(r["id"])}</b></div>
          <div style="margin-top:6px;"><b>Q:</b> {html_escape(r["question"])}</div>
          <div style="margin-top:6px;"><b>Answer:</b>
            <pre style="white-space:pre-wrap;">{html_escape(r["final_answer"])}</pre>
          </div>
          <details>
            <summary><b>BookScore</b></summary>
            {bs_html}
          </details>
          <details>
            <summary><b>Glossary Match</b></summary>
            {gm_html}
          </details>
          <details>
            <summary><b>Selection</b></summary>
            <pre>{html_escape(json.dumps(r.get("selection", {}), indent=2, ensure_ascii=False)[:200000])}</pre>
          </details>
          <details>
            <summary>Metrics</summary>
            <pre>{html_escape(json.dumps(metrics_display, indent=2, ensure_ascii=False))}</pre>
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
