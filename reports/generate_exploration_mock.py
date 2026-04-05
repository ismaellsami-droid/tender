#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact reformulation mock from anchor diagnostics data JSON.")
    parser.add_argument("--data", required=True, help="Path to the anchor diagnostics data JSON")
    parser.add_argument("--out", default=None, help="Optional output HTML path")
    return parser.parse_args()


def generate_exploration_mock(data_path: Path, out_path: Path | None = None) -> Path:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    final_out = out_path or data_path.with_name("exploration_sidecar_mock.html")
    final_out.write_text(
        _render_html(payload, f"Reformulation Mock - {data_path.stem}"),
        encoding="utf-8",
    )
    return final_out


def _display_term(term: str) -> str:
    return " ".join(part.capitalize() for part in term.split())


def _render_terms(terms: list[str]) -> str:
    if not terms:
        return "<span class='muted'>No terms sent to the LLM.</span>"
    return "".join(f"<span class='chip'>{html.escape(_display_term(term))}</span>" for term in terms)


def _fmt_score(value: object) -> str:
    if isinstance(value, (int, float)):
        return format(float(value), ".2g")
    return "NA"


def _exploration_derivation_meta(candidate: dict[str, object]) -> tuple[str, bool]:
    sources = {
        str(source).strip().lower()
        for source in (candidate.get("candidate_sources") or [])
        if str(source).strip()
    }
    if "quote_satellite" in sources or "question_satellite" in sources:
        source_anchor = str(candidate.get("source_anchor") or "").strip()
        if source_anchor:
            return f"satellite from {_display_term(source_anchor)}", True
        return "satellite expansion", True
    if "quote_reverse_anchor" in sources or "question_reverse_anchor" in sources:
        source_satellite = str(candidate.get("source_satellite") or "").strip()
        if source_satellite:
            return f"from {_display_term(source_satellite)}", True
        return "reverse anchor", True
    return "", False


def _render_candidate_chips(candidates: list[dict[str, object]]) -> str:
    if not candidates:
        return "<span class='muted'>No terms sent to the LLM.</span>"

    chips: list[str] = []
    for item in candidates:
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        question_score = _fmt_score(item.get("question_score"))
        best_keyword_score = _fmt_score(item.get("best_keyword_score"))
        chips.append(
            "<span class='chip'>"
            f"{html.escape(_display_term(term))}"
            f"<span class='chip-meta'>q {html.escape(question_score)} · k {html.escape(best_keyword_score)}</span>"
            "</span>"
        )
    if not chips:
        return "<span class='muted'>No terms sent to the LLM.</span>"
    return "".join(chips)


def _render_suggested_term_chips(candidates: list[dict[str, object]]) -> str:
    if not candidates:
        return "<span class='muted'>No suggested terms.</span>"
    chips: list[str] = []
    for item in candidates:
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        chips.append(f"<span class='chip'>{html.escape(_display_term(term))}</span>")
    if not chips:
        return "<span class='muted'>No suggested terms.</span>"
    return "".join(chips)


def _render_generated_candidates(items: list[dict[str, object]]) -> str:
    if not items:
        return "<div class='muted'>No generated candidates.</div>"
    blocks: list[str] = []
    for item in items:
        text = str(item.get("text") or "").strip()
        replacements = item.get("replacements") if isinstance(item.get("replacements"), list) else []
        replacement_text = ", ".join(
            f"{str(rep.get('span') or '').strip()} -> {str(rep.get('term') or '').strip()}"
            for rep in replacements
            if isinstance(rep, dict)
        )
        blocks.append(
            "<div class='candidate-row'>"
            f"<div class='candidate-text'>{html.escape(text)}</div>"
            f"<div class='candidate-meta'>{html.escape(replacement_text)}</div>"
            "</div>"
        )
    return "".join(blocks)


def _render_span_matches(items: list[dict[str, object]]) -> str:
    if not items:
        return "<div class='muted'>No span-term matches.</div>"
    blocks: list[str] = []
    for item in items:
        span = str(item.get("span") or "").strip()
        matches = item.get("matches") if isinstance(item.get("matches"), list) else []
        chips = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            term = str(match.get("term") or "").strip()
            cosine = match.get("cosine")
            chips.append(
                "<span class='chip'>"
                f"{html.escape(_display_term(term))}"
                f"<span class='chip-meta'>s {html.escape(_fmt_score(cosine))}</span>"
                "</span>"
            )
        chips_html = "".join(chips) if chips else "<span class='muted'>No matches.</span>"
        blocks.append(
            "<div class='match-block'>"
            f"<div class='match-title'>{html.escape(_display_term(span))}</div>"
            f"<div class='chips'>{chips_html}</div>"
            "</div>"
        )
    return "".join(blocks)


def _render_exploration_span_pools(items: list[dict[str, object]]) -> str:
    if not items:
        return "<div class='muted'>No exploration span pools.</div>"
    blocks: list[str] = []
    for item in items:
        span = str(item.get("span") or "").strip()
        candidates = item.get("candidates") if isinstance(item.get("candidates"), list) else []
        chips = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            term = str(candidate.get("term") or "").strip()
            if not term:
                continue
            score = _fmt_score(candidate.get("score"))
            span_score = _fmt_score(candidate.get("span_score"))
            derivation_text, is_derived = _exploration_derivation_meta(candidate)
            chip_class = "chip chip-derived" if is_derived else "chip"
            chips.append(
                (
                    f"<span class='{chip_class}'>"
                    f"{html.escape(_display_term(term))}"
                    f"<span class='chip-meta'>s {html.escape(span_score)} · f {html.escape(score)}</span>"
                )
                + (
                    f"<span class='chip-origin'>{html.escape(derivation_text)}</span>"
                    if derivation_text
                    else ""
                )
                + "</span>"
            )
        chips_html = "".join(chips) if chips else "<span class='muted'>No candidates.</span>"
        blocks.append(
            "<div class='match-block'>"
            f"<div class='match-title'>{html.escape(_display_term(span))}</div>"
            f"<div class='chips'>{chips_html}</div>"
            "</div>"
        )
    return "".join(blocks)


def _render_exploration_quote_pools(items: list[dict[str, object]]) -> str:
    if not items:
        return "<div class='muted'>No exploration quote pools.</div>"
    blocks: list[str] = []
    for item in items:
        label = str(item.get("span") or "").strip()
        quote_text = str(item.get("quote_text") or "").strip()
        candidates = item.get("candidates") if isinstance(item.get("candidates"), list) else []
        chips = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            term = str(candidate.get("term") or "").strip()
            if not term:
                continue
            score = _fmt_score(candidate.get("score"))
            quote_score = _fmt_score(candidate.get("quote_score"))
            derivation_text, is_derived = _exploration_derivation_meta(candidate)
            chip_class = "chip chip-derived" if is_derived else "chip"
            chips.append(
                (
                    f"<span class='{chip_class}'>"
                    f"{html.escape(_display_term(term))}"
                    f"<span class='chip-meta'>q {html.escape(quote_score)} · f {html.escape(score)}</span>"
                )
                + (
                    f"<span class='chip-origin'>{html.escape(derivation_text)}</span>"
                    if derivation_text
                    else ""
                )
                + "</span>"
            )
        chips_html = "".join(chips) if chips else "<span class='muted'>No candidates.</span>"
        blocks.append(
            "<div class='match-block'>"
            f"<div class='match-title'>{html.escape(label or 'Quote')}</div>"
            f"<div class='debug-subtle'>{html.escape(quote_text or 'No quote text captured.')}</div>"
            f"<div class='chips'>{chips_html}</div>"
            "</div>"
        )
    return "".join(blocks)


def _render_html(payload: dict[str, object], title: str) -> str:
    sections = payload.get("questions", []) if isinstance(payload.get("questions"), list) else []
    cards: list[str] = []
    for section in sections:
        reformulation = section.get("question_reformulation", {}) if isinstance(section.get("question_reformulation"), dict) else {}
        reformulation_status = str(reformulation.get("status") or "completed").strip()
        candidate_items = [
            item
            for item in reformulation.get("candidates", [])
            if isinstance(item, dict) and str(item.get("term", "")).strip()
        ]
        extracted_concepts = [
            str(item).strip()
            for item in reformulation.get("extracted_concepts", [])
            if str(item).strip()
        ]
        proposed_question = str(reformulation.get("proposed_question") or "").strip()
        llm_debug = reformulation.get("llm_debug", {}) if isinstance(reformulation.get("llm_debug"), dict) else {}
        raw_response = str(llm_debug.get("raw_response") or "").strip()
        parsed_question = str(llm_debug.get("parsed_reformulated_question") or "").strip()
        postprocess_status = str(llm_debug.get("postprocess_status") or "").strip()
        postprocess_reason = str(llm_debug.get("postprocess_reason") or "").strip()
        generated_candidates = [
            item for item in reformulation.get("generated_candidates", [])
            if isinstance(item, dict)
        ]
        span_term_matches = [
            item for item in reformulation.get("span_term_matches", [])
            if isinstance(item, dict)
        ]
        exploration = section.get("question_exploration", {}) if isinstance(section.get("question_exploration"), dict) else {}
        exploration_status = str(exploration.get("status") or "completed").strip()
        exploration_source_mode = str(exploration.get("source_mode") or "question").strip()
        exploration_concepts = [
            str(item).strip()
            for item in exploration.get("extracted_concepts", [])
            if str(item).strip()
        ]
        exploration_span_pools = [
            item for item in exploration.get("span_pools", [])
            if isinstance(item, dict)
        ]
        exploration_quote_pools = [
            item for item in exploration.get("quote_pools", [])
            if isinstance(item, dict)
        ]
        exploration_suggested_terms = [
            item for item in exploration.get("suggested_terms", [])
            if isinstance(item, dict) and str(item.get("term") or "").strip()
        ]
        exploration_reason = str(exploration.get("reason") or "").strip()
        cards.append(
            f"""
            <section class="question-card">
              <div class="question-meta">{html.escape(str(section.get('id', '')))}</div>
              <h2>{html.escape(str(section.get('question', '')))}</h2>
              <div class="lane">
                <div class="lane-title">Reformulation</div>
                <div class="block">
                  <div class="block-title">Status</div>
                  <div class="debug-row">
                    <span class="debug-pill">{html.escape(reformulation_status)}</span>
                  </div>
                </div>
                <div class="block">
                  <div class="block-title">Extracted Concepts</div>
                  <div class="chips">{_render_terms(extracted_concepts)}</div>
                </div>
                <div class="block">
                  <div class="block-title">Shortlist Terms</div>
                  <div class="chips">{_render_candidate_chips(candidate_items)}</div>
                </div>
                <div class="block">
                  <div class="block-title">Span-Term Matches</div>
                  <div>{_render_span_matches(span_term_matches)}</div>
                </div>
                <div class="block">
                  <div class="block-title">Generated Candidates Sent To LLM</div>
                  <div>{_render_generated_candidates(generated_candidates)}</div>
                </div>
                <div class="block">
                  <div class="block-title">LLM Reformulation</div>
                  <div class="proposal">{html.escape(proposed_question or 'No reformulation proposed.')}</div>
                </div>
                <div class="block">
                  <div class="block-title">LLM Raw Output</div>
                  <div class="proposal muted">{html.escape(raw_response or 'No raw LLM response captured.')}</div>
                </div>
                <div class="block">
                  <div class="block-title">Post-processing</div>
                  <div class="debug-row">
                    <span class="debug-pill">{html.escape(postprocess_status or 'unknown')}</span>
                    <span class="debug-text">{html.escape(postprocess_reason or 'no_reason')}</span>
                  </div>
                  <div class="debug-subtle">{html.escape(parsed_question or 'No parsed reformulated question.')}</div>
                </div>
              </div>
              <div class="lane">
                <div class="lane-title">Exploration</div>
                <div class="block">
                  <div class="block-title">Status</div>
                  <div class="debug-row">
                    <span class="debug-pill">{html.escape(exploration_status)}</span>
                  </div>
                </div>
                <div class="block">
                  <div class="block-title">Source Mode</div>
                  <div class="debug-row">
                    <span class="debug-pill">{html.escape(exploration_source_mode or 'unknown')}</span>
                    <span class="debug-text">{html.escape(exploration_reason or 'no_reason')}</span>
                  </div>
                </div>
                <div class="block">
                  <div class="block-title">Extracted Concepts</div>
                  <div class="chips">{_render_terms(exploration_concepts)}</div>
                </div>
                <div class="block">
                  <div class="block-title">Terms By {html.escape('Quote' if exploration_quote_pools else 'Span')}</div>
                  <div>{_render_exploration_quote_pools(exploration_quote_pools) if exploration_quote_pools else _render_exploration_span_pools(exploration_span_pools)}</div>
                </div>
                <div class="block">
                  <div class="block-title">Suggested Terms</div>
                  <div class="chips">{_render_suggested_term_chips(exploration_suggested_terms)}</div>
                </div>
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        --bg: #f3efe8;
        --panel: #fffaf2;
        --ink: #201a14;
        --muted: #6d6457;
        --line: #d9c8ae;
        --accent: #8e5a2b;
        --chip: #efe1c9;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        background: linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
        font-family: Georgia, "Iowan Old Style", serif;
      }}
      .wrap {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 32px 18px 64px;
      }}
      .hero {{
        padding: 24px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        margin-bottom: 22px;
      }}
      .hero h1 {{
        margin: 0 0 8px;
        font-size: clamp(28px, 4vw, 44px);
        line-height: 1.05;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
      }}
      .question-card {{
        margin-top: 18px;
        padding: 22px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--panel);
      }}
      .lane {{
        margin-top: 18px;
        padding: 16px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(255,255,255,0.42);
      }}
      .lane-title {{
        margin-bottom: 12px;
        color: var(--ink);
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.01em;
      }}
      .question-meta {{
        color: var(--accent);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .question-card h2 {{
        margin: 8px 0 16px;
        font-size: clamp(22px, 3vw, 30px);
        line-height: 1.2;
      }}
      .block + .block {{
        margin-top: 16px;
      }}
      .block-title {{
        margin-bottom: 10px;
        color: var(--accent);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      .chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: var(--chip);
        font-size: 14px;
      }}
      .chip-derived {{
        background: #f6e7c4;
        border-color: #c78a2a;
        box-shadow: inset 0 0 0 1px rgba(199, 138, 42, 0.14);
      }}
      .chip-meta {{
        color: var(--muted);
        font-size: 12px;
      }}
      .chip-origin {{
        display: inline-flex;
        padding: 3px 8px;
        border-radius: 999px;
        background: rgba(142, 90, 43, 0.12);
        color: var(--accent);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.02em;
      }}
      .proposal {{
        padding: 16px 18px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #fffdf8;
        font-size: 18px;
        line-height: 1.5;
      }}
      .candidate-row {{
        padding: 12px 14px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #fffdf8;
      }}
      .candidate-row + .candidate-row {{
        margin-top: 10px;
      }}
      .candidate-text {{
        font-size: 16px;
        line-height: 1.4;
      }}
      .candidate-meta {{
        margin-top: 6px;
        color: var(--muted);
        font-size: 13px;
      }}
      .match-block + .match-block {{
        margin-top: 10px;
      }}
      .match-title {{
        margin-bottom: 8px;
        font-size: 14px;
        font-weight: 700;
      }}
      .debug-row {{
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .debug-pill {{
        display: inline-flex;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: var(--chip);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .debug-text {{
        color: var(--ink);
        font-size: 14px;
      }}
      .debug-subtle {{
        margin-top: 8px;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.4;
      }}
      .muted {{
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="hero">
        <h1>{html.escape(title)}</h1>
        <p>Compact view of the reformulation shortlist sent to the LLM and the reformulated question it proposed.</p>
      </section>
      {''.join(cards)}
    </main>
  </body>
</html>
"""


if __name__ == "__main__":
    args = parse_args()
    out = generate_exploration_mock(Path(args.data), Path(args.out) if args.out else None)
    print(out)
