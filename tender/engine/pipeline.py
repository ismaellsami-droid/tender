# tender/engine/pipeline.py
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tender.corpora.registry import default_corpus_id, get_corpus
from tender.engine.trace import TraceCollector
from tender.engine.engine import answer_with_citations, _ch_sec_from_filename

REFUSAL_TEXT = "Je ne peux pas répondre à partir du corpus actuel."

# --- Extract (ch, §) from the pretty ref line you already build ---
_REF_CH_SEC_RE = re.compile(r"Ch\.\s*(\d{1,2}).*?§\s*(\d{1,2})", re.IGNORECASE)


def _parse_ch_sec_from_ref(ref: str) -> Optional[Tuple[int, int]]:
    m = _REF_CH_SEC_RE.search(ref or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _read_local_section_text(data_dir: str, filename: str) -> str:
    p = Path(data_dir) / filename
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(encoding="latin-1")
        except Exception as e:
            return f"[ERROR reading {p}: {e}]"

def build_selection_view(
    trace_dict: Dict[str, Any],
    *,
    final_quotes: List[Dict[str, str]],
    data_dir: str = "data",
) -> Dict[str, Any]:
    """
    Returns:
      {
        "event": "retrieval_pool",
        "count": N,
        "summary": {...},
        "pool": [
          {..., "selected":"Y/N", "quote": "...", "section_text": "...", ...}
        ]
      }
    """
    events: List[Dict[str, Any]] = trace_dict.get("events", [])

    # 1) Get latest retrieval_pool event
    retrieval_ev = None
    for ev in reversed(events):
        if ev.get("event") == "retrieval_pool":
            retrieval_ev = ev
            break
    if not retrieval_ev:
        return {"event": "retrieval_pool", "count": 0, "pool": [], "note": "No retrieval_pool event found"}

    pool = retrieval_ev.get("pool") or []
    if not isinstance(pool, list):
        pool = []

    # Map (ch, sec) -> quote text from the final_quotes list
    selected_by_ch_sec: Dict[Tuple[int, int], str] = {}
    for q in final_quotes or []:
        chsec = _parse_ch_sec_from_ref(q.get("ref", ""))
        if chsec:
            selected_by_ch_sec[chsec] = q.get("quote", "")

    # Build selection entries — selected flag already set by select_passages
    enriched_pool = []
    for item in pool:
        if not isinstance(item, dict):
            continue

        filename = item.get("filename")

        # Ensure chapter/section exist (fallback to filename parsing)
        ch = item.get("chapter")
        sec = item.get("section")
        if not isinstance(ch, int) or (sec is not None and not isinstance(sec, int)):
            if isinstance(filename, str) and filename:
                ch2, sec2 = _ch_sec_from_filename(filename)
                if isinstance(ch2, int):
                    ch = ch2
                if isinstance(sec2, int):
                    sec = sec2

        is_selected = str(item.get("selected", "")).upper() == "Y"

        out = dict(item)
        out["chapter"] = ch
        out["section"] = sec

        if is_selected:
            qtxt = ""
            if isinstance(ch, int) and isinstance(sec, int):
                qtxt = selected_by_ch_sec.get((ch, sec), "")
            out["quote"] = qtxt
        else:
            if isinstance(filename, str) and filename:
                out["section_text"] = _read_local_section_text(data_dir, filename)
            else:
                out["section_text"] = "[No filename available to load local text]"

        enriched_pool.append(out)


    # --- Build summary from enriched_pool ---
    retrieval_chapters = sorted({it.get("chapter") for it in enriched_pool if isinstance(it.get("chapter"), int)})

    used_chapters = sorted(
        {it.get("chapter") for it in enriched_pool if it.get("selected") == "Y" and isinstance(it.get("chapter"), int)}
    )

    used_ranks = []
    for it in enriched_pool:
        if it.get("selected") != "Y":
            continue
        rk = it.get("rank")
        sc = it.get("score")
        if isinstance(rk, int) and isinstance(sc, (int, float)):
            used_ranks.append(f"{rk}:{sc:.4f}")
        elif isinstance(rk, int):
            used_ranks.append(f"{rk}:{sc}")

    seen = set()
    used_ranks = [x for x in used_ranks if not (x in seen or seen.add(x))]

    summary = {
        "retrieval_chapters": retrieval_chapters,
        "used_chapters": used_chapters,
        "used_ranks": used_ranks,
    }

    return {
        "event": "retrieval_pool",
        "count": len(enriched_pool),
        "summary": summary,
        "pool": enriched_pool,
    }


def enrich_trace_with_retrieval_alignment(trace: Dict[str, Any]) -> None:
    """
    - Adds retrieval_rank / retrieval_score to cited sources (assistant_message_extracted.sources)
    - Adds a trace_summary event (coverage + ranks)
    Works purely from trace.events (no extra API calls).
    """
    events: List[Dict[str, Any]] = trace.get("events", [])

    # 1) Build file_id -> retrieval metadata index from retrieval_pool
    retrieval_index: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        if ev.get("event") != "retrieval_pool":
            continue
        pool = ev.get("pool") or []
        for item in pool:
            file_id = item.get("file_id")
            if isinstance(file_id, str) and file_id not in retrieval_index:
                retrieval_index[file_id] = {
                    "retrieval_rank": item.get("rank"),
                    "retrieval_score": item.get("score"),
                    "retrieval_filename": item.get("filename"),
                    "retrieval_chapter": item.get("chapter"),
                    "retrieval_section": item.get("section"),
                    "retrieval_chapter_title": item.get("chapter_title"),
                    "retrieval_section_title": item.get("section_title"),
                }

    # 2) Enrich cited sources in assistant_message_extracted
    used_files: List[str] = []
    used_ranks: List[int] = []
    used_chapters: List[int] = []

    for ev in events:
        if ev.get("event") != "assistant_message_extracted":
            continue
        sources = ev.get("sources") or []
        if not isinstance(sources, list):
            continue

        for s in sources:
            if not isinstance(s, dict):
                continue
            fid = s.get("file_id")
            if not isinstance(fid, str):
                continue

            meta = retrieval_index.get(fid)
            if meta:
                s.update(
                    {
                        "retrieval_rank": meta.get("retrieval_rank"),
                        "retrieval_score": meta.get("retrieval_score"),
                    }
                )

            used_files.append(fid)

            ch = s.get("chapter")
            if isinstance(ch, int):
                used_chapters.append(ch)

            rk = s.get("retrieval_rank")
            if isinstance(rk, int):
                used_ranks.append(rk)

    # 3) Compute retrieval coverage stats
    retrieval_files = set(retrieval_index.keys())

    retrieval_chapters = []
    for _fid, meta in retrieval_index.items():
        ch = meta.get("retrieval_chapter")
        if isinstance(ch, int):
            retrieval_chapters.append(ch)

    used_files_set = set(used_files)
    used_chapters_set = sorted(set([c for c in used_chapters if isinstance(c, int)]))
    retrieval_chapters_set = sorted(set([c for c in retrieval_chapters if isinstance(c, int)]))

    def _mean(xs: List[int]) -> Optional[float]:
        xs2 = [x for x in xs if isinstance(x, int)]
        if not xs2:
            return None
        return sum(xs2) / len(xs2)

    summary = {
        "retrieval_pool_files": len(retrieval_files),
        "used_files": len(used_files_set),
        "used_files_in_pool": len(used_files_set.intersection(retrieval_files)),
        "retrieval_chapters": retrieval_chapters_set,
        "used_chapters": used_chapters_set,
        "chapter_coverage_ratio": (len(used_chapters_set) / len(retrieval_chapters_set)) if retrieval_chapters_set else None,
        "used_ranks": sorted(set(used_ranks)),
        "max_used_rank": max(used_ranks) if used_ranks else None,
        "mean_used_rank": _mean(used_ranks),
    }

    events.append(
        {
            "t_ms": events[-1].get("t_ms", 0) if events else 0,
            "event": "trace_summary",
            **summary,
        }
    )


def _titleize_from_slug(tokens: str) -> str:
    words = [w for w in tokens.split("_") if w]
    return " ".join(w.capitalize() for w in words)


def _pretty_ref_from_slug(filename: str, *, reference_label: str) -> str:
    base = re.sub(r"\.[A-Za-z0-9]+$", "", filename.strip())

    m = re.search(r"_ch_(\d{1,2})_(.+?)_s(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        s_num = int(m.group(3))
        s_title = _titleize_from_slug(m.group(4))
        return f"{reference_label}, Ch. {ch_num} — {ch_title}, §{s_num} — {s_title}"

    m = re.search(r"_ch_(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        return f"{reference_label}, Ch. {ch_num} — {ch_title}"

    return filename.strip()


def _format_final_from_quotes(
    quotes: List[Dict[str, str]],
    *,
    reference_label: str,
) -> tuple[str, List[Dict[str, str]], List[str]]:
    """
    Convert [{"quote","filename"}] into:
      - final text blocks
      - quote/ref items for trace
      - filenames list
    """
    items: List[Dict[str, str]] = []
    filenames: List[str] = []
    blocks: List[str] = []

    for q in quotes:
        quote = (q.get("quote") or "").strip()
        filename = (q.get("filename") or "").strip()
        if not quote or not filename:
            continue
        ref = _pretty_ref_from_slug(filename, reference_label=reference_label)
        items.append({"quote": quote, "ref": ref})
        filenames.append(filename)
        blocks.append(f"{quote}\n{ref}")

    uniq_files = sorted(set(filenames))
    final = "\n\n".join(blocks).strip()
    return final, items, uniq_files


def run_pipeline(
    question: str,
    *,
    corpus_id: str = default_corpus_id(),
    data_dir: Optional[str] = None,
) -> dict:
    corpus = get_corpus(corpus_id)
    effective_data_dir = data_dir or corpus.data_dir

    tr = TraceCollector()
    tr.set_meta(
        question=question,
        corpus_id=corpus_id,
        data_dir=effective_data_dir,
    )

    tr.stamp("pipeline_start")
    tr.stamp("engine_call_start")
    out = answer_with_citations(
        question,
        corpus_id=corpus_id,
        retrieval_k=getattr(corpus, "retrieval_k", 20),
        selection_k=getattr(corpus, "selection_k", 8),
    )
    tr.stamp("engine_call_end", raw_len=len(out.get("model_output", "")))

    pool = out.get("pool", []) or []
    if isinstance(pool, list):
        tr.stamp("retrieval_pool", source="vector_store.search", count=len(pool), pool=pool)

    selected_sources = []
    for it in pool:
        if not isinstance(it, dict) or str(it.get("selected", "")).upper() != "Y":
            continue
        selected_sources.append(
            {
                "file_id": it.get("file_id"),
                "filename": it.get("filename"),
                "chapter": it.get("chapter"),
                "section": it.get("section"),
                "retrieval_rank": it.get("rank"),
                "retrieval_score": it.get("score"),
            }
        )
    tr.stamp(
        "retrieval_selection",
        strategy="top_k",
        selected_count=len(selected_sources),
        selection_k=getattr(corpus, "selection_k", 8),
    )
    tr.stamp(
        "assistant_message_extracted",
        message_id=None,
        text_blocks_count=1,
        text_total_len=len(out.get("model_output", "")),
        grounded_citation_count=len(selected_sources),
        sources_count=len(selected_sources),
        sources=selected_sources,
    )

    final_text, items, files = _format_final_from_quotes(
        out.get("quotes", []) or [],
        reference_label=corpus.reference_label,
    )
    tr.stamp("format_end", final_len=len(final_text))
    tr.stamp("sources_selected", count=len(files), filenames=files)
    tr.stamp("quotes_selected", count=len(items), quotes=items)

    if out.get("refused", False) or not items:
        tr.set_meta(audit_passed=False, audit_reason="No verified quote extracted from fixed context")
        tr.stamp("audit_refused")
        trace_dict = tr.to_dict()
        enrich_trace_with_retrieval_alignment(trace_dict)
        selection = build_selection_view(trace_dict, final_quotes=[], data_dir=effective_data_dir)
        return {"final_answer": REFUSAL_TEXT, "trace": trace_dict, "selection": selection}

    tr.set_meta(audit_passed=True)
    trace_dict = tr.to_dict()
    enrich_trace_with_retrieval_alignment(trace_dict)
    selection = build_selection_view(trace_dict, final_quotes=items, data_dir=effective_data_dir)
    return {"final_answer": final_text, "trace": trace_dict, "selection": selection}
