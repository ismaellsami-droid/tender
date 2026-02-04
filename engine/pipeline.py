# engine/pipeline.py
import re
import html

from hobbes_engine import answer_with_citations_only, AuditRefused

REFUSAL_TEXT = "Je ne peux pas répondre à partir du corpus actuel."

_MARKER_RE = re.compile(r"【\d+:\d+†([^】]+)】")  # captures filename inside the marker


def _titleize_from_slug(tokens: str) -> str:
    words = [w for w in tokens.split("_") if w]
    return " ".join(w.capitalize() for w in words)


def _pretty_ref_from_slug(filename: str) -> str:
    base = re.sub(r"\.[A-Za-z0-9]+$", "", filename.strip())

    m = re.search(r"_ch_(\d{1,2})_(.+?)_s(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        s_num = int(m.group(3))
        s_title = _titleize_from_slug(m.group(4))
        return f"Leviathan I, Ch. {ch_num} — {ch_title}, §{s_num} — {s_title}"

    m = re.search(r"_ch_(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        return f"Leviathan I, Ch. {ch_num} — {ch_title}"

    return filename.strip()


def _strip_model_ref(line: str) -> str:
    line = re.sub(
        r"\s+Leviathan\s+(?:I|Book\s*1|Book\s*I)\s*,?\s*(?:Ch\.|Chapter)\s*\d+.*$",
        "",
        line,
        flags=re.IGNORECASE,
    )
    return line.strip()


def parse_and_format(raw: str) -> str:
    raw = re.split(r"\n\s*SOURCES\s*\(verified\).*", raw, flags=re.IGNORECASE | re.DOTALL)[0]
    raw = re.sub(r"^\s*-{10,}\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    lines = [ln.rstrip() for ln in raw.split("\n")]

    items = []
    current_quote_lines = []
    current_filename = None

    def flush():
        nonlocal current_quote_lines, current_filename
        if not current_quote_lines:
            current_filename = None
            return

        quote_text = " ".join([ln.strip() for ln in current_quote_lines if ln.strip()])
        quote_text = _strip_model_ref(quote_text).strip()

        ref = _pretty_ref_from_slug(current_filename) if current_filename else ""

        if quote_text:
            if ref:
                items.append(f"{quote_text}\n{ref}")
            else:
                items.append(quote_text)

        current_quote_lines = []
        current_filename = None

    for ln in lines:
        s = ln.strip()

        if not s:
            flush()
            continue

        m = _MARKER_RE.search(s)
        if m:
            current_filename = m.group(1).strip()
            s = _MARKER_RE.sub("", s).strip()

        if re.fullmatch(r"(?i)leviathan_.*\.(txt|md|pdf)", s):
            current_filename = s
            continue

        if s.startswith('"') and current_quote_lines:
            flush()

        current_quote_lines.append(s)

    flush()

    final = "\n\n".join(items).strip()

    lines2 = final.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fixed = []
    prev_blank = True
    for ln in lines2:
        if ln.startswith('"') and not prev_blank:
            fixed.append("")
        fixed.append(ln)
        prev_blank = (ln.strip() == "")

    final = "\n".join(fixed)
    final = re.sub(r"\n{3,}", "\n\n", final).strip()
    return final


def run_pipeline(question: str, mode: str = "short") -> dict:
    """
    Stable entry point for UI + eval.
    Returns:
      { "final_answer": str, "trace": dict }
    """
    trace = {
        "mode": mode,
        "question": question,
    }

    try:
        raw = answer_with_citations_only(question)
        final_text = parse_and_format(raw)

        trace["audit_passed"] = True
        # on peut enrichir plus tard: queries, retrieved_chunks, rerank...
        return {"final_answer": final_text, "trace": trace}

    except AuditRefused as e:
        trace["audit_passed"] = False
        trace["audit_reason"] = str(e)
        return {"final_answer": REFUSAL_TEXT, "trace": trace}
