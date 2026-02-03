import html
import re
import streamlit as st
from hobbes_engine import answer_with_citations_only, AuditRefused, reset_conversation_thread


# ---------- Ref formatting from filename slug ----------

def _titleize_from_slug(tokens: str) -> str:
    words = [w for w in tokens.split("_") if w]
    return " ".join(w.capitalize() for w in words)


def _pretty_ref_from_slug(filename: str) -> str:
    # strip extension
    base = re.sub(r"\.[A-Za-z0-9]+$", "", filename.strip())

    # ch + section
    m = re.search(r"_ch_(\d{1,2})_(.+?)_s(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        s_num = int(m.group(3))
        s_title = _titleize_from_slug(m.group(4))
        return f"Leviathan I, Ch. {ch_num} — {ch_title}, §{s_num} — {s_title}"

    # ch only
    m = re.search(r"_ch_(\d{1,2})_(.+)$", base, flags=re.IGNORECASE)
    if m:
        ch_num = int(m.group(1))
        ch_title = _titleize_from_slug(m.group(2))
        return f"Leviathan I, Ch. {ch_num} — {ch_title}"

    # fallback
    return filename.strip()


# ---------- Parsing + final formatting (we own the output) ----------

_MARKER_RE = re.compile(r"【\d+:\d+†([^】]+)】")  # captures filename inside the marker


def _strip_model_ref(line: str) -> str:
    """
    Remove any model-printed refs like:
      ... Leviathan I, Ch. 13, §4
      ... Leviathan Book 1, Chapter 13, §4
    We rebuild refs ourselves from filename slug.
    """
    # remove trailing "Leviathan ..." fragments
    line = re.sub(r"\s+Leviathan\s+(?:I|Book\s*1|Book\s*I)\s*,?\s*(?:Ch\.|Chapter)\s*\d+.*$", "", line, flags=re.IGNORECASE)
    return line.strip()


def parse_and_format(raw: str) -> str:
    # 1) drop SOURCES block and separators
    raw = re.split(r"\n\s*SOURCES\s*\(verified\).*", raw, flags=re.IGNORECASE | re.DOTALL)[0]
    raw = re.sub(r"^\s*-{10,}\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 2) normalize whitespace a bit (no need to rely on model's blank lines)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    # 3) We'll build items by scanning lines and starting a new item when we see a quote
    lines = [ln.rstrip() for ln in raw.split("\n")]

    items = []
    current_quote_lines = []
    current_filename = None

    def flush():
        nonlocal current_quote_lines, current_filename
        if not current_quote_lines:
            current_filename = None
            return

        # merge quote lines
        quote_text = " ".join([ln.strip() for ln in current_quote_lines if ln.strip()])
        quote_text = _strip_model_ref(quote_text)

        # ensure it remains quoted nicely (if the model already uses quotes, keep as-is)
        quote_text = quote_text.strip()

        # build ref from filename (if we have it)
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

        # blank line: end current item
        if not s:
            flush()
            continue

        # capture filename markers if present
        m = _MARKER_RE.search(s)
        if m:
            current_filename = m.group(1).strip()
            # remove the marker from the line entirely
            s = _MARKER_RE.sub("", s).strip()

        # if the line is a bare slug line (sometimes the model prints it alone)
        if re.fullmatch(r"(?i)leviathan_.*\.(txt|md|pdf)", s):
            current_filename = s
            continue

        # start of a new quote: if we already have one open, flush it first
        if s.startswith('"') and current_quote_lines:
            flush()

        current_quote_lines.append(s)

    flush()

    # 4) Final output: exactly one blank line between items
    final = "\n\n".join(items).strip()

    # Enforce exactly one blank line before any line that starts with a quote (")
    lines = final.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fixed = []
    prev_was_blank = True  # treat start as blank so we don't insert at top

    for ln in lines:
        is_quote_start = ln.startswith('"')
        if is_quote_start and not prev_was_blank:
            fixed.append("")  # insert one blank line

        fixed.append(ln)
        prev_was_blank = (ln.strip() == "")

    # collapse multiple blank lines to a single blank line
    final = "\n".join(fixed)
    final = re.sub(r"\n{3,}", "\n\n", final).strip()

    return final



# ---------- Streamlit UI ----------

st.title("Hobbes PoC — citations only")

if st.button("New thread"):
    reset_conversation_thread()
    st.success("New conversation thread created.")

q = st.text_input("Question")

if st.button("Ask") and q.strip():
    try:
        raw = answer_with_citations_only(q.strip())
        final_text = parse_and_format(raw)
        safe = html.escape(final_text)

        st.markdown(
            f"""
            <div style="
                white-space: pre-wrap;
                word-break: break-word;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                font-size: 0.9rem;
                line-height: 1.35;
                padding: 0.75rem;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 8px;
            ">{safe}</div>
            """,
            unsafe_allow_html=True
        )

    except AuditRefused as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {e}")
