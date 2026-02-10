# hobbes_engine.py
#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from openai import OpenAI

# ---------------------------------------------------------------------
# CONFIG (copié de chat_hobbes.py)
# ---------------------------------------------------------------------

VECTOR_STORE_ID = "vs_69809284f32c8191936031cb849a1dfd"
ASSISTANT_ID = "asst_mJ0aX3HVGDQCvey3h29r9kgF"
MODEL = "gpt-4.1-mini"
STATE_FILE = Path("state.json")

STRICT_INSTRUCTIONS = """You are a source-only retrieval assistant for Hobbes.

TASK:
Given a user question, you must respond ONLY with verbatim quotations from the retrieved sources that help answer the question.

OUTPUT FORMAT (STRICT):
•⁠  ⁠Return between 2 and 4 quotations (minimum 1).
•⁠  ⁠Each quotation must be:
  1) Verbatim (exact text from the source, in English).
  2) Short (1–2 sentences, max ~40 words).
  3) Immediately followed by its reference on the same line, formatted as:
     Leviathan I, Ch. <number>, §<number>
     (If the section number cannot be inferred reliably, write:
      Leviathan I, Ch. <number>)

RULES:
•⁠  ⁠Do NOT explain, summarize, paraphrase, or comment in any way.
•⁠  ⁠Do NOT add any introductory or concluding text.
•⁠  ⁠Do NOT use outside knowledge.
•⁠  ⁠Use ONLY excerpts retrieved via file_search from the attached vector store.
•⁠  ⁠If you cannot find at least ONE relevant quotation, output ONLY:
  "Je ne peux pas répondre à partir du corpus actuel."
•⁠  ⁠Infer Chapter and Section numbers from the source filename when possible
  (e.g. filename contains "ch_13" and "s01" → Leviathan I, Ch. 13, §1).
"""

# ---------------------------------------------------------------------
# INTERNAL SINGLETON STATE (pour Streamlit)
# ---------------------------------------------------------------------

_client: Optional[OpenAI] = None
_engine_ready: bool = False
_cached_assistant_id: Optional[str] = None
_cached_thread_id: Optional[str] = None


# ---------------------------------------------------------------------
# Helpers (copiés / adaptés sans changer la logique)
# ---------------------------------------------------------------------

def _titleize_from_slug(tokens: str) -> str:
    """
    Convertit 'of_other_lawes_of_nature' -> 'Of Other Lawes Of Nature'
    (on garde la casse simple; tu peux affiner après)
    """
    words = [w for w in tokens.split("_") if w]
    return " ".join(w.capitalize() for w in words)


def _parse_hobbes_slug(file_id: str):
    """
    Parse un id du type:
    leviathan_book01_of_man_ch_15_of_other_lawes_of_nature_s02_justice_and_injustice_what.txt
    Retourne dict avec ch_num, ch_title, s_num, s_title (si trouvables).
    """
    # on retire l'extension éventuelle
    base = re.sub(r"\.[A-Za-z0-9]+$", "", file_id)

    # chapitre: ch_<num>_<titre...>_s<num>_...
    m_ch = re.search(r"_ch_(\d{1,2})_(.+?)_s(\d{1,2})_(.+)$", base)
    if m_ch:
        ch_num = int(m_ch.group(1))
        ch_title_slug = m_ch.group(2)
        s_num = int(m_ch.group(3))
        s_title_slug = m_ch.group(4)

        return {
            "ch_num": ch_num,
            "ch_title": _titleize_from_slug(ch_title_slug),
            "s_num": s_num,
            "s_title": _titleize_from_slug(s_title_slug),
        }

    # fallback: si pas de section
    m_ch_only = re.search(r"_ch_(\d{1,2})_(.+)$", base)
    if m_ch_only:
        ch_num = int(m_ch_only.group(1))
        ch_title_slug = m_ch_only.group(2)
        return {
            "ch_num": ch_num,
            "ch_title": _titleize_from_slug(ch_title_slug),
            "s_num": None,
            "s_title": None,
        }

    return None


def _pretty_ref_from_file_id(file_id: str) -> str:
    """
    Rend une référence lisible à partir du slug.
    Fallback: renvoie file_id tel quel si parsing impossible.
    """
    parsed = _parse_hobbes_slug(file_id)
    if not parsed:
        return file_id

    ch_num = parsed["ch_num"]
    ch_title = parsed["ch_title"]
    s_num = parsed.get("s_num")
    s_title = parsed.get("s_title")

    if s_num is not None and s_title:
        return f"Leviathan I, Ch. {ch_num} — {ch_title}, §{s_num} — {s_title}"
    return f"Leviathan I, Ch. {ch_num} — {ch_title}"


def update_assistant_instructions(client: OpenAI, assistant_id: str) -> None:
    client.beta.assistants.update(
        assistant_id=assistant_id,
        instructions=STRICT_INSTRUCTIONS,
        tools=[{"type": "file_search"}],
    )


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_assistant(client: OpenAI, assistant_id: str) -> str:
    # 1) reuse provided id if any
    if assistant_id:
        return assistant_id

    # 2) reuse from state.json if any
    state = load_state()
    if state.get("assistant_id"):
        return state["assistant_id"]

    # 3) otherwise create a new assistant
    a = client.beta.assistants.create(
        name="Hobbes Explorer (Strict)",
        model=MODEL,
        instructions=STRICT_INSTRUCTIONS,
        tools=[{"type": "file_search"}],
    )
    state["assistant_id"] = a.id
    save_state(state)
    # print conservé dans le script d'origine, mais ici on évite le bruit en lib
    return a.id


def attach_vector_store_to_assistant(client: OpenAI, assistant_id: str) -> None:
    # Bind the vector store to the assistant (not the run).
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    )


def create_thread(client: OpenAI) -> str:
    t = client.beta.threads.create()
    return t.id


def run_and_wait(client: OpenAI, thread_id: str, assistant_id: str) -> str:
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    while True:
        r = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if r.status in ("completed", "failed", "cancelled", "expired"):
            if r.status != "completed":
                raise RuntimeError(f"Run ended with status={r.status}")
            return run.id
        time.sleep(0.8)


def _extract_latest_assistant_message_or_none(
    client: OpenAI,
    thread_id: str,
) -> Optional[Tuple[List[str], int, List[Tuple[str, str]]]]:
    """
    Retourne:
      - text_blocks (liste des textes)
      - citation_count (nombre de grounded citations via file_citation.file_id)
      - sources: liste de (file_id, cited_span)
    ou None si aucun message assistant.
    """
    msgs = client.beta.threads.messages.list(thread_id=thread_id, limit=20)

    for m in msgs.data:
        if m.role != "assistant":
            continue

        citation_count = 0
        text_blocks: List[str] = []
        sources: List[Tuple[str, str]] = []

        # 1) Collect text + count grounded citations
        for part in m.content:
            if part.type == "text":
                text_blocks.append(part.text.value)

                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if fc and getattr(fc, "file_id", None):
                        citation_count += 1

        # 2) Collect sources like the script prints them
        for part in m.content:
            if part.type == "text":
                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if fc and getattr(fc, "file_id", None):
                        sources.append((fc.file_id, getattr(ann, "text", "")))

        return text_blocks, citation_count, sources

    return None


def _format_answer(text_blocks: List[str], sources: List[Tuple[str, str]]) -> str:
    # Emule ton affichage console mais en string
    out_lines: List[str] = []
    for block in text_blocks:
        out_lines.append(block.strip())

    out_lines.append("\n" + "-" * 80)
    out_lines.append("SOURCES (verified)")
    out_lines.append("-" * 80)

    for file_id, cited_span in sources:
        out_lines.append(f"- source: {_pretty_ref_from_file_id(file_id)}")
        out_lines.append(f"  cited_span: {cited_span}")

    return "\n".join(out_lines).strip()


def _ensure_engine_ready() -> Tuple[OpenAI, str, str]:
    """
    Prépare une fois:
    - client
    - assistant (ensure + update instructions + attach vector store)
    - thread persistant
    """
    global _client, _engine_ready, _cached_assistant_id, _cached_thread_id

    if _engine_ready and _client and _cached_assistant_id and _cached_thread_id:
        return _client, _cached_assistant_id, _cached_thread_id

    if _client is None:
        _client = OpenAI()

    # Save vector_store_id in state.json (optional) — identique à ton main()
    state = load_state()
    state["vector_store_id"] = VECTOR_STORE_ID
    save_state(state)

    assistant_id = ensure_assistant(_client, ASSISTANT_ID)
    update_assistant_instructions(_client, assistant_id)

    attach_vector_store_to_assistant(_client, assistant_id)

    thread_id = create_thread(_client)

    _cached_assistant_id = assistant_id
    _cached_thread_id = thread_id
    _engine_ready = True

    return _client, assistant_id, thread_id


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

class AuditRefused(Exception):
    """Raised when the audit mode refuses the response (not enough grounded citations)."""


def answer_with_citations_only(question: str, min_citations: int = 1) -> str:
    """
    Exactement la même logique que chat_hobbes.py:
    - ajoute le message user dans le thread
    - run
    - récupère le dernier message assistant
    - compte les grounded citations (file_citation.file_id)
    - refuse si < min_citations
    - renvoie le texte verbatim + sources vérifiées

    NOTE:
    - Thread persistant (comme ton script interactif) => contexte conversationnel conservé.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Empty question.")

    client, assistant_id, thread_id = _ensure_engine_ready()

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=q,
    )

    run_and_wait(client, thread_id, assistant_id)

    extracted = _extract_latest_assistant_message_or_none(client, thread_id)
    if extracted is None:
        raise RuntimeError("No assistant message found.")

    text_blocks, citation_count, sources = extracted

    # Audit check (identique)
    if citation_count < min_citations:
        raise AuditRefused(
            f"RESPONSE REFUSED (audit mode): "
            f"{citation_count} grounded citation(s), requires >= {min_citations}."
        )

    return _format_answer(text_blocks, sources)


def reset_conversation_thread() -> str:
    """
    Option utile pour Streamlit: reset le thread (nouvelle conversation)
    sans toucher à l'assistant ni au vector store.
    """
    global _cached_thread_id, _engine_ready

    client, assistant_id, _old_thread_id = _ensure_engine_ready()
    new_thread_id = create_thread(client)
    _cached_thread_id = new_thread_id
    # engine reste prêt, on change juste le thread
    _engine_ready = True
    return new_thread_id
