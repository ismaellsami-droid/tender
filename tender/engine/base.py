#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


class AuditRefused(Exception):
    """Raised when the audit mode refuses the response (not enough grounded citations)."""


class BaseCorpusEngine:
    """
    Generic engine that:
    - maintains a persistent assistant and a persistent thread per instance
    - binds a vector store to the assistant
    - answers with citations only (audit mode)
    - collects retrieval_pool from run steps (best effort)
    - enriches sources with filename -> (chapter/section/titles) when possible
    """

    def __init__(
        self,
        *,
        corpus_id: str,
        model: str,
        vector_store_id: str,
        instructions: str,
        assistant_id: Optional[str] = None,
        state_file: Path = Path("state.json"),
        poll_interval_s: float = 0.8,
    ) -> None:
        self.corpus_id = corpus_id
        self.model = model
        self.vector_store_id = vector_store_id
        self.instructions = instructions
        self.state_file = state_file
        self.poll_interval_s = poll_interval_s

        self._client: Optional[OpenAI] = None
        self._assistant_id: Optional[str] = assistant_id
        self._thread_id: Optional[str] = None
        self._ready: bool = False

        # file_id -> filename cache
        self._file_name_cache: Dict[str, Optional[str]] = {}

    # ----------------------------
    # small trace helper
    # ----------------------------
    def _t(self, trace, event: str, **data):
        if trace is None:
            return
        if hasattr(trace, "stamp"):
            trace.stamp(event, **data)
        else:
            trace.setdefault("events", []).append({"event": event, **data})

    # ----------------------------
    # state.json helpers
    # ----------------------------
    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # ----------------------------
    # OpenAI client
    # ----------------------------
    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    # ----------------------------
    # assistant / vector store binding
    # ----------------------------
    def _ensure_assistant(self) -> str:
        """
        Priority:
        1) self._assistant_id (passed from config)
        2) state.json key "assistant_id:<corpus_id>"
        3) create new assistant
        """
        client = self._get_client()

        if self._assistant_id:
            return self._assistant_id

        state = self._load_state()
        key = f"assistant_id:{self.corpus_id}"
        if state.get(key):
            self._assistant_id = state[key]
            return self._assistant_id

        a = client.beta.assistants.create(
            name=f"{self.corpus_id} (Strict)",
            model=self.model,
            instructions=self.instructions,
            tools=[{"type": "file_search"}],
        )
        self._assistant_id = a.id
        state[key] = a.id
        self._save_state(state)
        return a.id

    def _update_assistant_instructions(self, assistant_id: str) -> None:
        client = self._get_client()
        client.beta.assistants.update(
            assistant_id=assistant_id,
            instructions=self.instructions,
            tools=[{"type": "file_search"}],
        )

    def _attach_vector_store(self, assistant_id: str) -> None:
        client = self._get_client()
        client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}},
        )

    # ----------------------------
    # thread
    # ----------------------------
    def _create_thread(self) -> str:
        client = self._get_client()
        t = client.beta.threads.create()
        return t.id

    # ----------------------------
    # filename resolution
    # ----------------------------
    def file_id_to_filename(self, file_id: str) -> Optional[str]:
        if not file_id:
            return None
        if file_id in self._file_name_cache:
            return self._file_name_cache[file_id]

        client = self._get_client()
        try:
            f = client.files.retrieve(file_id)
            filename = getattr(f, "filename", None)
            self._file_name_cache[file_id] = filename
            return filename
        except Exception:
            self._file_name_cache[file_id] = None
            return None

    # ----------------------------
    # retrieval pool extraction (best effort)
    # ----------------------------
    def extract_file_search_pool_from_run_steps(self, thread_id: str, run_id: str, trace) -> None:
        client = self._get_client()
        try:
            steps = client.beta.threads.runs.steps.list(thread_id=thread_id, run_id=run_id)
        except Exception as e:
            self._t(trace, "run_steps_list_failed", error=str(e))
            return

        data = getattr(steps, "data", None) or (steps.get("data", []) if isinstance(steps, dict) else [])
        file_search_events = 0

        for step in data or []:
            details = getattr(step, "step_details", None) or (
                step.get("step_details", {}) if isinstance(step, dict) else {}
            )
            tool_calls = getattr(details, "tool_calls", None) or (
                details.get("tool_calls", []) if isinstance(details, dict) else []
            )

            for tc in tool_calls or []:
                tc_type = getattr(tc, "type", None) or (tc.get("type") if isinstance(tc, dict) else None)
                if tc_type != "file_search":
                    continue

                fs = getattr(tc, "file_search", None) or (tc.get("file_search", {}) if isinstance(tc, dict) else {})
                results = None
                if isinstance(fs, dict):
                    results = fs.get("results") or fs.get("output") or fs.get("documents")
                else:
                    results = getattr(fs, "results", None) or getattr(fs, "output", None) or getattr(fs, "documents", None)

                pool = []
                for i, r in enumerate(results or []):
                    if not isinstance(r, dict):
                        r = getattr(r, "model_dump", lambda: {})() or {}

                    file_id = r.get("file_id") or (r.get("file", {}) or {}).get("id")
                    filename = self.file_id_to_filename(file_id) if isinstance(file_id, str) else None
                    score = r.get("score") or r.get("relevance_score")

                    snippet = r.get("text") or r.get("content") or r.get("excerpt") or ""
                    if isinstance(snippet, dict):
                        snippet = snippet.get("text", "")

                    pool.append(
                        {
                            "rank": i + 1,
                            "file_id": file_id,
                            "filename": filename,
                            "score": score,
                            "snippet": (snippet[:500] if isinstance(snippet, str) else ""),
                        }
                    )

                self._t(trace, "retrieval_pool", source="run_steps.file_search", count=len(pool), pool=pool)
                file_search_events += 1

        self._t(trace, "run_steps_parsed", file_search_events=file_search_events, steps_count=len(data or []))

    # ----------------------------
    # assistant message extraction (citations + sources)
    # ----------------------------
    def extract_latest_assistant_message_or_none(
        self,
        thread_id: str,
        trace=None,
    ) -> Optional[Tuple[List[str], int, List[Tuple[str, str]], List[Dict[str, Any]]]]:
        client = self._get_client()

        self._t(trace, "messages_list_start", thread_id=thread_id)
        msgs = client.beta.threads.messages.list(thread_id=thread_id, limit=20)
        self._t(trace, "messages_list_end", count=len(getattr(msgs, "data", []) or []))

        for m in msgs.data:
            if m.role != "assistant":
                continue

            citation_count = 0
            text_blocks: List[str] = []
            sources: List[Tuple[str, str]] = []
            traced_sources: List[Dict[str, Any]] = []
            msg_id = getattr(m, "id", None)

            # 1) count citations
            for part in m.content:
                if part.type != "text":
                    continue
                text_blocks.append(part.text.value)
                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if fc and getattr(fc, "file_id", None):
                        citation_count += 1

            # 2) extract sources
            for part in m.content:
                if part.type != "text":
                    continue
                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if not (fc and getattr(fc, "file_id", None)):
                        continue
                    file_id = fc.file_id
                    cited_span = getattr(ann, "text", "") or ""
                    sources.append((file_id, cited_span))

                    filename = self.file_id_to_filename(file_id)
                    traced_sources.append(
                        {
                            "file_id": file_id,
                            "filename": filename,
                            "cited_span": cited_span[:400],
                        }
                    )

            self._t(
                trace,
                "assistant_message_extracted",
                message_id=msg_id,
                text_blocks_count=len(text_blocks),
                text_total_len=sum(len(t) for t in text_blocks),
                grounded_citation_count=citation_count,
                sources_count=len(traced_sources),
                sources=traced_sources,
            )

            return text_blocks, citation_count, sources, traced_sources

        self._t(trace, "no_assistant_message_found")
        return None

    # ----------------------------
    # formatting
    # ----------------------------
    def format_answer(self, text_blocks: List[str], sources: List[Tuple[str, str]]) -> str:
        out_lines: List[str] = []
        out_lines.extend([b.strip() for b in text_blocks])

        out_lines.append("\n" + "-" * 80)
        out_lines.append("SOURCES (verified)")
        out_lines.append("-" * 80)
        for file_id, cited_span in sources:
            out_lines.append(f"- file_id: {file_id}")
            out_lines.append(f"  cited_span: {cited_span}")

        return "\n".join(out_lines).strip()

    # ----------------------------
    # readiness
    # ----------------------------
    def ensure_ready(self) -> Tuple[str, str]:
        """
        Ensure assistant exists, instructions set, vector store attached, and thread exists.
        Returns (assistant_id, thread_id)
        """
        if self._ready and self._assistant_id and self._thread_id:
            return self._assistant_id, self._thread_id

        assistant_id = self._ensure_assistant()

        # Keep a pointer for debugging
        state = self._load_state()
        state[f"vector_store_id:{self.corpus_id}"] = self.vector_store_id
        self._save_state(state)

        self._update_assistant_instructions(assistant_id)
        self._attach_vector_store(assistant_id)

        self._thread_id = self._create_thread()
        self._ready = True
        return assistant_id, self._thread_id

    def reset_thread(self) -> str:
        """
        Create a new conversation thread (keeps same assistant & vector store binding).
        """
        self.ensure_ready()
        self._thread_id = self._create_thread()
        self._ready = True
        return self._thread_id

    # ----------------------------
    # run
    # ----------------------------
    def run_and_wait(self, thread_id: str, assistant_id: str, trace=None) -> str:
        client = self._get_client()
        self._t(trace, "run_create_start", thread_id=thread_id, assistant_id=assistant_id)

        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
        run_id = run.id
        self._t(trace, "run_created", run_id=run_id)

        t0 = time.time()
        while True:
            r = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            status = r.status
            self._t(trace, "run_poll", run_id=run_id, status=status)

            if status in ("completed", "failed", "cancelled", "expired"):
                dt_ms = int((time.time() - t0) * 1000)
                self._t(trace, "run_terminal", run_id=run_id, status=status, duration_ms=dt_ms)

                if status != "completed":
                    raise RuntimeError(f"Run ended with status={status}")

                # retrieval pool best-effort
                self.extract_file_search_pool_from_run_steps(thread_id, run_id, trace)
                return run_id

            time.sleep(self.poll_interval_s)

    # ----------------------------
    # public answer
    # ----------------------------
    def answer_with_citations_only(self, question: str, *, min_citations: int = 1, trace=None) -> str:
        q = (question or "").strip()
        if not q:
            raise ValueError("Empty question.")

        client = self._get_client()
        assistant_id, thread_id = self.ensure_ready()

        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=q)
        self.run_and_wait(thread_id, assistant_id, trace=trace)

        extracted = self.extract_latest_assistant_message_or_none(thread_id, trace=trace)
        if extracted is None:
            raise RuntimeError("No assistant message found.")

        text_blocks, citation_count, sources, _traced_sources = extracted

        if citation_count < min_citations:
            raise AuditRefused(
                f"RESPONSE REFUSED (audit mode): {citation_count} grounded citation(s), requires >= {min_citations}."
            )

        return self.format_answer(text_blocks, sources)
