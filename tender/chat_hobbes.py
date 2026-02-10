#!/usr/bin/env python3
import json
import time
from pathlib import Path
from openai import OpenAI

VECTOR_STORE_ID = "vs_69809284f32c8191936031cb849a1dfd"  # ton vector store
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

def update_assistant_instructions(client, assistant_id):
    client.beta.assistants.update(
        assistant_id=assistant_id,
        instructions=STRICT_INSTRUCTIONS,
        tools=[{"type": "file_search"}],
    )

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def ensure_assistant(client, assistant_id):
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
    print(f"✅ Created assistant: {a.id} (saved to {STATE_FILE})")
    return a.id

def attach_vector_store_to_assistant(client, assistant_id):
    # This is the key: bind the vector store to the assistant (not the run).
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    )

def create_thread(client):
    t = client.beta.threads.create()
    return t.id

def run_and_wait(client, thread_id, assistant_id):
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

def print_latest_assistant_message(client, thread_id, min_citations=1):
    msgs = client.beta.threads.messages.list(thread_id=thread_id, limit=20)

    for m in msgs.data:
        if m.role != "assistant":
            continue

        citation_count = 0
        text_blocks = []

        # 1) Collect text + count grounded citations
        for part in m.content:
            if part.type == "text":
                text_blocks.append(part.text.value)

                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if fc and getattr(fc, "file_id", None):
                        citation_count += 1

        # 2) Audit check
        if citation_count < min_citations:
            print("\n" + "=" * 80)
            print("⚠️  RESPONSE REFUSED (audit mode)")
            print("=" * 80)
            print(
                f"The assistant returned {citation_count} grounded citation(s), "
                f"but at least {min_citations} is required.\n"
                "This answer is NOT displayed."
            )
            return

        # 3) Display response (approved)
        print("\n" + "=" * 80)
        print("ASSISTANT (grounded)")
        print("=" * 80)

        for block in text_blocks:
            print(block.strip())

        # 4) Display sources clearly
        print("\n" + "-" * 80)
        print("SOURCES (verified)")
        print("-" * 80)

        for part in m.content:
            if part.type == "text":
                anns = getattr(part.text, "annotations", None) or []
                for ann in anns:
                    fc = getattr(ann, "file_citation", None)
                    if fc and getattr(fc, "file_id", None):
                        print(f"- file_id: {fc.file_id}")
                        print(f"  cited_span: {getattr(ann, 'text', '')}")

        return

    print("No assistant message found.")


def main():
    client = OpenAI()

    # Save vector_store_id in state.json (optional)
    state = load_state()
    state["vector_store_id"] = VECTOR_STORE_ID
    save_state(state)

    assistant_id = ensure_assistant(client, ASSISTANT_ID)
    update_assistant_instructions(client, assistant_id)
    attach_vector_store_to_assistant(client, assistant_id)

    # ✅ Bind vector store to the assistant (THIS fixes your error)
    attach_vector_store_to_assistant(client, assistant_id)

    thread_id = create_thread(client)

    print(f"✅ Using vector_store_id: {VECTOR_STORE_ID}")
    print(f"✅ Using assistant_id:   {assistant_id}")
    print(f"✅ Started thread_id:    {thread_id}")
    print("\nPose tes questions (FR ou EN). Tape 'exit' pour quitter.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=q,
        )

        run_and_wait(client, thread_id, assistant_id)
        print_latest_assistant_message(client, thread_id, min_citations=1)

    print("\nBye!")

if __name__ == "__main__":
    main()
