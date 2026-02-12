# tender/app.py
import html
import traceback

import streamlit as st

from tender.engine.pipeline import run_pipeline
from tender.engine.engine import reset_conversation_thread
from tender.corpora.registry import list_corpora


def main():
    st.title("Tender — citations only")

    # ✅ Liste dynamique des corpus depuis le registry
    corpora = list_corpora()
    if not corpora:
        st.error("No corpus configured in registry.")
        st.stop()

    by_id = {c.corpus_id: c for c in corpora}
    labels = {c.corpus_id: c.label for c in corpora}

    corpus_ids: list[str] = [c.corpus_id for c in corpora]

    corpus_id = st.sidebar.selectbox(
        "Corpus",
        options=corpus_ids,
        format_func=lambda x: labels[x],
        index=0,
    )

    if st.button("New thread"):
        reset_conversation_thread(corpus_id=corpus_id)
        st.success("New conversation thread created.")

    q = st.text_input("Question")

    if st.button("Ask") and q.strip():
        try:
            result = run_pipeline(
                q.strip(),
                mode="short",
                corpus_id=corpus_id,
                data_dir=by_id[corpus_id].data_dir,
            )
            final_text = result["final_answer"]
            safe = html.escape(final_text)

            st.markdown(
                f"""
                <div style="white-space: pre-wrap; word-break: break-word;
                            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                            font-size: 0.9rem; line-height: 1.35;
                            padding: 0.75rem;
                            border: 1px solid rgba(49, 51, 63, 0.2);
                            border-radius: 8px;">{safe}</div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.error("Erreur pendant run_pipeline()")
            st.code(traceback.format_exc())
            st.stop()


if __name__ == "__main__":
    main()
