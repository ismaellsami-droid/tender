import html
import traceback
import streamlit as st
from tender.engine.pipeline import run_pipeline
from tender.hobbes_engine import reset_conversation_thread

def main():
    st.title("Hobbes PoC — citations only")

    if st.button("New thread"):
        reset_conversation_thread()
        st.success("New conversation thread created.")

    q = st.text_input("Question")

    if st.button("Ask") and q.strip():
        try:
            result = run_pipeline(q.strip(), mode="short")
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
