import html
import streamlit as st
from hobbes_engine import answer_with_citations_only, AuditRefused, reset_conversation_thread

st.title("Hobbes PoC — citations only")

if st.button("New thread"):
    reset_conversation_thread()
    st.success("New conversation thread created.")

q = st.text_input("Question")
if st.button("Ask") and q.strip():
    try:
        ans = answer_with_citations_only(q.strip())

        # Échapper le HTML pour éviter tout souci d'affichage
        safe = html.escape(ans)

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
