"""
Facts-Only MF Assistant — Streamlit UI

A RAG-based chatbot answering factual questions about HDFC Mutual Fund schemes
(Top 100, Flexi Cap, ELSS Tax Saver, Mid-Cap Opportunities) using official
AMC, SEBI, and AMFI sources. Accessed via Groww platform.

No investment advice is given. Every answer includes a source citation.
"""

import streamlit as st
from rag_engine import RAGEngine

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facts-Only MF Assistant",
    page_icon="💹",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for cleaner look ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88em;
        margin-bottom: 16px;
    }
    .source-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1a73e8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88em;
        margin-top: 8px;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 8px;
    }
    .refusal-box {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Title & Welcome ─────────────────────────────────────────────────────────────
st.title("💹 Facts-Only MF Assistant")
st.markdown(
    "**Welcome!** Get verified factual answers about HDFC Mutual Fund schemes "
    "sourced from official AMC, SEBI, and AMFI pages. "
    "All answers include a citation link."
)

# ── Disclaimer (required per problem spec) ─────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer-box">
    ⚠️ <strong>Disclaimer:</strong> This assistant provides <em>factual information only</em>,
    sourced exclusively from official HDFC Mutual Fund, SEBI, and AMFI websites.
    It does <strong>not</strong> provide investment advice, return predictions, or portfolio
    recommendations. For investment decisions, consult a SEBI-registered investment advisor.
    No PII (PAN, Aadhaar, phone, email, account numbers) is collected or stored.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Scope Note ──────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Scope: AMC, Schemes & Sources"):
    st.markdown(
        """
        **Platform:** Groww  
        **AMC:** HDFC Mutual Fund  
        **Schemes covered:**
        - HDFC Top 100 Fund *(Large Cap)*
        - HDFC Flexi Cap Fund *(Flexi Cap)*
        - HDFC ELSS Tax Saver *(ELSS, 3-year lock-in, 80C)*
        - HDFC Mid-Cap Opportunities Fund *(Mid Cap)*

        **Sources:** 22 official pages from HDFC AMC, SEBI, and AMFI  
        *(see `data/sources.csv` for the full list)*  
        **Last updated from sources:** December 2024
        """
    )

st.markdown("---")

# ── Example Questions ──────────────────────────────────────────────────────────
st.markdown("### 💡 Try these example questions:")
st.caption("*Facts-only. No investment advice.*")

EXAMPLES = [
    "What is the expense ratio of HDFC Top 100 Fund?",
    "What is the lock-in period for ELSS funds?",
    "Minimum SIP for HDFC ELSS Tax Saver?",
]

cols = st.columns(3)
clicked_example = None
for col, example in zip(cols, EXAMPLES):
    with col:
        if st.button(example, use_container_width=True, key=f"btn_{example[:20]}"):
            clicked_example = example

# ── Query Input ─────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")
query_input = st.text_input(
    label="Type your factual question about HDFC Mutual Fund schemes:",
    value=clicked_example or "",
    placeholder="e.g. What is the exit load for HDFC Flexi Cap Fund?",
    max_chars=300,
    key="query_input",
)

submit = st.button("Get Answer", type="primary", use_container_width=True)

# ── Initialize RAG Engine (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base from official sources...")
def load_rag_engine() -> RAGEngine:
    return RAGEngine()


try:
    rag = load_rag_engine()
except FileNotFoundError as exc:
    st.error(f"❌ Corpus not found: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"❌ Failed to initialize RAG engine: {exc}")
    st.stop()

# ── Process Query ───────────────────────────────────────────────────────────────
active_query = query_input.strip() if submit else (clicked_example or "")

if active_query:
    with st.spinner("🔍 Searching official AMC/SEBI/AMFI sources..."):
        result = rag.answer(active_query)

    st.markdown("---")
    st.markdown(f"**Your question:** *{active_query}*")

    if result["type"] == "answer":
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="source-box">
            📎 <strong>Source:</strong>
            <a href="{result['source_url']}" target="_blank">{result['source_name']}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("*Last updated from sources: December 2024*")

    elif result["type"] == "refusal":
        st.markdown(
            f'<div class="refusal-box">🚫 {result["message"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"📚 **Learn more (educational):** [{result['link']}]({result['link']})"
        )

    else:  # error
        st.error(f"⚠️ {result['message']}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Data sourced from official HDFC Mutual Fund, SEBI, and AMFI websites. "
    "Mutual fund investments are subject to market risks. "
    "Read all scheme-related documents carefully before investing.</small>",
    unsafe_allow_html=True,
)
