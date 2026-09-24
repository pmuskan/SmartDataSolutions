import json

import streamlit as st

from src.config import load_config
from src.generate import RAGGenerator
from src.index import VectorAndSparseIndex
from src.ingest.pipeline import run_ingestion_pipeline
from src.llm import LLMClient
from src.retrieve import HybridRetriever
from src.schemas import Chunk

st.set_page_config(
    page_title="Apple Q3 2022 Form 10-Q Multimodal RAG",
    page_icon="🍎",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1d1d1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #86868b;
        margin-bottom: 2rem;
    }
    .evidence-card {
        background-color: #f5f5f7;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #d2d2d7;
        margin-bottom: 1rem;
    }
    .modality-badge {
        background-color: #0071e3;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_system():
    cfg = load_config()
    processed_chunks_file = cfg.resolve_path(cfg.processed_dir) / "chunks.jsonl"
    if not processed_chunks_file.exists():
        chunks = run_ingestion_pipeline()
    else:
        chunks = []
        with open(processed_chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))

    index = VectorAndSparseIndex()
    index.load_index(chunks)
    retriever = HybridRetriever(index)
    llm_client = LLMClient()
    generator = RAGGenerator(llm_client)

    return retriever, generator


retriever, generator = init_rag_system()

st.markdown('<div class="main-header">🍎 Apple Inc. Q3 2022 Form 10-Q Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Q&A engine over text, financial tables, and figures with strict grounding & page citations.</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Active Strategy")
st.sidebar.info(
    "🎯 **Selected Best Strategy:**\n\n"
    "• **Chunking:** Section-Aware + Single Unit Tables\n"
    "• **Retrieval:** Hybrid (Chroma Dense + BM25 RRF)\n"
    "• **Routing:** Intent-based Query Router\n"
    "• **Reranking:** Cross-Encoder (`ms-marco-MiniLM`)\n"
    "• **Grounding:** Strict Abstention & Citations"
)
top_k = st.sidebar.slider("Top-K Context Chunks", min_value=1, max_value=10, value=5)
retrieval_mode = "hybrid+rerank"
use_router = True

# Pre-set query buttons
st.sidebar.header("Example Queries")
example_queries = [
    "What were total net sales for the three months ended June 25, 2022?",
    "What were Services net sales and YoY growth rate?",
    "What was Products gross margin % in Q3 2022 vs Q3 2021, and why did it fall?",
    "Which segment had the largest YoY net sales decline in Q3 2022, and why?",
    "What image appears on the cover page?",
    "What was Apple's Q3 2023 revenue?"
]

selected_example = None
for eq in example_queries:
    if st.sidebar.button(eq, use_container_width=True):
        selected_example = eq

# Main layout: Query input + Evidence panel
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your query:", value=selected_example if selected_example else "")

    if st.button("Submit Query", type="primary", use_container_width=True) or user_query:
        if user_query:
            with st.spinner("Retrieving evidence & generating grounded answer..."):
                retrieved_chunks = retriever.retrieve(
                    query=user_query,
                    top_k=top_k,
                    mode="hybrid+rerank",
                    use_router=True
                )
                answer = generator.generate_answer(query=user_query, retrieved_chunks=retrieved_chunks)

                st.session_state["current_answer"] = answer
                st.session_state["retrieved_chunks"] = retrieved_chunks

    if "current_answer" in st.session_state:
        answer = st.session_state["current_answer"]

        if answer.abstained:
            st.warning(f"⚠️ **Abstention Response:** {answer.text}")
        else:
            st.success("### Answer:")
            st.markdown(answer.text)

            if answer.citations:
                st.markdown("#### 📌 Citations:")
                for c in answer.citations:
                    st.markdown(f"- **Page {c.page}**: `{c.section_or_table_title}`")

            if answer.tool_calls:
                st.markdown("#### 🧮 Calculator Tool Traces:")
                for tc in answer.tool_calls:
                    st.info(f"Formula: `{tc.expression}` → Result: **{tc.result}**")

            st.caption(f"Latency: {answer.latency_ms:.1f} ms | Chunks used: {len(answer.used_chunk_ids)}")

with col_right:
    st.subheader("Retrieved Evidence Panel")
    if "retrieved_chunks" in st.session_state:
        chunks = st.session_state["retrieved_chunks"]
        for idx, rc in enumerate(chunks):
            c = rc.chunk
            with st.expander(f"Chunk [{idx+1}] - Page {c.page} ({c.type.upper()}) | Score: {rc.score:.3f}"):
                st.markdown(f"**Section Path:** `{c.section_path}`")
                if c.title:
                    st.markdown(f"**Title:** {c.title}")
                if c.units:
                    st.markdown(f"**Units:** {c.units}")
                st.text_area(f"Chunk Payload [{c.id}]", value=c.text, height=180, key=f"chunk_text_{idx}")
    else:
        st.info("Retrieved context chunks and scores will appear here after submitting a query.")
