import streamlit as st
import os
from app.embeddings.embedder import embed_text
from app.vectorstore.store import VectorStore
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer
from app.config import CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, LLM_MODEL, CLOUDFLARE_AI_BASE_URL, HEADERS

st.set_page_config(page_title="DocuMind", layout="centered")
st.title("📄 DocuMind – AI Document Q&A")
st.markdown("Upload text and ask questions powered by Cloudflare RAG.")

# -----------------------
# Global in-memory vector store
# -----------------------
if "vector_store" not in st.session_state:
    if os.path.exists("data/index.faiss"):
        st.session_state.vector_store = VectorStore.load(dim=768)
    else:
        st.session_state.vector_store = None

# -----------------------
# Load Document
# -----------------------
st.header("1️⃣ Load Document")

doc_text = st.text_area(
    "Paste your document text here:",
    height=200
)

if st.button("Load Document"):
    if not doc_text.strip():
        st.error("Document text cannot be empty")
    else:
        with st.spinner("Processing document..."):
            try:
                # Treat whole text as one chunk (for simplicity)
                chunks = [doc_text]

                embeddings = embed_text(chunks)
                store = VectorStore(len(embeddings[0]))
                store.add(embeddings, chunks)
                store.save()

                st.session_state.vector_store = store
                st.success(f"Document loaded successfully ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Error loading document: {e}")

if st.button("Clear Document"):
    st.session_state.vector_store = None
    if os.path.exists("data"):
        import shutil
        shutil.rmtree("data")
    st.success("Document cleared")

# -----------------------
# Ask Question
# -----------------------
st.header("2️⃣ Ask Question")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    store = st.session_state.vector_store
    if store is None:
        st.error("Please load a document first!")
    elif not question.strip():
        st.error("Question cannot be empty")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Retrieve top-k relevant chunks
                docs = retrieve(question, store, k=3)
                context = "\n".join([r[0] for r in results])

                # Generate answer using Cloudflare LLM
                answer = generate_answer(context, question)

                # Display results
                st.subheader("💡 Answer")
                st.write(answer)

                st.subheader("📚 Sources")
                for text, score in results:
                    st.markdown(f"- {text}  \n  _Similarity score: {score:.4f}_")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
