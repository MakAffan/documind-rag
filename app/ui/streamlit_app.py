import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocuMind", layout="centered")

st.title("📄 DocuMind – AI Document Q&A")

st.markdown("Upload text and ask questions powered by Cloudflare RAG.")

# -----------------------
# Load document
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
        with st.spinner("Loading document..."):
            res = requests.post(
                f"{API_URL}/load",
                json={"text": doc_text},
                timeout=60,
            )

        if res.status_code == 200:
            st.success("Document loaded successfully")
        else:
            st.error(res.text)

# -----------------------
# Ask question
# -----------------------
st.header("2️⃣ Ask Question")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.error("Question cannot be empty")
    else:
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": 3},
                timeout=60,
            )

        if res.status_code == 200:
            data = res.json()
            st.subheader("💡 Answer")
            st.write(data["answer"])

            st.subheader("📚 Sources")
            for src in data["sources"]:
                st.markdown(f"- {src}")
        else:
            st.error(res.text)
