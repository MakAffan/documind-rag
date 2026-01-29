from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.embeddings.embedder import embed_text
from app.vectorstore.store import VectorStore
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer

app = FastAPI(title="DocuMind - Cloudflare RAG")

# In-memory vector store (for now)
vector_store = None


class LoadRequest(BaseModel):
    text: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/load")
def load_document(payload: LoadRequest):
    global vector_store

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # For now, treat the whole text as one chunk
    chunks = [text]

    embeddings = embed_text(chunks)
    vector_store = VectorStore(len(embeddings[0]))
    vector_store.add(embeddings, chunks)

    return {"status": "document loaded", "chunks": len(chunks)}


@app.post("/ask")
def ask_question(payload: AskRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No document loaded")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    docs = retrieve(question, vector_store, k=payload.top_k)
    context = "\n".join(docs)

    answer = generate_answer(context, question)

    return {
        "question": question,
        "answer": answer,
        "sources": docs,
    }
