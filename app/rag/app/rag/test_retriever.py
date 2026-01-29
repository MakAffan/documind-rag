print("TEST RETRIEVER STARTED")

from app.vectorstore.store import VectorStore
from app.embeddings.embedder import embed_text
from app.rag.retriever import retrieve

def main():
    texts = [
        "Python is a programming language",
        "Cats are animals",
        "The sky is blue",
    ]

    embeddings = embed_text(texts)

    store = VectorStore(len(embeddings[0]))
    store.add(embeddings, texts)

    question = "What is Python?"
    results = retrieve(question, store, k=1)

    print("Retrieved:", results)

if __name__ == "__main__":
    main()
