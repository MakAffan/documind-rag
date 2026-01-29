print("TEST FILE STARTED")

from app.vectorstore.store import VectorStore
from app.embeddings.embedder import embed_text

def main():
    texts = [
        "Python is a programming language",
        "Cats are animals",
        "The sky is blue",
    ]

    embeddings = embed_text(texts)

    store = VectorStore(len(embeddings[0]))
    store.add(embeddings, texts)

    query = "What is Python?"
    query_embedding = embed_text([query])

    results = store.search(query_embedding, k=1)
    print(results)

if __name__ == "__main__":
    main()
