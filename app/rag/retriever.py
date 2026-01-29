from app.embeddings.embedder import embed_text
from app.vectorstore.store import VectorStore

def retrieve(
    question: str,
    store: VectorStore,
    k: int = 3
) -> list[str]:
    """
    Convert the question into an embedding and retrieve
    top-k relevant chunks from the vector store.
    """

    if store is None:
        raise ValueError("Vector store is not initialized")

    # 1. Embed the user question
    query_embedding = embed_text([question])

    # 2. Search the vector store
    results = store.search(query_embedding, k=k)

    return results
