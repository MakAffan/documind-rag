import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        # L2 distance index
        self.index = faiss.IndexFlatL2(dim)
        self.texts: list[str] = []

    def add(self, embeddings: list[list[float]], texts: list[str]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: list[list[float]], k: int = 3) -> list[str]:
        query_vector = np.array(query_embedding).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return [self.texts[i] for i in indices[0]]
    
