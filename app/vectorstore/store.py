import faiss
import pickle
import os

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.texts[idx], float(dist)))
        return results


    def save(self, path="data"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    @classmethod
    def load(cls, dim, path="data"):
        index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/texts.pkl", "rb") as f:
            texts = pickle.load(f)

        store = cls(dim)
        store.index = index
        store.texts = texts
        return store
