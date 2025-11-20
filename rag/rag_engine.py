from embeddings.nomic_embedder import NomicEmbedder
from vector_store.faiss_store import FaissVectorStore
from utils.chunker import chunk_text

class RAGEngine:
    def __init__(self, dim=768):
        self.embedder = NomicEmbedder()
        self.db = FaissVectorStore(dim)

    def add_document(self, text):
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            vec = self.embedder.embed(c)
            self.db.add(i, vec, c)

    def query(self, question, k=5):
        vec = self.embedder.embed(question)
        return self.db.query(vec, k)
