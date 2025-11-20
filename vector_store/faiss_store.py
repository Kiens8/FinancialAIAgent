import faiss
import numpy as np

class FaissVectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.ids = []

    def add(self, id, vector, text):
        self.index.add(np.array([vector]).astype('float32'))
        self.texts.append(text)
        self.ids.append(id)

    def query(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype('float32'), k)
        results = []
        for idx in I[0]:
            if idx != -1:
                results.append(self.texts[idx])
        return results
