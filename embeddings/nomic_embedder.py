import requests
import numpy as np

class NomicEmbedder:
    def __init__(self, model="nomic-embed-text:latest"):
        self.model = model

    def embed(self, text: str):
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return np.array(res.json()["embedding"], dtype=np.float32)
