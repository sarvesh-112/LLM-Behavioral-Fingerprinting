from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceEncoder:
    """
    Sentence embedding encoder for behavioral analysis.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encode a list of texts into embeddings.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    @staticmethod
    def cosine_similarity(vec1, vec2):
        return float(np.dot(vec1, vec2))
