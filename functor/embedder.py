"""
embedder.py

Singleton SBERT embedder using the same model as Democritus (all-MiniLM-L6-v2).
Keeping the model consistent means all embeddings live in the same 384-dim space,
which is a requirement for the coend/end computation to be meaningful.
"""

from __future__ import annotations

from typing import List, Union
import numpy as np


class SharedEmbedder:
    """
    Lazy-loaded, singleton SBERT embedder.

    Uses all-MiniLM-L6-v2 (384-dim), same model as Democritus's manifold_builder.
    The singleton ensures we load the model once even when CausalFunctor is
    instantiated multiple times across domains.
    """

    _instance: "SharedEmbedder | None" = None
    _model = None

    @classmethod
    def get(cls) -> "SharedEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Democritus_OpenAI"))
            from sentence_transformers import SentenceTransformer
            print("[Embedder] Loading all-MiniLM-L6-v2…")
            self.__class__._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Embedder] Ready.")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized 384-dim embeddings.
        Returns shape (N, 384).
        """
        self._load()
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def cosine_similarity_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Fast cosine similarity between two embedding matrices.
        a: (M, D), b: (N, D)  →  returns (M, N)
        Assumes L2-normalized embeddings (normalize=True above).
        """
        return (a @ b.T).astype(np.float32)
