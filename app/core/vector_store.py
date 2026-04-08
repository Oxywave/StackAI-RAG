"""
In-memory vector store with cosine similarity search.

Design notes
------------
Embeddings are stored as a numpy matrix — one row per chunk — so similarity
against a query vector is a single matrix-vector multiply followed by a norm
division. This is fast enough for thousands of chunks without any external
search infrastructure.

Persistence is handled by saving the matrix and metadata to two files in the
storage directory: vectors.npy (the raw float32 matrix) and chunks.json (the
chunk metadata). On startup, if those files exist, the store reloads them
automatically. This means ingested documents survive a server restart.

We use Mistral's mistral-embed model to produce 1024-dimensional embeddings.
All vectors are L2-normalised before storage so the cosine similarity reduces
to a plain dot product — one fewer division at query time.

Thread safety: this implementation is single-threaded. For a production system
you'd wrap mutations in a lock, but that's out of scope here.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
from mistralai import Mistral

from app.config import settings
from app.services.ingestion import Chunk


EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, 0-1


class VectorStore:
    def __init__(self):
        self._client = Mistral(api_key=settings.mistral_api_key)
        self._vectors: Optional[np.ndarray] = None  # shape (n, EMBED_DIM)
        self._chunks: List[Chunk] = []

        self._vectors_path = os.path.join(settings.storage_dir, "vectors.npy")
        self._chunks_path = os.path.join(settings.storage_dir, "chunks.json")

        os.makedirs(settings.storage_dir, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk]) -> None:
        """Embed a list of chunks and add them to the store."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vectors = self._embed(texts)  # (n, EMBED_DIM), already normalised

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        self._chunks.extend(chunks)
        self._save()

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        """Return the top-k chunks most similar to the query."""
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query_vec = self._embed([query])[0]  # (EMBED_DIM,)
        scores = self._vectors @ query_vec   # dot product = cosine sim (normalised)

        # grab top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def clear(self) -> None:
        """Wipe everything — vectors and chunks — from memory and disk."""
        self._vectors = None
        self._chunks = []
        for path in (self._vectors_path, self._chunks_path):
            if os.path.exists(path):
                os.remove(path)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Call the Mistral embedding API and return an L2-normalised matrix.
        Batches of up to 32 texts at a time to stay within API limits.
        """
        all_embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=EMBED_MODEL,
                inputs=batch,
            )
            batch_vecs = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            all_embeddings.append(batch_vecs)

        matrix = np.vstack(all_embeddings)
        return _l2_normalise(matrix)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

        with open(self._chunks_path, "w") as f:
            json.dump([asdict(c) for c in self._chunks], f)

    def _load(self) -> None:
        if not os.path.exists(self._vectors_path):
            return
        if not os.path.exists(self._chunks_path):
            return

        self._vectors = np.load(self._vectors_path)

        with open(self._chunks_path) as f:
            raw = json.load(f)
        self._chunks = [Chunk(**item) for item in raw]


# ------------------------------------------------------------------
# module-level singleton
# ------------------------------------------------------------------

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the shared VectorStore instance, creating it on first call."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm. Rows with zero norm are left as-is."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms
