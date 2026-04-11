# In-memory vector store for semantic retrieval - cosine simlaity search

import json
import os
from dataclasses import asdict
from typing import List, Optional
import numpy as np
from mistralai import Mistral

from app.config import settings
from app.core.models import SearchResult
from app.services.ingestion import Chunk


EMBED_MODEL = "mistral-embed"
# Embedding dimensionality from Mistral
EMBED_DIM = 1024


class VectorStore:
    def __init__(self):
        # Embedding API client
        self._client = Mistral(api_key=settings.mistral_api_key)
        
        # Matrix of stored embeddings; one row per chunk
        self._vectors: Optional[np.ndarray] = None

        self._chunks: List[Chunk] = []

        self._vectors_path = os.path.join(settings.storage_dir, "vectors.npy")
        self._chunks_path = os.path.join(settings.storage_dir, "chunks.json")

        os.makedirs(settings.storage_dir, exist_ok=True)
        self._load()



    # # Add new document chunks into the vector store
    def add(self, chunks: List[Chunk]) -> None:
        
        # Nothing to add
        if not chunks:
            return

        # Extract text from chunks 
        texts = [c.text for c in chunks]

        # Mistral Embed the new chunk texts into vectors
        new_vectors = self._embed(texts)

        # If store is empty, initialize the matrix
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            # Otherwise append new rows to existing vector matrix
            self._vectors = np.vstack([self._vectors, new_vectors])


        self._chunks.extend(chunks)
        self._save()


    # Semantic search by cosine similarity against query embedding —> return top k most relevant chunks with scores
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        
        # If store empty, nothing can be retrieved
        if self._vectors is None or len(self._chunks) == 0:
            return []

        # Embed the query into a single vector
        query_vec = self._embed([query])[0]

        # Cosine similarity becomes dot product because vectors are L2-normalized
        scores = self._vectors @ query_vec      # @ is dot product

        # Get indices of the top k chunks in descending order
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return ranked chunk + score pairs — wrapping in SearchResult
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]


     # Wipe everything in the store  
    def clear(self) -> None:

        # Reset in-memory state
        self._vectors = None
        self._chunks = []

        # Remove persisted files if they exist
        for path in (self._vectors_path, self._chunks_path):
            if os.path.exists(path):
                os.remove(path)



    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


     # Embed text in batches and return a normalized matrix
    def _embed(self, texts: List[str]) -> np.ndarray:

        all_embeddings = []
        batch_size = 32     # Batch size for embedding API calls

        # Process texts in batches to avoid one large API request
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Request embeddings for this batch of texts
            response = self._client.embeddings.create(
                model=EMBED_MODEL,
                inputs=batch,
            )

            # Convert returned embeddings into float32 numpy array
            batch_vecs = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )

            # Store batch result for later concatenation
            all_embeddings.append(batch_vecs)

        # Every vector that enters the store is normalised
        return _l2_normalise(np.vstack(all_embeddings))



    def _save(self) -> None:
        
        # Save vector matrix only if it exists
        if self._vectors is not None:
            np.save(self._vectors_path, self._vectors)

        # chunks are saved as list of dicts in json file
        with open(self._chunks_path, "w") as f:
            json.dump([asdict(c) for c in self._chunks], f)


    # Reload vectors and chunks from disk if saved files exist
    def _load(self) -> None:

        # Need both files to restore a consistent store
        if not os.path.exists(self._vectors_path):
            return
        if not os.path.exists(self._chunks_path):
            return

        # Load embedding matrix from .npy file
        self._vectors = np.load(self._vectors_path)

        # Load chunk metadata and rebuild Chunk objects
        with open(self._chunks_path) as f:
            raw = json.load(f)
        self._chunks = [Chunk(**item) for item in raw]

# Module-level singleton store shared across the app
_store: Optional[VectorStore] = None

# Return the shared vector store
def get_vector_store() -> VectorStore:
    
    # Create vector store on first call
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

# Row-wise L2 normalization
# divide each vector by its own length so cosine similarity = dot product
def _l2_normalise(matrix: np.ndarray) -> np.ndarray:


     # Compute L2 norm of each row vector
    norms = np.linalg.norm(matrix, axis = 1, keepdims = True)
    
    # Avoid division by zero: treat zero norms as 1 so those rows stay unchanged
    norms = np.where(norms == 0, 1.0, norms)
    
    # Normalize each row vector to unit length
    return matrix / norms
