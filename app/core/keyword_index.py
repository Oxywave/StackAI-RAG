# BM25 keyword search

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from app.config import settings
from app.core.models import SearchResult
from app.services.ingestion import Chunk

# BM25 term-frequency saturation parameter — larger values make repeated term matches matter more
K1 = 1.5

# BM25 length-normalization parameter – 0 = no length penalty, 1 = full normalization
B = 0.75

# Hardcoded NLTK English stopword list. 
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "should've", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
    "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
}

# Tokenize — lowercase, remove punctuation, split, drop stopwords and 1-char tokens
def tokenize(text: str) -> List[str]:

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# In-memory BM25 index over document chunks
class BM25Index:


    def __init__(self):

        # Stored chunk metadata in index order
        self._chunks: List[Chunk] = []

        # Per-chunk term frequency maps: one dict per chunk
        self._tf: List[Dict[str, int]] = []   # term freq per chunk

        # Document frequency per term: number of chunks containing the term
        self._df: Dict[str, int] = {}         # number of chunks containing each term

        # Chunk lengths measured in tokens
        self._doc_lengths: List[int] = []     # token count per chunk

        # Average chunk length across the whole index
        self._avgdl: float = 0.0

        self._index_path = os.path.join(settings.storage_dir, "keyword_index.json")
        os.makedirs(settings.storage_dir, exist_ok=True)
        self._load()


    # Add new chunks into the BM25 index
    def add(self, chunks: List[Chunk]) -> None:
   
        if not chunks:
            return


        for chunk in chunks:
            # Tokenize chunk text into normalized search terms
            tokens = tokenize(chunk.text)

            # Count how many times each term appears in this chunk
            tf = dict(Counter(tokens))

            # Store chunk metadata and per-chunk term statistics
            self._chunks.append(chunk)
            self._tf.append(tf)
            self._doc_lengths.append(len(tokens))

            # Update document frequency once per unique term in this chunk
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        # Recompute average document length after adding new chunks
        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)
        self._save()


    # Score all indexed chunks against a query using BM25
    def search(self, query: str, top_k: int) -> List[SearchResult]:
        # Empty index -> no results
        if not self._chunks:
            return []

        # Tokenize query in the same way as document chunks
        query_tokens = tokenize(query)

        # Query with no meaningful tokens cannot retrieve anything
        if not query_tokens:
            return []
        
        n = len(self._chunks)

        # One BM25 score per chunk, initialized to zero
        scores = [0.0] * n

        # Process each query term independently and add its BM25 contribution
        for term in query_tokens:
            if term not in self._df:
                continue
            
            # Compute inverse document frequency for this term
            idf = self._idf(term, n)

            # Score this term against every chunk
            for i, tf_map in enumerate(self._tf):
                # Term frequency of current term in this chunk
                tf = tf_map.get(term, 0)

                # No contribution if chunk does not contain the term
                if tf == 0:
                    continue

                dl = self._doc_lengths[i]    # Length of current chunk in tokens
                
                # BM25 term-frequency component with length normalization
                normalised_tf = (tf * (K1 + 1)) / (
                    tf + K1 * (1 - B + B * dl / self._avgdl)
                )

                # Add this term's BM25 contribution to chunk score
                scores[i] += idf * normalised_tf

        # Rank chunks by descending BM25 score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Convert top-scoring chunks into SearchResult objects
        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:      # Stop once scores become non-positive
                break
            results.append(SearchResult(chunk=self._chunks[idx], score=score))

        return results


    # Wipe whole index from memory and disk
    def clear(self) -> None:
        """Wipe the index from memory and disk."""
        self._chunks = []
        self._tf = []
        self._df = {}
        self._doc_lengths = []
        self._avgdl = 0.0
        if os.path.exists(self._index_path):
            os.remove(self._index_path)



    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # Compute BM25 inverse document frequency for one term
    def _idf(self, term: str, n: int) -> float:
        
        # Number of chunks containing this term
        df = self._df.get(term, 0)

        # +0.5 is smoothing term and +1 is to ensure positive
        # BM25 IDF formula: rarer terms receive higher weight
        return math.log((n - df + 0.5) / (df + 0.5) + 1)    


     # Save all BM25 state to disk as JSON
    def _save(self) -> None:
        from dataclasses import asdict
        data = {
            "k1": K1,
            "b": B,
            "avgdl": self._avgdl,
            "doc_lengths": self._doc_lengths,
            "tf": self._tf,
            "df": self._df,
            "chunks": [asdict(c) for c in self._chunks],
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f)


    # Reload BM25 state from disk if previously saved
    def _load(self) -> None:
        if not os.path.exists(self._index_path):
            return

        with open(self._index_path) as f:
            data = json.load(f)

        self._avgdl = data["avgdl"]
        self._doc_lengths = data["doc_lengths"]
        self._tf = data["tf"]
        self._df = data["df"]
        self._chunks = [Chunk(**c) for c in data["chunks"]]


_index: Optional[BM25Index] = None

# Return the shared BM25 index, creating it on first use
def get_keyword_index() -> BM25Index:

    global _index
    if _index is None:
        _index = BM25Index()
    return _index
