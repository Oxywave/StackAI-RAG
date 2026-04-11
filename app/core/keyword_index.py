"""
BM25 keyword index

BM25 (Best Match 25) scores how relevant a document is to a query based on the
words they share. It improves on simple word counting in two ways:

  1. Term frequency saturation — finding a word 10 times doesn't make a document
     10x more relevant. BM25 applies a curve that flattens at higher counts (k1).

  2. Length normalisation — long documents contain more words by definition, so
     matching a term there means less than matching it in a short focused chunk (b).

IDF (inverse document frequency) gives rare words more weight than common ones.
A word in every document tells you nothing; a word in two documents is a strong signal.

Formula per query term t, document d:
  score += IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

  where tf    = times t appears in d
        dl    = length of d in tokens
        avgdl = average document length across all indexed chunks
        IDF   = log((N - df + 0.5) / (df + 0.5) + 1)
        N     = total chunks indexed
        df    = number of chunks containing t

k1 = 1.5  — term frequency saturation, standard range is 1.2–2.0
b  = 0.75 — length normalisation, 0 = off, 1 = full
"""

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from app.config import settings
from app.core.models import SearchResult
from app.services.ingestion import Chunk


K1 = 1.5
B = 0.75

# Hardcoded NLTK English stopword list. NO externl libraries are used here

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


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self):
        self._chunks: List[Chunk] = []
        self._tf: List[Dict[str, int]] = []   # term freq per chunk
        self._df: Dict[str, int] = {}         # number of chunks containing each term
        self._doc_lengths: List[int] = []     # token count per chunk
        self._avgdl: float = 0.0

        self._index_path = os.path.join(settings.storage_dir, "keyword_index.json")
        os.makedirs(settings.storage_dir, exist_ok=True)
        self._load()

    def add(self, chunks: List[Chunk]) -> None:
        """Index a list of chunks."""
        if not chunks:
            return

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = dict(Counter(tokens))

            self._chunks.append(chunk)
            self._tf.append(tf)
            self._doc_lengths.append(len(tokens))

            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)
        self._save()

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        """Score every indexed chunk against the query and return the top-k."""
        if not self._chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n = len(self._chunks)
        scores = [0.0] * n

        for term in query_tokens:
            if term not in self._df:
                continue

            idf = self._idf(term, n)

            for i, tf_map in enumerate(self._tf):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue

                dl = self._doc_lengths[i]
                normalised_tf = (tf * (K1 + 1)) / (
                    tf + K1 * (1 - B + B * dl / self._avgdl)
                )
                scores[i] += idf * normalised_tf

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break
            results.append(SearchResult(chunk=self._chunks[idx], score=score))

        return results

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

    def _idf(self, term: str, n: int) -> float:
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

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


def get_keyword_index() -> BM25Index:
    """Return the shared BM25Index instance, creating it on first call."""
    global _index
    if _index is None:
        _index = BM25Index()
    return _index
