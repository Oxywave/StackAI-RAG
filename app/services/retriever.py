"""
Hybrid retrieval — combines semantic search and BM25 keyword search using
Reciprocal Rank Fusion.

Running two independent searches and merging their ranked lists tends to
outperform either approach alone. Semantic search handles paraphrasing and
conceptual similarity well but can miss exact terminology. BM25 is strong on
precise keyword matches but blind to meaning. Fusing both covers their
respective blind spots.

Reciprocal Rank Fusion (RRF) is used to merge the two lists. It avoids
normalising raw scores across incompatible scales (cosine vs BM25) by working
only on rank positions. The contribution of each result is 1 / (k + rank),
where k=60 dampens the outsized influence of the very top result.

Formula for a chunk appearing at rank r_s in semantic and r_k in keyword:
  rrf_score = 1 / (60 + r_s)  +  1 / (60 + r_k)

Chunks that appear in only one list still get a partial score.
Final list is sorted by RRF score and trimmed to top_k.
"""

from typing import List

from app.config import settings
from app.core.keyword_index import get_keyword_index
from app.core.models import SearchResult
from app.core.vector_store import get_vector_store
from app.services.query_processor import Intent, ProcessedQuery


RRF_K = 60


def retrieve(processed: ProcessedQuery, top_k: int = None) -> List[SearchResult]:
    """
    Run hybrid retrieval for a processed query.

    Returns an empty list if the intent is not KNOWLEDGE.
    Otherwise searches both the vector store and BM25 index, fuses the
    results with RRF, and returns the top_k chunks.
    """
    if processed.intent != Intent.KNOWLEDGE:
        return []

    if top_k is None:
        top_k = settings.top_k

    query = processed.rewritten
    fetch_k = top_k * 2

    semantic_results = get_vector_store().search(query, fetch_k)

    # Similarity gate — if the best semantic match falls below the configured
    # threshold the query is off-topic for this knowledge base. Return nothing
    # immediately so the generator can issue an "insufficient evidence" response
    # rather than fabricating an answer from loosely related chunks.
    if not semantic_results or semantic_results[0].score < settings.similarity_threshold:
        return []

    keyword_results = get_keyword_index().search(query, fetch_k)

    return _rrf_merge(semantic_results, keyword_results, top_k)


def _rrf_merge(
    semantic: List[SearchResult],
    keyword: List[SearchResult],
    top_k: int,
) -> List[SearchResult]:
    """
    Merge two ranked result lists with Reciprocal Rank Fusion.

    Each chunk is identified by (source, chunk_index). Scores from both lists
    are accumulated then the combined dict is sorted and trimmed to top_k.
    """
    scores = {}
    best_result = {}

    for rank, result in enumerate(semantic):
        key = _chunk_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        best_result[key] = result

    for rank, result in enumerate(keyword):
        key = _chunk_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        if key not in best_result:
            best_result[key] = result

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    merged = []
    for key in ranked_keys[:top_k]:
        result = best_result[key]
        merged.append(SearchResult(chunk=result.chunk, score=scores[key]))

    return merged


def _chunk_key(result: SearchResult) -> tuple:
    """Stable identity key for a chunk — source filename + position index."""
    return (result.chunk.source, result.chunk.chunk_index)
