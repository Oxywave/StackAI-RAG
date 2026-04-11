
# Hybrid retrieval - semantic search + BM25 keyword search using RRF

from typing import List

from app.config import settings
from app.core.keyword_index import get_keyword_index
from app.core.models import SearchResult
from app.core.vector_store import get_vector_store
from app.services.query_processor import Intent, ProcessedQuery

# damping constant for RRF
RRF_K = 60


# Hybrid retrieval — find most relevant chunk to answer a query
# takes in processed query, retrieve top k relevant chunks
def retrieve(processed: ProcessedQuery, top_k: int) -> List[SearchResult]:

    # Only run retrieval if query is Knowledge 
    if processed.intent != Intent.KNOWLEDGE:
        return []

    # use rewriteen / processed query, not raw input from user
    query = processed.rewritten 
    # fetch twice more than needed for RRK merging candidate
    fetch_k = top_k * 2

    # run both semantic + bm25 search
    semantic_results = get_vector_store().search(query, fetch_k)
    keyword_results = get_keyword_index().search(query, fetch_k)

    # merg using RRF
    return _rrf_merge(semantic_results, keyword_results, top_k)



# Reciprocal Rank Fusion 
# 

def _rrf_merge(semantic: List[SearchResult], keyword: List[SearchResult], top_k: int,) -> List[SearchResult]:
    # dict to store scores by chunk key
    scores = {}

    # dict to store best search result for each chunk key 
    best_result = {}

    # scores semantic results 
    for rank, result in enumerate(semantic):

        # assign unique id to each retrieved chunk 
        key = _chunk_key(result)

        # add to the existing score for this chunk if exist
        # using rank position only - higher ranked restuls scores higher
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        # stores the search result with highest semantic score for each chunk key
        best_result[key] = result

    # scores keyword results
    for rank, result in enumerate(keyword):

        # same as above
        key = _chunk_key(result)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        # avoid duplicate results 
        if key not in best_result:
            best_result[key] = result

    # sort by total score
    ranked_keys = sorted(scores, key = lambda k: scores[k], reverse = True)


    merged = []

    # build final output - take top few 
    for key in ranked_keys[:top_k]:
        result = best_result[key]
        merged.append(SearchResult(chunk = result.chunk, score=scores[key]))

    return merged


# create unique chunk key — source filename + position index
def _chunk_key(result: SearchResult) -> tuple:
    return (result.chunk.source, result.chunk.chunk_index)
