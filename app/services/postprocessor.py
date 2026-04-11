"""
Post-processing — filters and deduplicates retrieval results before they
reach the answer generator.

Two passes run in sequence:

1. Threshold guard — drops any chunk whose RRF score falls below a minimum
   cutoff. RRF scores are not on a fixed scale, so the cutoff is derived from
   the fusion constant: 1 / (RRF_K + top_k). Any chunk that would rank below
   top_k in both retrieval systems scores lower than this and is discarded.
   This prevents weak, loosely-related passages from polluting the answer.

2. Near-duplicate removal — compares every remaining pair of chunks using
   Jaccard similarity on their token sets. If two chunks share more than 80%
   of their tokens, the lower-ranked one is dropped. This handles PDFs that
   repeat the same content across sections (summaries, headers, boilerplate)
   without sending redundant context to the generator.

The result is a short, high-confidence, non-redundant list of chunks ready
for answer synthesis.
"""

from typing import List

from app.core.models import SearchResult
from app.core.keyword_index import tokenize


RRF_K = 60
DUPLICATE_THRESHOLD = 0.80


def postprocess(results: List[SearchResult], top_k: int) -> List[SearchResult]:
    """
    Filter and deduplicate a ranked list of retrieval results.

    Applies the score threshold first, then removes near-duplicates.
    Returns at most top_k results.
    """
    if not results:
        return []

    min_score = 1.0 / (RRF_K + top_k)
    filtered = [r for r in results if r.score >= min_score]
    deduped = _remove_near_duplicates(filtered)

    return deduped[:top_k]



def _remove_near_duplicates(results: List[SearchResult]) -> List[SearchResult]:
    """
    Remove chunks that are near-copies of a higher-ranked chunk.

    Uses Jaccard similarity on token sets. A chunk is considered a duplicate
    if it shares more than DUPLICATE_THRESHOLD of its tokens with any
    already-accepted chunk. The input list is assumed to be sorted by score
    descending — higher-ranked chunks are always kept over lower-ranked ones.
    """
    kept = []
    kept_token_sets = []

    for result in results:
        tokens = set(tokenize(result.chunk.text))

        is_duplicate = False
        for existing_tokens in kept_token_sets:
            if _jaccard(tokens, existing_tokens) >= DUPLICATE_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(result)
            kept_token_sets.append(tokens)

    return kept


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0
