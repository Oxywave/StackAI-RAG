# Post-processing retrieval before anwwer generation
# Remove weak chunks if below RRF threshold + remove redundant chunks 

from typing import List
from app.core.models import SearchResult
from app.core.keyword_index import tokenize

# Same damping constant used in RRF
RRF_K = 60

# If >= 80% similar (Jaccard Similarity) —> considered duplicate
DUPLICATE_THRESHOLD = 0.80


# Main post-processing entry point
def postprocess(results: List[SearchResult], top_k: int) -> List[SearchResult]:

    # If empty retrieval result lists 
    if not results:
        return []

    # min accepted score 
    min_score = 1.0 / (RRF_K + top_k)

    # Drop chunks with final combined score below min_score 
    filtered = [r for r in results if r.score >= min_score]

    # Remove near-duplicates
    deduped = _remove_near_duplicates(filtered)

    # Left is cleaned list, return
    return deduped[:top_k]



# Removing near-duplicate chunks 
def _remove_near_duplicates(results: List[SearchResult]) -> List[SearchResult]:

    # Stores surviving chunks 
    kept = []

    # Token sets of kept chunks — in sync with kept
    kept_token_sets = []

    for result in results:
        
        # Convert chunks to token sets for Jaccard similarity
        tokens = set(tokenize(result.chunk.text))

        is_duplicate = False

        # Compare current chunk against every previously kept chunk
        for existing_tokens in kept_token_sets:

            # Drop chunk if too similar — above 0.8 Jaccard similarity + break
            if _jaccard(tokens, existing_tokens) >= DUPLICATE_THRESHOLD:
                is_duplicate = True
                break
        
        # Keep unique chunks 
        if not is_duplicate:
            kept.append(result)
            kept_token_sets.append(tokens)

    return kept

# Compute jaccard similarity btw 2 chunks — scan for word overlap
def _jaccard(a: set, b: set) -> float:

    # Both empty = identical edge case
    if not a and not b:
        return 1.0  

    # Calculate intersection and union of token sets
    intersection = len(a & b)
    union = len(a | b)

    # diving intersection by union — scores btw 0 and 1 always
    return intersection / union if union > 0 else 0.0
