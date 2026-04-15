# Query endpoint — POST /api/query

import json
import os
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.generator import FlaggedSentence, generate
from app.services.postprocessor import postprocess
from app.services.query_processor import Intent, process_query
from app.services.retriever import retrieve

# FastAPI router 
router = APIRouter()

# Incoming request schema for /query
class QueryRequest(BaseModel):
    query: str = Field(..., min_length = 1, description = "The user's question")
    top_k: Optional[int] = Field(None, ge = 1, le = 20, description = "Number of chunks to retrieve")

# Citation format returned to client
class CitationOut(BaseModel):
    source: str
    page: int


# Hallucinated sentence diagnostics returned to client
class FlaggedSentenceOut(BaseModel):
    sentence: str
    coverage: float
    unsupported_terms: List[str]

# Standardized response schema for all query types
class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: List[CitationOut]
    original_query: str
    rewritten_query: str
    flagged_sentences: List[FlaggedSentenceOut]


# POST /query endpoint — runs the full end-to-end RAG pipeline
@router.post("/query", response_model = QueryResponse)
async def query(request: QueryRequest):
    
    # Trim whitespace from user query
    query_text = request.query.strip()

   
    if not query_text:
        raise HTTPException(status_code=400, detail="Query must not be empty.")  # Reject empty queries 

    # Use config top k if not provided in request
    top_k = request.top_k or settings.top_k

    # 1. Classify intent + rewrite query if needed
    processed = process_query(query_text) 

    # 2. Retrieve candidate chunks from hybrid retriever
    results = retrieve(processed, top_k=top_k)

    # 3. Remove weak / duplicate retrieval results
    clean_results = postprocess(results, top_k=top_k)

    # 4. Generate final answer using processed query + cleaned chunks
    answer = generate(processed, clean_results)

    # Convert internal answer object into API response schema
    response = QueryResponse(
        answer = answer.answer,
        intent = answer.intent.value,

        # Convert internal Citation objects into response format
        citations=[CitationOut(source=c.source, page=c.page) for c in answer.citations],

        # Include original and rewritten queries for observability/debugging
        original_query=processed.original,
        rewritten_query=processed.rewritten,

        # Include removed sentences for transparency/debugging
        flagged_sentences = [
            FlaggedSentenceOut(
                sentence = f.sentence,
                coverage = f.coverage,
                unsupported_terms = f.unsupported_terms,
            )
            for f in answer.flagged_sentences
        ],
    )
    
    # Persist this Q&A interaction to chat log
    _append_chat_log(response)
    return response

# Add query/answer interaction to log
def _append_chat_log(response: QueryResponse) -> None:


    try:
        log_path = os.path.join(settings.storage_dir, "chat_log.json")
        os.makedirs(settings.storage_dir, exist_ok=True)            # Ensure storage directory exists

        # Build structured log entry for this interaction
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": response.original_query,
            "rewritten_query": response.rewritten_query,
            "intent": response.intent,
            "answer": response.answer,
            "citations": [{"source": c.source, "page": c.page} for c in response.citations],
            "flagged_sentences": [          # Save any hallucination-filtered sentences for debugging
                {
                    "sentence": f.sentence,
                    "coverage": f.coverage,
                    "unsupported_terms": f.unsupported_terms,
                }
                for f in response.flagged_sentences
            ],
        }

        # Load existing log or start fresh
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = []

        log.append(entry)

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    except Exception:
        # Never let logging failure break a query response
        pass
