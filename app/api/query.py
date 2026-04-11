"""
Query endpoint — POST /api/query

Single entry point for all user questions. Orchestrates the full pipeline:
  1. process_query   — classify intent, rewrite if knowledge-seeking
  2. retrieve        — hybrid semantic + BM25 search (skipped for non-knowledge)
  3. postprocess     — threshold filter and near-duplicate removal
  4. generate        — produce final answer shaped by intent

The response always has the same shape regardless of intent, so the client
never needs to branch on whether the answer came from documents, chitchat,
or a refusal.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.generator import generate
from app.services.postprocessor import postprocess
from app.services.query_processor import Intent, process_query
from app.services.retriever import retrieve


router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class CitationOut(BaseModel):
    source: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: List[CitationOut]
    original_query: str
    rewritten_query: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a question and receive an answer grounded in the uploaded documents.

    The pipeline classifies the question, retrieves relevant passages if needed,
    filters and deduplicates them, then generates a final answer with citations.
    Chitchat is answered conversationally. Sensitive requests are refused.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    top_k = request.top_k or settings.top_k

    processed = process_query(query_text)

    results = retrieve(processed, top_k=top_k)
    clean_results = postprocess(results, top_k=top_k)
    answer = generate(processed, clean_results)

    return QueryResponse(
        answer=answer.answer,
        intent=answer.intent.value,
        citations=[CitationOut(source=c.source, page=c.page) for c in answer.citations],
        original_query=processed.original,
        rewritten_query=processed.rewritten,
    )
