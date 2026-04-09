"""
Answer generation — synthesises a final response from retrieved chunks.

The prompt sent to Mistral is shaped by the query intent determined in
commit 6. Three paths exist:

  KNOWLEDGE  — a grounded RAG prompt. The retrieved chunks are formatted as
               numbered context passages and the model is instructed to answer
               using only that context. If the context doesn't contain enough
               information, the model says so rather than guessing. Source
               citations (filename + page) are collected from every chunk used.

  CHITCHAT   — no retrieval context is needed. A short, friendly prompt asks
               the model to respond conversationally. No citations produced.

  REFUSAL    — no Mistral call is made at all. The refusal reason from the
               classifier is turned into a polite, fixed response message.

The output in all three cases is a GeneratedAnswer dataclass with a consistent
shape: answer text, list of source citations, and the intent that produced it.
This uniform shape means the API layer in commit 10 can handle all three paths
without branching on intent itself.
"""

import os
from dataclasses import dataclass, field
from typing import List

from mistralai import Mistral

from app.config import settings
from app.core.models import SearchResult
from app.services.query_processor import Intent, ProcessedQuery


GENERATION_MODEL = "mistral-small-latest"
MAX_TOKENS = 512


@dataclass
class Citation:
    source: str
    page: int


@dataclass
class GeneratedAnswer:
    answer: str
    intent: Intent
    citations: List[Citation] = field(default_factory=list)


def generate(processed: ProcessedQuery, results: List[SearchResult]) -> GeneratedAnswer:
    """
    Generate a final answer given a processed query and post-processed chunks.

    Routes to the appropriate generation path based on intent.
    """
    if processed.intent == Intent.REFUSAL:
        return _refusal_answer(processed)

    if processed.intent == Intent.CHITCHAT:
        return _chitchat_answer(processed)

    return _rag_answer(processed, results)


def _rag_answer(processed: ProcessedQuery, results: List[SearchResult]) -> GeneratedAnswer:
    """Build a grounded answer from retrieved context chunks."""
    if not results:
        return GeneratedAnswer(
            answer="I couldn't find relevant information in the uploaded documents to answer that question.",
            intent=processed.intent,
            citations=[],
        )

    context_blocks = []
    for i, result in enumerate(results, start=1):
        context_blocks.append(
            f"[{i}] (source: {result.chunk.source}, page {result.chunk.page})\n{result.chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions based on uploaded documents.

Use only the context passages below to answer the question. Do not use outside knowledge.
If the context does not contain enough information to answer, say so clearly — do not guess.
Be concise. Reference the passage numbers (e.g. [1], [2]) where relevant.

Context:
{context}

Question: {processed.rewritten}

Answer:"""

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.complete(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )

    answer_text = response.choices[0].message.content.strip()

    seen = set()
    citations = []
    for result in results:
        key = (result.chunk.source, result.chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=result.chunk.source, page=result.chunk.page))

    return GeneratedAnswer(
        answer=answer_text,
        intent=processed.intent,
        citations=citations,
    )


def _chitchat_answer(processed: ProcessedQuery) -> GeneratedAnswer:
    """Generate a friendly conversational response — no document context needed."""
    prompt = f"""You are a helpful assistant. The user is making conversation rather than asking about a document.
Respond naturally and helpfully in one or two sentences.

User: {processed.original}"""

    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.complete(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=128,
    )

    return GeneratedAnswer(
        answer=response.choices[0].message.content.strip(),
        intent=processed.intent,
        citations=[],
    )


def _refusal_answer(processed: ProcessedQuery) -> GeneratedAnswer:
    """Return a polite refusal without making any API call."""
    reason = processed.refusal_reason or "This request falls outside what I can help with."
    answer = f"I'm not able to help with that. {reason}"

    return GeneratedAnswer(
        answer=answer,
        intent=processed.intent,
        citations=[],
    )
