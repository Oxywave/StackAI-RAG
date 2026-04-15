# Answer generation

import re
from dataclasses import dataclass, field
from typing import List
from mistralai import Mistral
from app.config import settings
from app.core.models import SearchResult
from app.services.query_processor import Intent, ProcessedQuery


GENERATION_MODEL = "mistral-small-latest"

# Max length for generated answer output
MAX_TOKENS = 512

# Default decline message when the model can't answer from the PDFs
NO_ANSWER_MESSAGE = "The context provided cannot answer this question."

# Sentences below this threshold are treated as unsupported and removed
EVIDENCE_COVERAGE_THRESHOLD = 0.30


# When hallucination is flagged
# Show user the original str, the coverage %, and the unsupported terms as evidence 
@dataclass
class FlaggedSentence:
    sentence: str
    coverage: float          # % of sentence tokens found in retrieved chunks 
    unsupported_terms: List[str]  # content words the model used that don't appear in any chunk


# Filename + page # for citation in generated answer
@dataclass
class Citation:
    source: str
    page: int


# For UI
# Standard response to user
@dataclass
class GeneratedAnswer:
    answer: str
    intent: Intent
    citations: List[Citation] = field(default_factory = list)
    flagged_sentences: List[FlaggedSentence] = field(default_factory = list)


    
# Main public entry point — router
def generate(processed: ProcessedQuery, results: List[SearchResult]) -> GeneratedAnswer:
    
    # choose refusal, chitchat, or grounded RAG answering based on intent

    if processed.intent == Intent.REFUSAL:
        return _refusal_answer(processed)

    if processed.intent == Intent.CHITCHAT:
        return _chitchat_answer(processed)

    return _rag_answer(processed, results)


# Building grounded answer from chunks retrieved
def _rag_answer(processed: ProcessedQuery, results: List[SearchResult]) -> GeneratedAnswer:

    # Check if result list input is empty 
    if not results:
        return GeneratedAnswer(answer = NO_ANSWER_MESSAGE, intent = processed.intent, citations=[],
        )


    context_blocks = []
    for result in results:
        context_blocks.append(
            f"(source: {result.chunk.source}, page {result.chunk.page})\n{result.chunk.text}"
        )
    context = "\n\n".join(context_blocks)

    # Mistral prompt forces grounded answering from context only
    # Exclude citation marker for response clealiness, use bullets point when asked to list
    prompt = f"""You are a helpful assistant answering questions based on uploaded documents.

    Use only the context passages below to answer the question. Do not use outside knowledge. If the context does not contain enough information to answer, say so clearly — do not guess.
    Be concise. Only use bullet points or numbered lists if the user explicitly asks to list or enumerate items — otherwise respond in plain prose paragraphs. 
    Do not include citation markers like [1] or [2] in your answer.

Context:
{context}

Question: {processed.rewritten}

Answer:"""

    # Call Mistral to generate grounded answer
    client = Mistral(api_key = settings.mistral_api_key)
    response = client.chat.complete(model = GENERATION_MODEL, 
        messages = [{"role": "user", "content": prompt}], 
        temperature = 0.2, 
        max_tokens = MAX_TOKENS,
    )

    # clean citation markers just in case
    answer_text = _clean_answer(response.choices[0].message.content.strip())

    # Similarity gate (2nd layer) 
    # If model says context is insufficient, normalize to the same no-answer message
    if _is_no_answer(answer_text):
        return GeneratedAnswer(
            answer = NO_ANSWER_MESSAGE,
            intent = processed.intent,
            citations = [],
        )

    # Hallucination Detector
    # Remove unsupported sentences from the answer
    answer_text, flagged = _hallucination_detector(answer_text, results)

    seen = set()
    citations = []

    for result in results:
        key = (result.chunk.source, result.chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=result.chunk.source, page=result.chunk.page))

    # Return grounded answer, citations, and any flagged dropped sentences
    return GeneratedAnswer(
        answer = answer_text,
        intent = processed.intent,
        citations = citations,
        flagged_sentences = flagged,
    )




# Remove markers from Mistral response — Clean answer
def _clean_answer(text: str) -> str:
    # Remove inline citation markers: [1], [1, 2], [1][2], [1,2,3] etc.
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

    # Collapse any double spaces left by removed markers
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# Similarity gate layer 2 
def _is_no_answer(text: str) -> bool:
    lower = text.lower()

    # Common phrases for whole answer refusal cases
    # Serves different purpose then _META_PHRASES for Hallucination Detector 
    indicators = [
        "cannot answer", "can't answer",
        "not contain", "does not contain", "doesn't contain",
        "no information", "not enough information",
        "couldn't find", "could not find",
        "not able to answer", "unable to answer", "not mentioned",
        "no relevant", "outside the scope",  "not covered",
    ]

    return any(phrase in lower for phrase in indicators)



# Split generated answer to sentences 
def _split_sentences(text: str) -> List[str]:

    # uses punctuation as trigger
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]



# Hallucination Detector
def _hallucination_detector(answer: str, results: List[SearchResult]):

    from app.core.keyword_index import tokenize

    # Reuse same tokenizer as the keyword index for lexical consistency
    chunk_tokens: set = set()

    # Every meaningful word from every retrieved chunks goes into the vocabulary set
    for r in results:
        chunk_tokens.update(tokenize(r.chunk.text))

    # splitting answer into sentences 
    sentences = _split_sentences(answer)
    kept = []
    flagged: List[FlaggedSentence] = []

    # Process each sentence  
    for sentence in sentences:
        tokens = tokenize(sentence)

        # Keep short or meta sentences automatically — not factual claims
        if len(tokens) < 3 or _is_meta_sentence(sentence):
            kept.append(sentence)
            continue

        # Measure what fraction of sentence tokens appear in retrieved chunks
        covered = sum(1 for t in tokens if t in chunk_tokens)
        coverage = covered / len(tokens)

        # Keep sentence above threshold 
        if coverage >= EVIDENCE_COVERAGE_THRESHOLD:
            kept.append(sentence)

        else:
            # Record unsupported details for debugging / observability
            unsupported = [t for t in tokens if t not in chunk_tokens]
            flagged.append(FlaggedSentence(sentence = sentence, coverage = round(coverage, 2), unsupported_terms = unsupported,
            ))

    # failsafe - if every sentence was flagged, return error message
    cleaned = " ".join(kept) if kept else "Entire content flagged as Hallucination."

    return cleaned, flagged





# Identify sentences that are just stating limits of the source/context
def _is_meta_sentence(sentence: str) -> bool:

    lower = sentence.lower().strip()
    return any(phrase in lower for phrase in _META_PHRASES)



# Exemption list for the hallucination detector 
_META_PHRASES = [
    # Source-referncing langauge 
    "the context", "the document", "the paper", "the text", "the passage",
    "the provided", "the excerpt", "the source",

    # Explicit absence / limitation language
    "does not explicitly", "does not directly", "does not provide",
    "does not contain", "does not mention", "does not state",
    "does not specify", "does not address", "does not include",
    "does not appear", "do not explicitly", "do not directly",
    "not explicitly", "not directly", "not provided", "not mentioned",
    "not stated", "not specified", "not addressed", "not available",
    "no information", "no direct", "no explicit", "no mention",
    "insufficient", "unable to determine", "cannot be determined",
    "it is unclear", "it is not clear",
]




# Chitchat 
def _chitchat_answer(processed: ProcessedQuery) -> GeneratedAnswer:

    prompt = f"""You are a helpful assistant. The user is making conversation rather than asking about a document.
Respond naturally and helpfully in one or two sentences.

User: {processed.original}"""

    # higher temperature to sound like real chitchat
    # Cap response length short for quick replies
    client = Mistral(api_key=settings.mistral_api_key)
    response = client.chat.complete(
        model = GENERATION_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.7,
        max_tokens = 128,
    )

    return GeneratedAnswer(
        answer=response.choices[0].message.content.strip(),
        intent=processed.intent,
        citations=[],
    )


# Refual — makes 0 API calls
def _refusal_answer(processed: ProcessedQuery) -> GeneratedAnswer:


    reason = processed.refusal_reason or "This request falls outside what I can help with."
    answer = f"I'm not able to help with that. {reason}"

    return GeneratedAnswer(
        answer=answer,
        intent=processed.intent,
        citations=[],
    )
