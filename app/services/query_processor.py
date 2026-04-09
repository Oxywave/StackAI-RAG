"""
Query processing — intent classification and query rewriting.

Two things happen here before any search runs:

1. Intent classification — a single Mistral call determines whether the query
   actually needs the knowledge base. Greetings and small talk are answered
   directly. Requests for PII or legal/medical advice are refused. Everything
   else is routed to retrieval.

2. Query rewriting — vague or conversational queries get sharpened into
   something more likely to retrieve the right chunks. "what did it say about
   revenue?" becomes a cleaner, self-contained question. This runs only when
   the intent is knowledge-seeking.

Both steps use Mistral chat completions. Classification asks for a JSON object
so the response is always machine-readable. Rewriting is a freeform generation
with a tight prompt.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mistralai import Mistral

from app.config import settings


class Intent(str, Enum):
    KNOWLEDGE = "knowledge"     # needs knowledge base search
    CHITCHAT  = "chitchat"      # greeting, small talk, general question
    REFUSAL   = "refusal"       # PII request or legal/medical advice


@dataclass
class ProcessedQuery:
    original: str
    rewritten: str
    intent: Intent
    refusal_reason: Optional[str] = None


def process_query(query: str) -> ProcessedQuery:
    """
    Classify the query and rewrite it if it needs retrieval.
    Returns a ProcessedQuery with intent, rewritten text, and optional refusal reason.
    """
    intent, refusal_reason = _classify(query)

    if intent != Intent.KNOWLEDGE:
        return ProcessedQuery(
            original=query,
            rewritten=query,
            intent=intent,
            refusal_reason=refusal_reason,
        )

    rewritten = _rewrite(query)
    return ProcessedQuery(
        original=query,
        rewritten=rewritten,
        intent=intent,
    )


def _classify(query: str) -> tuple:
    """
    Ask Mistral to classify the query into one of three intents.
    Returns (Intent, refusal_reason_or_None).
    """
    client = Mistral(api_key=settings.mistral_api_key)

    prompt = f"""Classify the user query into exactly one of these intents:

- "knowledge": the user is asking a question that requires searching a document knowledge base
- "chitchat": greetings, small talk, or questions that don't require any document search (e.g. "hello", "how are you", "what can you do")
- "refusal": the query requests personal identifiable information (SSN, passwords, credit cards), or asks for professional legal or medical advice that requires a licensed professional

Respond with a JSON object only. No explanation outside the JSON.

Format:
{{"intent": "<knowledge|chitchat|refusal>", "reason": "<one sentence, only required when intent is refusal, otherwise null>"}}

Query: {query}"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    parsed = _safe_parse_json(raw)

    intent_str = parsed.get("intent", "knowledge").lower()
    reason = parsed.get("reason") or None

    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.KNOWLEDGE

    return intent, reason


def _rewrite(query: str) -> str:
    """
    Rephrase the query into a clean, self-contained search question.
    Expands vague pronouns, removes filler, makes implicit context explicit.
    """
    client = Mistral(api_key=settings.mistral_api_key)

    prompt = f"""Rewrite the following user question into a clear, self-contained search query.
- Expand vague references like "it", "they", "that thing"
- Remove conversational filler ("can you tell me", "I was wondering")
- Keep it concise — one or two sentences maximum
- Do not answer the question, only rewrite it
- If the question is already clear, return it unchanged

Original: {query}
Rewritten:"""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=120,
    )

    rewritten = response.choices[0].message.content.strip()
    rewritten = re.sub(r'^(Rewritten:|Query:)\s*', '', rewritten, flags=re.IGNORECASE)
    return rewritten or query


def _safe_parse_json(text: str) -> dict:
    """Parse JSON from the model response, returning an empty dict on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}
