# Query processing
# #intent classification (1 Mistral call) and query rewriting (sharpening)

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from mistralai import Mistral
from app.config import settings

# fixed set of 3 intents 
class Intent(str, Enum):
    KNOWLEDGE = "knowledge"     
    CHITCHAT  = "chitchat"      
    REFUSAL   = "refusal"       

# structured output of query processing
@dataclass
class ProcessedQuery:
    original: str       # raw query from user
    rewritten: str      # rewritten query for retrieval 
    intent: Intent  
    refusal_reason: Optional[str] = None


# Main entry 
# classify first, rewrite if needed, then return structure output for retriever downstream
def process_query(query: str) -> ProcessedQuery:

    # Classify intent
    intent, refusal_reason = _classify(query)

    # If chitchat / refusal — NO retrieval
    if intent != Intent.KNOWLEDGE:
        return ProcessedQuery(
            original = query,
            rewritten = query,
            intent = intent,
            refusal_reason = refusal_reason,
        )

    # Knowledge queries —> rewirte for better retrieval, then return
    rewritten = _rewrite(query)
    return ProcessedQuery(
        original = query,
        rewritten = rewritten,
        intent = intent,
    )


# Intent classifier —> Mistral assigns one label to the query 

def _classify(query: str) -> tuple:

    # build API client
    client = Mistral(api_key = settings.mistral_api_key)

    # Few-shot prompt with output JSON format for downstream
    prompt = f"""Classify the user query into exactly one of these intents:

- "knowledge": the user is asking a question that requires searching a document knowledge base
- "chitchat": greetings, small talk, or questions that don't require any document search (e.g. "hello", "how are you", "what can you do")
- "refusal": the query requests personal identifiable information (SSN, passwords, credit cards), or asks for professional legal or medical advice that requires a licensed professional

Respond with a JSON object only. No explanation outside the JSON.

Format:
{{"intent": "<knowledge|chitchat|refusal>", "reason": "<one sentence, only required when intent is refusal, otherwise null>"}}

Query: {query}"""

    # API call
    # temp 0 -> deterministic Mistral output, response_format to ensure we get JSON back
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages = [{"role": "user", "content": prompt}],
        response_format = {"type": "json_object"},
        temperature = 0,
    )

    # Read raw model output, parse JSON, extract intent + refusal reason if any
    raw = response.choices[0].message.content.strip()
    parsed = _safe_parse_json(raw)

    # if returned label is invalid, default to knowledge 
    intent_str = parsed.get("intent", "knowledge").lower()
    reason = parsed.get("reason") or None

    # If invalid default to KNOWLEDGE
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.KNOWLEDGE   # Keep code running if Mistral acts weirdly

    return intent, reason


# Query rewriter 
def _rewrite(query: str) -> str:

    client = Mistral(api_key=settings.mistral_api_key)

    # Few-shot prompt Mistral for sharp queries 
    prompt = f"""Rewrite the following user question into a clear, self-contained search query.
- Expand vague references like "it", "they", "that thing"
- Remove conversational filler ("can you tell me", "I was wondering")
- Keep it concise — one or two sentences maximum
- Do not answer the question, only rewrite it
- If the question is already clear, return it unchanged

Original: {query}
Rewritten:"""

    
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.1,
        max_tokens = 120,           # cap length at about 2 sentences for concise rewrites
    )

    # Make sure output is clean – remove 'Rewritten' or 'Query' if Mistral includes it  
    rewritten = response.choices[0].message.content.strip()
    rewritten = re.sub(r'^(Rewritten:|Query:)\s*', '', rewritten, flags=re.IGNORECASE)

    # if failed, use original query 
    return rewritten or query



# JSON parser 
def _safe_parse_json(text: str) -> dict:

    # Try normal parsing first 
    try:
        return json.loads(text)
    
    # If fails - Mistral addede extra text
    except json.JSONDecodeError:
        
        #Regex to extraxt JSON from text then parse
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # If all fails return empty -> default to KNOWLEDGE in classifer above
    return {}
