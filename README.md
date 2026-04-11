# StackAI-RAG

RAG with ability to upload PDF and ask questions about it. Use FastAPI + Mistral AI. No external libraries.

---

## Workflow 

PDFs are uploaded via the ingest endpoint. The system extracts text page by page, splits it into overlapping chunks, embeds each chunk using Mistral's embedding model, and indexes them in two ways — a vector store for semantic search and a BM25 index for keyword search.

When a user asks a question, the system detects whether it needs to search the knowledge base at all. If yes, both searches are executed and it merges and re-ranks the results. It then checks if the evidence is strong enough and calls Mistral to formulate a response grounded in retrieved chunks.

If the retrieved context doesn't contain enough information to answer the question, the system returns a clear "can't answer" response with no citations — instead of citing irrelevant chunks.

---

## Project Structure

```
StackAI-RAG/
├── app/
│   ├── main.py               # FastAPI app, CORS, static files, /health
│   ├── config.py             # All settings loaded from environment
│   ├── api/
│   │   ├── ingest.py         # POST /ingest — PDF upload and indexing
│   │   └── query.py          # POST /query — question answering
│   ├── core/
│   │   ├── models.py         # Shared types (SearchResult)
│   │   ├── vector_store.py   # In-memory cosine similarity search via Mistral embeddings
│   │   └── keyword_index.py  # BM25 keyword search, built from scratch
│   └── services/
│       ├── ingestion.py       # PDF extraction and sliding-window chunking
│       ├── query_processor.py # Intent detection and query rewriting
│       ├── retriever.py       # Hybrid search — semantic + keyword fusion (RRF)
│       ├── postprocessor.py   # Score threshold filter and near-duplicate removal
│       └── generator.py       # Answer generation, citation cleanup, no-answer detection
├── ui/
│   └── index.html             # Single-page chat interface
├── storage/                   # Persisted vector and keyword index data — gitignored
├── .env.example               # All required environment variables
├── start.sh                   # Quick-start script
└── requirements.txt
```

---

## Chat UI

The frontend is a single-page HTML/CSS/JS chat interface served at the root URL (`http://localhost:8000`). No frameworks, no build step — just one `index.html` file.

**Left sidebar:**
- Drag-and-drop PDF upload (multiple files supported)
- Document list showing filenames and chunk counts
- Clear knowledge base button

**Right chat area:**
- Message bubbles (user on the right, assistant on the left)
- Typing indicator (bouncing dots) while waiting for a response
- Intent badges below each response — `knowledge`, `chitchat`, or `refusal`
- Source citation tags showing filename and page number

**Response formatting:**
- Responses render with proper markdown — bold, italic, bullet points, numbered lists
- Lists are only used when the user explicitly asks for them; default responses are plain prose
- Citation markers like `[1]`, `[2]` are automatically stripped from Mistral's output

---

## Getting Started

### 1. Clone and set up

```bash
git clone <repo-url>
cd StackAI-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your MISTRAL_API_KEY
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`. The chat UI loads at the root URL. Hit `/health` to confirm it's up.

---

## Configuration

All tunable parameters live in `.env`. Copy `.env.example` to get started.

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `MISTRAL_API_KEY` | — | Required. Your Mistral AI API key |
| `CHUNK_SIZE` | 512 | Characters per text chunk when splitting PDFs |
| `CHUNK_OVERLAP` | 64 | Overlap between consecutive chunks to preserve context |
| `TOP_K` | 5 | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.35 | Minimum score — below this, the system returns "insufficient evidence" |
| `STORAGE_DIR` | ./storage | Where the vector and keyword indexes are saved to disk |

**Why these defaults?**

`CHUNK_SIZE 512` — 512 chars covers about a paragraph or two, should suit most documents.

`CHUNK_OVERLAP 64` — This is a bit over 10% of chunk size above. As such, sentences that fall on boundary don't get cut in half semantically — the overlap carries them into the next chunk.

`TOP_K 5` — Can be increased for broad research questions or lowered if needed. 5 will handle for now.

`SIMILARITY_THRESHOLD 0.35` — if below this score, retrieved chunks are too loosely related to the query to be trusted as evidence. Can be tuned later with documents.

---

## API

### Ingest

**`POST /api/ingest`**
Upload one or more PDF files into the knowledge base.

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@report.pdf" \
  -F "files=@handbook.pdf"
```

Response:
```json
{
  "files_processed": 2,
  "total_chunks": 34,
  "results": [
    { "filename": "report.pdf", "chunks": 20 },
    { "filename": "handbook.pdf", "chunks": 14 }
  ]
}
```

**`DELETE /api/ingest`**
Wipe the entire knowledge base and start fresh.

```bash
curl -X DELETE http://localhost:8000/api/ingest
```

---

### Query

**`POST /api/query`**
Ask a question against the uploaded documents.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the Q3 earnings?"}'
```

Optional `top_k` parameter (1–20) controls how many chunks are retrieved:

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the Q3 earnings?", "top_k": 3}'
```

Response:
```json
{
  "answer": "Q3 net revenue reached 4.2 billion, up 12% year over year.",
  "intent": "knowledge",
  "citations": [
    { "source": "annual_report.pdf", "page": 4 }
  ],
  "original_query": "What were the Q3 earnings?",
  "rewritten_query": "What were the Q3 earnings figures?"
}
```

The `intent` field will be `knowledge`, `chitchat`, or `refusal` depending on how the query was classified. Citations are only returned for `knowledge` responses where the context was sufficient to answer.

---

## How It Works

### Intent Classification
Every query is first classified by Mistral into one of three intents:
- **Knowledge** — needs document context, triggers retrieval pipeline
- **Chitchat** — conversational, answered directly without documents
- **Refusal** — inappropriate or out-of-scope, returns a polite decline

### Hybrid Retrieval
Knowledge queries run through two independent search systems:
- **Semantic search** — query is embedded via Mistral's `mistral-embed` model and compared against chunk embeddings using cosine similarity
- **BM25 keyword search** — TF-IDF scoring with k1=1.5, b=0.75, built from scratch with hardcoded stopwords

Results from both systems are merged using Reciprocal Rank Fusion (RRF) with k=60, which avoids normalizing incompatible score scales by working only on rank positions.

### Post-processing
Retrieved chunks pass through two filters:
- **Score threshold** — drops chunks whose RRF score falls below `1/(60 + top_k)`
- **Near-duplicate removal** — Jaccard similarity at 0.80 threshold removes overlapping chunks from the sliding window

### Answer Generation
The final prompt is shaped by intent:
- **Knowledge** — grounded RAG prompt with retrieved context, instructed to only use provided passages
- **Chitchat** — short conversational prompt, no document context
- **Refusal** — fixed polite decline, no API call

After generation, citation markers (`[1]`, `[2]`) are stripped from the answer. If the model indicates it cannot answer from the context (phrases like "does not contain", "cannot answer"), citations are cleared so irrelevant source tags don't appear.
