# StackAI-RAG

RAG with ability to upload PDF and ask questions about it. Use FastAPI + Mistral AI. No external libraries.

---

## How It Works

PDFs are uploaded through the ingest endpoint. The system extracts text page by page, splits it into overlapping chunks, embeds each chunk using Mistral's embedding model, and indexes them in two ways — a vector store for semantic search and a BM25 index for keyword search.

When a user asks a question, the system detects whether it needs to search the knowledge base at all. If it does, it runs both searches, merges and re-ranks the results, checks whether the evidence is strong enough to answer, and calls Mistral to generate a response grounded in the retrieved chunks.

---

## Project Structure

```
StackAI-RAG/
├── app/
│   ├── main.py               # FastAPI app, CORS, /health
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
│       ├── retrieval.py       # Hybrid search — semantic + keyword fusion (RRF)
│       ├── reranker.py        # Re-ranking and similarity threshold guard
│       └── generator.py       # Answer generation with intent-shaped prompts
├── ui/                        # Chat frontend
├── storage/                   # Persisted vector and keyword index data — gitignored
├── .env.example               # All required environment variables
└── requirements.txt
```

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

Server starts at `http://localhost:8000`. Hit `/health` to confirm it's up.

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

`CHUNK_OVERLAP 64` — This is a bit over 10% of chunk size above. As sucuh, sentences that fall on boundary don't get cut in half semantically — the overlap carries them into the next chunk.

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
