# StackAI-RAG

This is a Python backend for a RAG pipeline with ability to upload PDF and ask questions against them. Built with FastAPI and the Mistral AI API. No RAG frameworks, no external vector databases, no third-party search libraries.

---

## Project Structure 

```
StackAI-RAG/
├── app/
│   ├── main.py          # FastAPI app, CORS, /health
│   ├── config.py        # All settings loaded from environment
│   ├── api/
│   │   ├── ingest.py    # POST /ingest endpoint (commit 5)
│   │   └── query.py     # POST /query endpoint (commit 10)
│   ├── core/
│   │   ├── vector_store.py   # Custom cosine similarity store (commit 3)
│   │   └── keyword_index.py  # Custom BM25 index (commit 4)
│   └── services/
│       ├── ingestion.py       # PDF extraction + chunking (commit 2)
│       ├── query_processor.py # Intent detection + query rewriting (commit 6)
│       ├── retrieval.py       # Hybrid search fusion (commit 7)
│       ├── reranker.py        # Re-ranking + threshold guard (commit 8)
│       └── generator.py       # LLM answer generation (commit 9)
├── ui/                  # Chat frontend (commit 11)
├── storage/             # Persisted index data — gitignored
├── .env.example         # All required environment variables
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
| `MISTRAL_API_KEY` | — | Mistral AI API key |
| `CHUNK_SIZE` | 512 | Chars / text chunk when splitting PDFs |
| `CHUNK_OVERLAP` | 64 | Overlap between consecutive chunks to preserve context |
| `TOP_K` | 5 | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | 0.35 | Minscore — below this, the system returns "insufficient evidence" |
| `STORAGE_DIR` | ./storage | Where the vector and keyword indexes are saved to disk |

**Why these defaults?**

`CHUNK_SIZE 512` — 512 chars covers about a paragraph or two, should suit most documents. 

`CHUNK_OVERLAP 64` — This is a bit over 10% of chunk size above. As sucuh, sentences that fall on boundary don't get cut in half semantically — the overlap carries them into the next chunk.

`TOP_K 5` — Can be increased for broad research questions or lowered if needed. 5 will handle for now.

`SIMILARITY_THRESHOLD 0.35` — if below this score, retrieved chunks are too loosely related to the query to be trusted as evidence. Can be tuned later with documents.