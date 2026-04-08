# StackAI-RAG

A Python backend for a Retrieval-Augmented Generation (RAG) pipeline that lets you upload PDF documents and ask questions against them. Built with FastAPI and the Mistral AI API. No RAG frameworks, no external vector databases, no third-party search libraries — the retrieval and search logic is written from scratch.

---

## Project Structure

```
StackAI-RAG/
├── app/
│   ├── main.py               # FastAPI app, CORS, /health
│   ├── config.py             # All settings loaded from environment
│   ├── api/
│   │   ├── ingest.py         # POST /ingest endpoint
│   │   └── query.py          # POST /query endpoint
│   ├── core/
│   │   ├── vector_store.py   # Custom cosine similarity store
│   │   └── keyword_index.py  # Custom BM25 index
│   └── services/
│       ├── ingestion.py       # PDF extraction + chunking
│       ├── query_processor.py # Intent detection + query rewriting
│       ├── retrieval.py       # Hybrid search fusion
│       ├── reranker.py        # Re-ranking + threshold guard
│       └── generator.py       # LLM answer generation
├── ui/                        # Chat frontend
├── storage/                   # Persisted index data — gitignored
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

`CHUNK_SIZE 512` — sits in a comfortable middle ground. Too small and a single chunk rarely contains enough context to answer anything. Too large and you're stuffing unrelated content into one embedding, which muddies the search signal. 512 characters covers roughly a paragraph or two, which is usually the natural unit of an idea in a document.

`CHUNK_OVERLAP 64` — about 12% of the chunk size. Sentences that fall on a chunk boundary don't get cut in half semantically — the overlap carries them into the next chunk. Any higher and you're storing a lot of redundant text; any lower and you start losing boundary context.

`TOP_K 5` — five chunks is usually enough to answer a question without overwhelming the LLM's context window with noise. You can bump this up for broad research questions or down if you find the answers are getting diluted.

`SIMILARITY_THRESHOLD 0.35` — below this score, the retrieved chunks are too loosely related to the query to be trusted as evidence. Rather than generate an answer from weak matches, the system refuses and says so. 0.35 is a starting point — you'll want to tune this against your own documents.

---

*Full system design, architecture diagrams, and API documentation added in commit 12.*
