from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router


app = FastAPI(
    title="StackAI RAG",
    description="Retrieval-Augmented Generation pipeline over PDF knowledge bases",
    version="1.0.0",
)

# Letting everything in for simplicity for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(ingest_router, prefix="/api")
app.include_router(query_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(ui_dir):
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
