from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(
    title="StackAI RAG",
    description="Retrieval-Augmented Generation pipeline over PDF knowledge bases",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers will be registered here in later commits
# app.include_router(ingest_router, prefix="/api")
# app.include_router(query_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Mount the UI as static files (populated in commit 11)
ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(ui_dir):
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
