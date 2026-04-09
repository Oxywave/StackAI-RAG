from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List

from pydantic import BaseModel

from app.core.keyword_index import get_keyword_index
from app.core.vector_store import get_vector_store
from app.services.ingestion import ingest_pdf


router = APIRouter()


class FileResult(BaseModel):
    filename: str
    chunks: int


class IngestResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: List[FileResult]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files for ingestion into the knowledge base.

    Each file is extracted, chunked, embedded, and indexed in both the
    vector store and keyword index. Returns a summary per file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []

    for upload in files:
        _validate_pdf(upload)

        raw = await upload.read()

        try:
            chunks = ingest_pdf(raw, upload.filename)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"{upload.filename}: could not parse PDF — {str(e)}",
            )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=f"{upload.filename}: no text could be extracted. "
                       "The file may be a scanned image without embedded text.",
            )

        get_vector_store().add(chunks)
        get_keyword_index().add(chunks)

        results.append(FileResult(filename=upload.filename, chunks=len(chunks)))

    return IngestResponse(
        files_processed=len(results),
        total_chunks=sum(r.chunks for r in results),
        results=results,
    )


@router.delete("/ingest")
async def clear_index():
    """Wipe the entire knowledge base — vector store and keyword index."""
    get_vector_store().clear()
    get_keyword_index().clear()
    return {"detail": "Knowledge base cleared."}


def _validate_pdf(upload: UploadFile) -> None:
    """Raise 400 if the uploaded file doesn't look like a PDF."""
    filename = upload.filename or ""
    content_type = upload.content_type or ""

    if not filename.lower().endswith(".pdf") and "pdf" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"{filename or 'unknown'}: only PDF files are accepted.",
        )
