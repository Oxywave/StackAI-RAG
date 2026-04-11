from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List

from pydantic import BaseModel

from app.core.keyword_index import get_keyword_index
from app.core.vector_store import get_vector_store
from app.services.ingestion import ingest_pdf


router = APIRouter()

# The two response models
# Per-file ingestion result returned in API response 
class FileResult(BaseModel):
    filename: str   # file name
    chunks: int     # chunk count

# Full response schema for /ingest endpoint
class IngestResponse(BaseModel):
    files_processed: int
    total_chunks: int
    results: List[FileResult]


# POST /ingest endpoint — upload one or more PDFs and add them into the knowledge base
@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):

    # Reject empty upload request
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []

    # Process each uploaded file independently
    for upload in files:
        _validate_pdf(upload)

        raw = await upload.read()

        # Run PDF ingestion pipeline (extract text -> chunk)
        try:
            chunks = ingest_pdf(raw, upload.filename)

        # If parsing/chunking fails, return PDF parsing error
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

        # Add chunks into semantic vector index
        get_vector_store().add(chunks)

        # Add chunks into BM25 keyword index
        get_keyword_index().add(chunks)

        # Record successful ingestion result for this file
        results.append(FileResult(filename = upload.filename, chunks = len(chunks)))
    
    # Return ingestion summary for all processed files
    return IngestResponse(
        files_processed=len(results),
        total_chunks=sum(r.chunks for r in results),
        results=results,
    )


# Clears the entire indexed knowledge base
@router.delete("/ingest")
async def clear_index():

    get_vector_store().clear()
    get_keyword_index().clear()
    return {"detail": "Knowledge base cleared."}


# Reject uploads that do not look like PDFs
def _validate_pdf(upload: UploadFile) -> None:

    filename = upload.filename or ""
    content_type = upload.content_type or ""

    if not filename.lower().endswith(".pdf") and "pdf" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"{filename or 'unknown'}: only PDF files are accepted.",
        )
