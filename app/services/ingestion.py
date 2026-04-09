"""
PDF ingestion pipeline — text extraction and chunking.

We chunk per page rather than concatenating the whole document first. This keeps
page numbers accurate on every chunk, which matters when we surface citations.

Chunk boundaries are nudged to the nearest sentence ending (period, !, ?) within
an 80-character look-back window. If no sentence boundary is found we fall back
to the nearest whitespace, and if that also fails we just cut at the hard limit.
This keeps most chunks semantically complete without complicating the algorithm.

Chunks shorter than MIN_CHUNK_CHARS are dropped — they're usually stray headers,
page numbers, or extraction artifacts that add noise without any retrieval value.

Tables and figures are treated as plain text for now. pdfplumber can detect table
regions, but extracting them as structured data and deciding how to chunk a table
is its own project.
"""

import io
import re
from dataclasses import dataclass
from typing import List, Tuple

import pdfplumber

from app.config import settings


MIN_CHUNK_CHARS = 20


@dataclass
class Chunk:
    text: str
    source: str      # original filename
    page: int        # 1-indexed
    chunk_index: int # position across the whole document
    char_start: int  # character offset within the page text


def extract_pages(file_bytes: bytes) -> List[dict]:
    """
    Open a PDF from raw bytes and pull text out page by page.
    Pages that come back empty (scanned images, blank pages) are skipped.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            pages.append({"page": page_num, "text": text.strip()})
    return pages


def _find_split_point(text: str, ideal_end: int, look_back: int = 80) -> int:
    """
    Given an ideal cut position, look back up to `look_back` chars for a
    cleaner split — sentence boundary first, word boundary second.
    """
    window_start = max(0, ideal_end - look_back)
    window = text[window_start:ideal_end]

    sentence_ends = list(re.finditer(r'[.!?]\s+', window))
    if sentence_ends:
        return window_start + sentence_ends[-1].end()

    word_boundary = re.search(r'\s\S+$', window)
    if word_boundary:
        return window_start + word_boundary.start() + 1

    return ideal_end


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int]]:
    """
    Sliding-window chunking with overlap.
    Returns a list of (chunk_text, char_start) pairs.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append((text[start:], start))
            break

        split_at = _find_split_point(text, end)
        chunks.append((text[start:split_at], start))
        start = split_at - overlap

    return chunks


def ingest_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Full pipeline for one PDF:
      1. Extract text page by page
      2. Chunk each page with overlap
      3. Tag every chunk with source file, page number, and position
    """
    pages = extract_pages(file_bytes)
    if not pages:
        return []

    all_chunks = []
    chunk_index = 0

    for page_data in pages:
        raw_chunks = split_into_chunks(
            page_data["text"],
            settings.chunk_size,
            settings.chunk_overlap,
        )

        for chunk_text, char_start in raw_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            all_chunks.append(Chunk(
                text=chunk_text,
                source=filename,
                page=page_data["page"],
                chunk_index=chunk_index,
                char_start=char_start,
            ))
            chunk_index += 1

    return all_chunks
