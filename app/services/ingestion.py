# Ingestion pipeline for PDF — text extraction and chunking.
# Take a PDF, extract text page by page, split that text into chunks, and attach metadata to each chunk so retrieval and citation work later.


import io
import re
from dataclasses import dataclass
from typing import List, Tuple
import pdfplumber
from app.config import settings

# This is like 3-4 words — so we set this as floor
MIN_CHUNK_CHARS = 20

# Define standard chunk object
@dataclass
class Chunk:
    text: str
    source: str      # original filename
    page: int        # 1-indexed
    chunk_index: int # position across the whole document
    char_start: int  # character offset within the page text


# Main Entry Point for PDF ingestion
# 

def ingest_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:

    # Extract non-empty text pages from the PDF - runs pdfplumber 
    pages = extract_pages(file_bytes)

    # If no text —> return empty list 
    if not pages:
        return []

    all_chunks = []
    chunk_index = 0      # chunk counter 

    # Process page one by one - preserve metadata 
    for page_data in pages:
        
        # Split one page at a time into 
        raw_chunks = split_into_chunks(page_data["text"], settings.chunk_size, settings.chunk_overlap,
        )

        # Convert each raaw chunks to structured object with metadata
        for chunk_text, char_start in raw_chunks:

            # Drop white space from both ends
            chunk_text = chunk_text.strip()

            # Drop chunks too short to avoid noise 
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            # # Create Chunk object with text, source, location metadata
            all_chunks.append(Chunk(
                text = chunk_text,
                source = filename,
                page=page_data["page"],
                chunk_index =chunk_index,
                char_start = char_start,
            ))
            chunk_index += 1

    # Return full list of processed chunks for this whole PDF
    return all_chunks


# split one page text into chunks 
def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int]]:

    # List, entires include (chunk_text, starting pos)
    chunks = []

    # moving pointer for current starting pos
    start = 0

    # Keep going until start pos exceeds text length — each itr will produce one chukn 
    while start < len(text):

        # Ideal end point so 512 beyond current start
        end = start + chunk_size

        # If remaining is smaller than smaller than full chunk, keep rest and break
        if end >= len(text):
            chunks.append((text[start:], start))
            break
        
        # Find clean split before the end 
        split_at = _find_split_point(text, end)

        # Save current chunk and its start pos
        chunks.append((text[start:split_at], start))

        # Move start pointer forward by chunk size minus overlap for next chunk
        start = split_at - overlap

    return chunks



# Finding the clean split point — To avoid splitting mid-sentence
# As deisgned, ideal is current widow start + 512 & we look back 80 chars for split
def _find_split_point(text: str, ideal_end: int, look_back: int = 80) -> int:


    # Looking back window 
    window_start = max(0, ideal_end - look_back)

    # Slice out the 80 chars window 
    window = text[window_start:ideal_end]

    # Search for sentence end with Regex — that is cleanest split
    sentence_ends = list(re.finditer(r'[.!?]\s+', window))
    if sentence_ends:

        # update new window start pos
        return window_start + sentence_ends[-1].end()

    # If no sentence end exist – avoid cutting mid-word by searching white-space
    word_boundary = re.search(r'\s\S+$', window)

    if word_boundary:
        # update new window start pos
        return window_start + word_boundary.start() + 1

    return ideal_end




# Text extraction from PDF - page by page, skip empty 
def extract_pages(file_bytes: bytes) -> List[dict]:

    pages = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

        # Enumerate page from 1 
        for page_num, page in enumerate(pdf.pages, start = 1):
            
            # extraction
            text = page.extract_text()

            # skip if blank
            if not text or not text.strip():
                continue
            
            # store page  # and text
            pages.append({"page": page_num, "text": text.strip()})
    return pages
