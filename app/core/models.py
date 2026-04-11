# SearchResult object used throughout retriever + postprocessor to store retrieved chunks + scores
from dataclasses import dataclass
from app.services.ingestion import Chunk

# A chunk paired with its relebvance score
@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # 0-1, higher is more relevant
