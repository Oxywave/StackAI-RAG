from dataclasses import dataclass

from app.services.ingestion import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # 0-1, higher is more relevant
