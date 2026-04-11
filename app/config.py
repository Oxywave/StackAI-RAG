from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")

    storage_dir: str = Field("./storage", env="STORAGE_DIR")

    # Chunking
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(5, env="TOP_K")
    # Mistral's dense embeddings compress all text into a narrow cone — even
    # unrelated queries score ~0.40–0.55 cosine similarity. A threshold of 0.35
    # is effectively useless. 0.60 sits above the noise floor and below typical
    # on-topic scores (0.65+), making the similarity gate actually meaningful.
    similarity_threshold: float = Field(0.60, env="SIMILARITY_THRESHOLD")

    # Server
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
