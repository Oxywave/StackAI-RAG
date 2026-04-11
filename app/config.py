from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # Must have mistral API key to run the app
    mistral_api_key: str = Field(..., env = "MISTRAL_API_KEY") 

    # Direct to Storage
    storage_dir: str = Field("./storage", env = "STORAGE_DIR")

    # Chunking
    # default to 512 and 64 unless overridden in .env. 
    chunk_size: int = Field(512, env = "CHUNK_SIZE")
    chunk_overlap: int = Field(64, env = "CHUNK_OVERLAP")

    # Retrieval
    # default to 5 top chunks for answer generation + threshold default to 0.6
    top_k: int = Field(5, env = "TOP_K")
    similarity_threshold: float = Field(0.60, env = "SIMILARITY_THRESHOLD") # refined through testing

    # Server
    host: str = Field("0.0.0.0", env = "HOST")
    port: int = Field(8000, env = "PORT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"} # read from .env at strat of app


settings = Settings()
