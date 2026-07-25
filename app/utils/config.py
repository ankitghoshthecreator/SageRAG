import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically locate the .env file in the root workspace folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
env_path = os.path.join(root_dir, ".env")


class Settings(BaseSettings):
    # ── API ────────────────────────────────────────────────────────────────────
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SageRAG"

    # ── Security & JWT ─────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "super-secret-jwt-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "sage_rag"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/sage_rag"

    # ── Qdrant Vector DB ───────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "sage_documents"

    # ── Redis ──────────────────────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # ── MinIO ──────────────────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "sage-documents"
    MINIO_SECURE: bool = False

    # ── LLM Provider (openai | gemini | groq | openrouter | local) ─────────────
    LLM_PROVIDER: str = "openai"
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # ── LLM API Keys ──────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # ── Langfuse Observability ─────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"

    # ── Document Processing ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_UPLOAD_SIZE_MB: int = 50
    UPLOAD_DIR: str = "uploads"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 20       # candidates fetched before reranking
    TOP_K_RERANK: int = 6           # final chunks sent to LLM context

    model_config = SettingsConfigDict(
        env_file=env_path if os.path.exists(env_path) else None,
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
