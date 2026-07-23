import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically locate the .env file in the root workspace folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
env_path = os.path.join(root_dir, ".env")

class Settings(BaseSettings):
    # API Configuration
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SageRAG"

    # Security & Authentication
    JWT_SECRET_KEY: str = "super-secret-jwt-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520

    # PostgreSQL Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "sage_rag"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/sage_rag"

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "sage_documents"

    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # MinIO Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "sage-documents"
    MINIO_SECURE: bool = False

    # LLM Settings
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = "mock-key"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # Monitoring & Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"
    OPENTELEMETRY_ENDPOINT: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=env_path if os.path.exists(env_path) else None,
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
