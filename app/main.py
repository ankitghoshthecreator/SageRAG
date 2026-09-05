from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils.config import settings
from app.database.connection import init_db
from app.auth.router import router as auth_router
from app.documents.router import router as documents_router
from app.chat.router import router as chat_router
from app.evaluation.router import router as evaluation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB tables
    init_db()
    yield
    # Shutdown: cleanup if needed
    pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "SageRAG – Enterprise Knowledge Assistant. "
        "Hybrid RAG with fine-tuned LLMs, Qdrant, and JWT auth."
    ),
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def read_root():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": "4.0.0",
    }


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(
    auth_router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["Authentication"],
)

app.include_router(
    documents_router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["Documents"],
)

app.include_router(
    chat_router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["Chat"],
)

if settings.EVALUATION_ENABLED:
    app.include_router(
        evaluation_router,
        prefix=f"{settings.API_V1_STR}/evaluation",
        tags=["Evaluation"],
    )
