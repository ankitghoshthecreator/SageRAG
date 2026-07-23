from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils.config import settings
from app.database.connection import init_db
from app.auth.router import router as auth_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB models
    init_db()
    yield
    # Shutdown: Clean up or close connections if needed
    pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise Knowledge Assistant utilizing Hybrid RAG and fine-tuned LLMs.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": f"Welcome to the {settings.PROJECT_NAME} API"
    }

# Mount auth router under API V1 prefix
app.include_router(
    auth_router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["Authentication"]
)
