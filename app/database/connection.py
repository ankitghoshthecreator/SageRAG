import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.config import settings
from app.database.models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sagerag.database")

database_url = settings.DATABASE_URL
connect_args = {}

# Check for SQLite or setup fallback
if database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif database_url.startswith("postgresql"):
    # We attempt to connect to Postgres. If it fails (e.g., service not running),
    # we fall back to SQLite to ensure a seamless setup experience.
    try:
        temp_engine = create_engine(database_url)
        with temp_engine.connect() as conn:
            pass
        temp_engine.dispose()
        logger.info("Successfully connected to primary PostgreSQL database.")
    except Exception as e:
        logger.warning(
            f"Failed to connect to primary PostgreSQL database at {database_url}. Error: {e}"
        )
        logger.warning("Falling back to local SQLite database: sqlite:///sage_rag.db")
        database_url = "sqlite:///sage_rag.db"
        connect_args = {"check_same_thread": False}

engine = create_engine(database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables in the database if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        raise e

def get_db():
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
