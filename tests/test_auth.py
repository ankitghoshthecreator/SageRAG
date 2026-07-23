import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database.connection import get_db
from app.database.models import Base

TEST_DB_FILE = "./test_sage_rag.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_FILE}"

test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    # Make sure test database is empty
    if os.path.exists(TEST_DB_FILE):
        os.remove(TEST_DB_FILE)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    # Clean up file afterwards
    if os.path.exists(TEST_DB_FILE):
        try:
            os.remove(TEST_DB_FILE)
        except Exception:
            pass

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply overrides to fastapi app
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_register_user():
    # Successful Registration
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "testuser@example.com",
            "password": "securepassword123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "testuser@example.com"
    assert "id" in data
    assert "hashed_password" not in data

    # Attempt Duplicate Registration (Username)
    response_dup = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "other@example.com",
            "password": "securepassword123"
        }
    )
    assert response_dup.status_code == 400
    assert "Username already registered" in response_dup.json()["detail"]

def test_login_user():
    # Login with valid credentials
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": "testuser",
            "password": "securepassword123"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"].lower() == "bearer"
    token = data["access_token"]

    # Login with invalid credentials
    response_invalid = client.post(
        "/api/v1/auth/login",
        data={
            "username": "testuser",
            "password": "wrongpassword"
        }
    )
    assert response_invalid.status_code == 401

    # Access /me with JWT token
    headers = {"Authorization": f"Bearer {token}"}
    response_me = client.get("/api/v1/auth/me", headers=headers)
    assert response_me.status_code == 200
    data_me = response_me.json()
    assert data_me["username"] == "testuser"
    assert data_me["email"] == "testuser@example.com"

def test_unauthenticated_me():
    # Access /me without authentication headers
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
