"""
Seed Data Script
================
Creates a demo user and uploads a sample document using the REST API.
Useful for quick testing or CI setup.
"""
import httpx
import sys

API_BASE = "http://localhost:8000/api/v1"

def main():
    print("Seeding data...")
    try:
        # 1. Create User
        print("Creating demo user...")
        resp = httpx.post(f"{API_BASE}/auth/register", json={
            "username": "demo_user",
            "email": "demo@example.com",
            "password": "password123"
        })
        if resp.status_code == 201:
            print("  User created successfully.")
        elif resp.status_code == 400 and "already registered" in resp.text:
            print("  User already exists.")
        else:
            print(f"  Failed to create user: {resp.text}")
            sys.exit(1)

        # 2. Login
        print("Logging in...")
        resp = httpx.post(f"{API_BASE}/auth/login", data={
            "username": "demo_user",
            "password": "password123"
        })
        if resp.status_code != 200:
            print(f"  Login failed: {resp.text}")
            sys.exit(1)
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("  Logged in successfully.")

        # 3. Create a dummy text file
        print("Uploading dummy document...")
        dummy_content = b"SageRAG is an enterprise RAG assistant. It uses Qdrant and FastAPI."
        files = {"file": ("demo.txt", dummy_content, "text/plain")}
        
        resp = httpx.post(f"{API_BASE}/documents/upload", headers=headers, files=files)
        if resp.status_code == 202:
            print("  Document uploaded successfully.")
        else:
            print(f"  Document upload failed: {resp.text}")

    except httpx.RequestError as e:
        print(f"Error connecting to API: {e}")
        print("Is the server running?")

if __name__ == "__main__":
    main()
