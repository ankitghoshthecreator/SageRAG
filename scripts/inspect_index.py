"""
Inspect Index Script
====================
Prints the number of chunks in the Qdrant vector index.
Useful for verifying that documents have been successfully ingested.
"""
import httpx
from app.utils.config import settings

def main():
    qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    collection = settings.QDRANT_COLLECTION_NAME
    
    print(f"Inspecting Qdrant index at {qdrant_url} (collection: {collection})...")
    
    try:
        resp = httpx.get(f"{qdrant_url}/collections/{collection}")
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("result", {}).get("status")
            points = data.get("result", {}).get("points_count", 0)
            print(f"Collection status: {status}")
            print(f"Total chunks in vector index: {points}")
        elif resp.status_code == 404:
            print(f"Collection '{collection}' does not exist.")
        else:
            print(f"Error fetching collection info: {resp.text}")
    except httpx.RequestError as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Is Qdrant running?")

if __name__ == "__main__":
    main()
