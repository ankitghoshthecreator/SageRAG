"""
Health Check Script
===================
Verifies the status of the main API and its dependencies.
"""
import httpx
import sys

def check_service(name: str, url: str) -> bool:
    try:
        resp = httpx.get(url, timeout=5.0)
        if resp.status_code == 200:
            print(f"[OK] {name}")
            return True
        else:
            print(f"[FAIL] {name} (Status: {resp.status_code})")
            return False
    except httpx.RequestError:
        print(f"[FAIL] {name} (Connection Error)")
        return False

def main():
    print("--- SageRAG Health Check ---")
    
    services = {
        "FastAPI Backend": "http://localhost:8000/",
        # Add other URLs here if you want to check them directly, e.g., Qdrant, MinIO.
        # But checking the root of FastAPI gives a good indication since it returns {"status": "healthy"}.
    }

    all_healthy = True
    for name, url in services.items():
        if not check_service(name, url):
            all_healthy = False
            
    print("----------------------------")
    if all_healthy:
        print("Status: All systems go.")
        sys.exit(0)
    else:
        print("Status: Some systems failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
