import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("sagerag.vector_store")

class ChromaStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            import chromadb
            from app.utils.config import settings

            # Local path to store persistent vector database
            db_path = os.path.join(os.getcwd(), "local_data", "chroma_db")
            os.makedirs(db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=db_path)
            collection_name = settings.QDRANT_COLLECTION_NAME # Reuse the settings name
            
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"} # use cosine distance metric
            )
            logger.info(f"Connected to ChromaDB at {db_path}, collection: {collection_name}")
        except Exception as e:
            logger.error(f"Cannot initialize ChromaDB: {e}")
            raise e

    def upsert(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        if not chunks:
            return
        
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "document_id": c["document_id"],
                "filename": c["filename"],
                "page_number": int(c["page_number"]),
                "chunk_index": int(c["chunk_index"]),
            }
            metadatas.append(meta)

        self.collection.add(
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
            ids=ids
        )
        logger.info(f"ChromaDB upserted {len(chunks)} chunks.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where_filter = None
        if filter_doc_id:
            where_filter = {"document_id": filter_doc_id}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )

        formatted = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                meta = metadatas[i]
                # Chroma distance is cosine distance (1 - cosine_similarity).
                # Convert it to a similarity score (1 - distance).
                score = 1.0 - float(distances[i])
                formatted.append({
                    "chunk_id": ids[i],
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "page_number": int(meta["page_number"]),
                    "chunk_index": int(meta["chunk_index"]),
                    "text": documents[i],
                    "score": score
                })
        return formatted

    def delete_by_document_id(self, document_id: str):
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(f"Deleted vectors for document {document_id} from ChromaDB.")

# Initialize the vector store instance
vector_store = ChromaStore()
