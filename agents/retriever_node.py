import json
import os
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
EMBEDDING_PORT = os.getenv("EMBEDDING_PORT", "8080")


_SYSTEM_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _SYSTEM_ROOT / "question_embeddings.json"


def _load_embedding_cache() -> dict[str, list[float]]:
    # Priority 1: explicit env var
    env_path = os.getenv("EMBEDDING_CACHE_PATH", "")
    candidates = [Path(env_path)] if env_path else []
    # Priority 2: well-known location next to the system root
    candidates.append(_DEFAULT_CACHE)

    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data: dict[str, list[float]] = json.load(f)
            print(f"[retriever] Loaded {len(data)} pre-computed embeddings from {p}")
            return data

    return {}


### Retriever class that connects to Qdrant and performs vector similarity search
class Retriever:
    def __init__(self):
        self.client = QdrantClient(host=SERVER_HOST, port=int(QDRANT_PORT))
        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        self._cache: dict[str, list[float]] = _load_embedding_cache()

    ## Return a pre-computed embedding vector, finding the nearest cached key if no exact match
    def vectorize(self, query: str) -> np.ndarray:
        if query in self._cache:
            return np.array(self._cache[query], dtype=np.float32)
        # Fuzzy fallback: find the cached key with the highest token-overlap (Jaccard)
        if self._cache:
            query_tokens = set(query.lower().split())
            best_key = max(
                self._cache,
                key=lambda k: len(query_tokens & set(k.lower().split()))
                              / max(len(query_tokens | set(k.lower().split())), 1),
            )
            score = len(query_tokens & set(best_key.lower().split())) \
                    / max(len(query_tokens | set(best_key.lower().split())), 1)
            print(f"[retriever] Cache miss — using nearest cached key (Jaccard={score:.2f}): '{best_key[:80]}'")
            return np.array(self._cache[best_key], dtype=np.float32)
        return np.array([])

    ## Search Qdrant for the top-k most similar document vectors
    def vector_search(self, query: str, k: int = 5):
        vec = self.vectorize(query)
        if vec.size == 0:
            return type("Result", (), {"points": []})()
        return self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=vec,
            limit=k,
        )


## LangGraph node that retrieves top-k relevant documents from Qdrant for the given question
def retriever_node(state: dict) -> dict:
    query = state.get("question", "")
    if isinstance(query, dict):
        query = query.get("task", "")

    retriever = Retriever()
    result = retriever.vector_search(str(query))
    points = result.points

    chunk_ids = [str(p.payload.get("chunk_id")) for p in points]
    page_contents = [
        f"{p.payload.get('title', '')}: {p.payload.get('text', '')}"
        for p in points
    ]

    print(f"[retriever_node] Retrieved {len(page_contents)} documents.")
    return {
        "question": query,
        "documents": page_contents,
        "doc_ids": chunk_ids,
    }
