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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

_SYSTEM_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _SYSTEM_ROOT / "question_embeddings.json"


def _load_embedding_cache() -> dict[str, list[float]]:
    env_path = os.getenv("EMBEDDING_CACHE_PATH", "")
    candidates = [Path(env_path)] if env_path else []
    candidates.append(_DEFAULT_CACHE)

    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data: dict[str, list[float]] = json.load(f)
            print(f"[retriever] Loaded {len(data)} pre-computed embeddings from {p}")
            return data

    return {}


def _cache_path() -> Path:
    env_path = os.getenv("EMBEDDING_CACHE_PATH", "")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    if _DEFAULT_CACHE.exists():
        return _DEFAULT_CACHE
    return _DEFAULT_CACHE


def _embed_via_server(query: str) -> list[float] | None:
    url = f"http://{SERVER_HOST}:{EMBEDDING_PORT}/v1/embeddings"
    try:
        resp = requests.post(
            url,
            json={"model": EMBEDDING_MODEL, "input": [query]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        print(f"[retriever] Embedding server error: {exc}")
        return None


class Retriever:
    def __init__(self):
        self.client = QdrantClient(host=SERVER_HOST, port=int(QDRANT_PORT))
        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        self._cache: dict[str, list[float]] = _load_embedding_cache()
        self._cache_file: Path = _cache_path()

    def _save_to_cache(self, query: str, vector: list[float]) -> None:
        self._cache[query] = vector
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as exc:
            print(f"[retriever] Could not persist cache: {exc}")

    def _nearest_cached(self, query: str) -> np.ndarray:
        """Find the closest cached query by Jaccard token similarity.

        Stopwords are filtered before comparison so common words like "the" or
        "what" don't inflate similarity scores between unrelated questions.
        """
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "what", "which", "who", "whom", "when", "where", "why", "how",
            "of", "in", "on", "at", "to", "for", "from", "by", "with",
            "and", "or", "but", "not", "does", "do", "did", "has", "have",
            "had", "will", "would", "could", "should", "may", "might",
            "this", "that", "these", "those", "it", "its", "about",
        }
        def _tokens(s: str) -> set:
            return {w for w in s.lower().split() if w not in _STOPWORDS and len(w) > 1}

        query_tokens = _tokens(query)
        if not query_tokens or not self._cache:
            return np.array([])

        best_key = max(
            self._cache,
            key=lambda k: len(query_tokens & _tokens(k))
                          / max(len(query_tokens | _tokens(k)), 1),
        )
        score = len(query_tokens & _tokens(best_key)) / max(len(query_tokens | _tokens(best_key)), 1)
        print(f"[retriever] Nearest cached key (Jaccard={score:.2f}): '{best_key[:80]}'")
        return np.array(self._cache[best_key], dtype=np.float32)

    def vectorize(self, query: str) -> np.ndarray:
        if query in self._cache:
            return np.array(self._cache[query], dtype=np.float32)

        print(f"[retriever] Cache miss for '{query[:80]}' — calling embedding server")
        vector = _embed_via_server(query)
        if vector is not None:
            self._save_to_cache(query, vector)
            return np.array(vector, dtype=np.float32)

        print(f"[retriever] Server unavailable — using nearest cached key")
        return self._nearest_cached(query)

    def vector_search(self, query: str, k: int = 5):
        vec = self.vectorize(query)
        if vec.size == 0:
            return type("Result", (), {"points": []})()
        return self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=vec,
            limit=k,
        )


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
    for i, doc in enumerate(page_contents):
        preview = " ".join(doc.split()[:50])
        print(f"[retriever_node] Doc {i+1}: {preview}")
    return {
        "question": query,
        "documents": page_contents,
        "doc_ids": chunk_ids,
    }
