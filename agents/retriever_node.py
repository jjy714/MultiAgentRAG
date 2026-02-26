import os
import requests
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
EMBEDDING_PORT = os.getenv("EMBEDDING_PORT", "8080")


class Retriever:

    def __init__(self):
        self.client = QdrantClient(host=SERVER_HOST, port=int(QDRANT_PORT))
        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    def vectorize(self, query: str) -> np.ndarray:
        response = requests.post(
            f"http://{SERVER_HOST}:{EMBEDDING_PORT}/v1/embed",
            json={"query": query},
        )
        embedding = response.json().get("result", {}).get("embedding", [])
        return np.array(embedding)

    def vector_search(self, query: str, k: int = 5):
        return self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=self.vectorize(query),
            limit=k,
        )


def retriever_node(state: dict) -> dict:
    """Retrieves top-k relevant documents from Qdrant for the given question."""
    query = state.get("question", "")
    if isinstance(query, dict):
        query = query.get("task", "")

    retriever = Retriever()
    result = retriever.vector_search(str(query))
    points = result.points

    chunk_ids = [str(p.payload.get("chunk_id")) for p in points]
    page_contents = [p.payload.get("page_content", "") for p in points]

    print(f"[retriever_node] Retrieved {len(page_contents)} documents.")
    return {
        "question": query,
        "documents": page_contents,
        "doc_ids": chunk_ids,
    }
