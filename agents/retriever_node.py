import os
import requests
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# Environment variables for Qdrant and embedding service configuration
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
EMBEDDING_PORT = os.getenv("EMBEDDING_PORT", "8080")


### Retriever class that connects to Qdrant and performs vector similarity search
class Retriever:
    """
    args   : {}
    return : {
        "Retriever": "instance with an active QdrantClient connection"
    }
    """

    ## Initialize the QdrantClient and ensure the target collection exists
    def __init__(self):
        """
        args   : {}
        return : {
            "None": "initializes self.client"
        }
        """
        self.client = QdrantClient(host=SERVER_HOST, port=int(QDRANT_PORT))
        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    ## Send the query text to the embedding service and return a numpy vector
    def vectorize(self, query: str) -> np.ndarray:
        """
        args   : {
            "query (str)": "plain-text search query"
        }
        return : {
            "np.ndarray": "embedding vector of the query"
        }
        """
        response = requests.post(
            f"http://{SERVER_HOST}:{EMBEDDING_PORT}/v1/embed",
            json={"query": query},
        )
        embedding = response.json().get("result", {}).get("embedding", [])
        return np.array(embedding)

    ## Search Qdrant for the top-k most similar document vectors
    def vector_search(self, query: str, k: int = 5):
        """
        args   : {
            "query (str)": "search query text",
            "k (int)": "number of results to return"
        }
        return : {
            "QueryResponse": "Qdrant query result containing scored points"
        }
        """
        return self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=self.vectorize(query),
            limit=k,
        )


## LangGraph node that retrieves top-k relevant documents from Qdrant for the given question
def retriever_node(state: dict) -> dict:
    """
    args   : {
        "state (dict)": "graph state containing 'question' key"
    }
    return : {
        "dict": "updated state with 'question', 'documents', and 'doc_ids' keys"
    }
    """
    query = state.get("question", "")
    if isinstance(query, dict):
        # Extract plain-text task string if question is a structured dict
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
