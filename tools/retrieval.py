from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL=os.getenv("QDRANT_URL")
client = QdrantClient(":memory:")


# @tool
# def document_from_collection():
#     qdrant = QdrantVectorStore.from_documents(
#         docs,
#         embeddings,
#         url=QDRANT_URL,
#         prefer_grpc=True,
#         collection_name=QDRANT_COLLECTION_NAME,
#     )
#     return qdrant

def initiate_qdrant_client(mode: str):
    if mode == "dense":
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            url=QDRANT_URL,
            retrieval_mode=RetrievalMode.DENSE,
        )
    if mode == "sparse":
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            url=QDRANT_URL,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse",
        )
    if mode == "hybrid":
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=QDRANT_COLLECTION_NAME,
            url=QDRANT_URL,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    
    return qdrant


@tool
def dense_search(query):
    qdrant = initiate_qdrant_client("dense")
    found_docs = qdrant.similarity_search(query)
    return found_docs

@tool
def sparse_search(query):
    qdrant = initiate_qdrant_client("sparse")
    found_docs = qdrant.similarity_search(query)
    return found_docs

@tool
def hybrid_search(query):
    qdrant = initiate_qdrant_client("hybrid")
    found_docs = qdrant.similarity_search(query)
    return found_docs