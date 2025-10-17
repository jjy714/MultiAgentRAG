from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import getpass
from dotenv import load_dotenv
import os

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL=os.getenv("QDRANT_URL")

client = QdrantClient(":memory:")

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

docs = []  # put docs here

def document_from_collection():
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        prefer_grpc=True,
        collection_name="my_documents",
    )
    return qdrant

def initiate_qdrant_client():
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )
    return qdrant