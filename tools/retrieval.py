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

# Initialize HuggingFace sentence embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
client = QdrantClient(":memory:")


## Initialize a QdrantVectorStore client in the specified retrieval mode
def initiate_qdrant_client(mode: str):
    """
    args   : {
        "mode (str)": "retrieval mode — 'dense', 'sparse', or 'hybrid'"
    }
    return : {
        "QdrantVectorStore": "configured Qdrant vector store instance"
    }
    """
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

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

## LangChain tool that performs dense vector similarity search against Qdrant
@tool
def dense_search(query):
    """
    args   : {
        "query (str)": "search query text"
    }
    return : {
        "List[Document]": "most similar documents using dense vector retrieval"
    }
    """
    qdrant = initiate_qdrant_client("dense")
    found_docs = qdrant.similarity_search(query)
    return found_docs

## LangChain tool that performs sparse keyword-based similarity search against Qdrant
@tool
def sparse_search(query):
    """
    args   : {
        "query (str)": "search query text"
    }
    return : {
        "List[Document]": "most similar documents using sparse vector retrieval"
    }
    """
    qdrant = initiate_qdrant_client("sparse")
    found_docs = qdrant.similarity_search(query)
    return found_docs

## LangChain tool that performs hybrid dense+sparse similarity search against Qdrant
@tool
def hybrid_search(query):
    """
    args   : {
        "query (str)": "search query text"
    }
    return : {
        "List[Document]": "most similar documents using hybrid retrieval"
    }
    """
    qdrant = initiate_qdrant_client("hybrid")
    found_docs = qdrant.similarity_search(query)
    return found_docs