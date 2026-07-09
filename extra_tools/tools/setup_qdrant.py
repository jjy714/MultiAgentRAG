"""
One-shot setup: download Qwen3-Embedding-0.6B, create example documents
for the 3 test questions, embed everything, create the kilt_dpr_100w
collection in Qdrant, and upload. Also writes query embeddings to
question_embeddings.json so the retriever cache is pre-warmed.
"""

import json
import uuid
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "kilt_dpr_100w"
VECTOR_SIZE = 1024
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "question_embeddings.json"

# Example documents — one passage per test question
DOCUMENTS = [
    {
        "title": "Lung Cancer",
        "text": (
            "Lung cancer is one of the most common and serious types of cancer. "
            "The main symptoms include a persistent cough that does not go away, "
            "chest pain that worsens with deep breathing or coughing, shortness of "
            "breath, coughing up blood or rust-colored sputum, fatigue and weakness, "
            "unexplained weight loss, loss of appetite, hoarseness, and recurrent "
            "respiratory infections such as bronchitis or pneumonia. Symptoms often "
            "appear only after the disease has progressed, making early detection difficult."
        ),
        "wikipedia_id": "lung_cancer_001",
        "chunk_id": 0,
    },
    {
        "title": "Lung Cancer Diagnosis",
        "text": (
            "Diagnosing lung cancer typically involves imaging tests such as chest X-rays "
            "and CT scans, followed by a biopsy to confirm the presence of cancer cells. "
            "The two main types are non-small cell lung cancer (NSCLC), which accounts for "
            "about 85% of cases, and small cell lung cancer (SCLC). Smoking is the leading "
            "cause, responsible for approximately 85% of all lung cancer diagnoses."
        ),
        "wikipedia_id": "lung_cancer_002",
        "chunk_id": 1,
    },
    {
        "title": "Breast Cancer Treatment",
        "text": (
            "Stage 2 breast cancer treatment typically involves a combination of approaches. "
            "Surgery options include lumpectomy (removal of the tumor and surrounding tissue) "
            "or mastectomy (removal of the entire breast). Chemotherapy is commonly administered "
            "either before surgery (neoadjuvant) to shrink the tumor or after surgery (adjuvant) "
            "to eliminate remaining cancer cells. Radiation therapy follows surgery to reduce "
            "recurrence risk. Hormone receptor-positive tumors are treated with hormone therapy "
            "such as tamoxifen or aromatase inhibitors. HER2-positive tumors receive targeted "
            "therapy with trastuzumab (Herceptin)."
        ),
        "wikipedia_id": "breast_cancer_001",
        "chunk_id": 0,
    },
    {
        "title": "Breast Cancer Staging",
        "text": (
            "Breast cancer staging is based on tumor size, lymph node involvement, and metastasis. "
            "Stage 2 is divided into 2A and 2B. Stage 2A includes tumors up to 2 cm with cancer "
            "in 1-3 nearby lymph nodes, or tumors between 2-5 cm without lymph node involvement. "
            "Stage 2B includes tumors between 2-5 cm with 1-3 positive lymph nodes, or tumors "
            "larger than 5 cm with no lymph node involvement. Five-year survival rates for stage "
            "2 breast cancer are generally above 80%."
        ),
        "wikipedia_id": "breast_cancer_002",
        "chunk_id": 1,
    },
    {
        "title": "Type 2 Diabetes Risk Factors",
        "text": (
            "Type 2 diabetes develops when the body cannot use insulin effectively. Key risk "
            "factors include obesity (especially abdominal fat), physical inactivity, family "
            "history of diabetes, age over 45, prediabetes, a history of gestational diabetes, "
            "polycystic ovarian syndrome (PCOS), and high blood pressure or abnormal cholesterol "
            "levels. Certain ethnic groups including African American, Hispanic/Latino, Native "
            "American, and Asian American populations face higher risk. Poor diet high in "
            "processed foods and sugary beverages also significantly increases risk."
        ),
        "wikipedia_id": "diabetes_001",
        "chunk_id": 0,
    },
    {
        "title": "Type 2 Diabetes Prevention",
        "text": (
            "Type 2 diabetes can often be prevented or delayed through lifestyle changes. "
            "Losing 5-7% of body weight, engaging in 150 minutes of moderate physical activity "
            "per week, eating a balanced diet rich in fiber and low in refined carbohydrates, "
            "and quitting smoking are all effective prevention strategies. The Diabetes Prevention "
            "Program study showed that lifestyle intervention reduced diabetes risk by 58% in "
            "high-risk individuals. Metformin may also be prescribed for those at very high risk."
        ),
        "wikipedia_id": "diabetes_002",
        "chunk_id": 1,
    },
]

TEST_QUERIES = [
    "What are the main symptoms of lung cancer?",
    "What is the standard treatment protocol for stage 2 breast cancer?",
    "What are the risk factors for developing type 2 diabetes?",
]


def main():
    print("Loading Qwen3-Embedding-0.6B ...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    print(
        f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}"
    )

    texts = [f"{d['title']}: {d['text']}" for d in DOCUMENTS]
    print(f"\nEmbedding {len(texts)} documents ...")
    doc_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"\nEmbedding {len(TEST_QUERIES)} test queries ...")
    query_vectors = model.encode(
        TEST_QUERIES, normalize_embeddings=True, show_progress_bar=True
    )

    # Write query cache
    cache: dict = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    for q, v in zip(TEST_QUERIES, query_vectors):
        cache[q] = v.tolist()
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"Query cache written to {CACHE_PATH} ({len(cache)} entries)")

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if client.collection_exists(COLLECTION_NAME):
        print(
            f"\nCollection '{COLLECTION_NAME}' already exists — deleting and recreating."
        )
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created (dim={VECTOR_SIZE}, Cosine).")

    points = []
    for doc, vec in zip(DOCUMENTS, doc_vectors):
        uid = f"{doc['wikipedia_id']}_{doc['chunk_id']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, uid))
        points.append(
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={
                    "title": doc["title"],
                    "text": doc["text"],
                    "wikipedia_id": doc["wikipedia_id"],
                    "chunk_id": doc["chunk_id"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nUpload complete. Collection now has {info.points_count} points.")
    print(
        "\nDone. Update QDRANT_COLLECTION_NAME=kilt_dpr_100w in .env.dev to use this collection."
    )


if __name__ == "__main__":
    main()
