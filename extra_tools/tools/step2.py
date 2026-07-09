import os
import glob
import gc
import polars as pl
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

VECTOR_DIR = "data/capstone/kilt_vectors_ready"
CHECKPOINT_FILE = "upload_progress.txt"
QDRANT_HOST = "0.0.0.0"
QDRANT_PORT = 6333
COLLECTION_NAME = "kilt_dpr_100w"
VECTOR_SIZE = 1024
TIMEOUT = 120
UPLOAD_BATCH_SIZE = 1000


def get_last_successful_index():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            content = f.read().strip()
            return int(content) if content else -1
    return -1


def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))


def main():
    print(f"Connecting to Qdrant at {QDRANT_HOST}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=TIMEOUT)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    files = sorted(glob.glob(os.path.join(VECTOR_DIR, "*.parquet")))
    total_files = len(files)

    if total_files == 0:
        print(f"No parquet files found in {VECTOR_DIR}")
        return

    last_index = get_last_successful_index()
    print(f"Found {total_files} vector files.")
    if last_index >= 0:
        print(f"Resuming from file index {last_index + 1}...")

    for i, filepath in enumerate(files):
        if i <= last_index:
            continue

        filename = os.path.basename(filepath)
        print(f"\nUploading [{i+1}/{total_files}]: {filename}")

        try:
            df = pl.read_parquet(filepath)
            rows = df.to_dicts()
            pbar = tqdm(total=len(rows), desc="   Streaming to Qdrant")

            for batch_start in range(0, len(rows), UPLOAD_BATCH_SIZE):
                batch_rows = rows[batch_start : batch_start + UPLOAD_BATCH_SIZE]
                points = [
                    PointStruct(
                        id=row["id"],
                        vector=row["vector"],
                        payload={
                            "text": row["text"],
                            "title": row["title"],
                            "wikipedia_id": row["wikipedia_id"],
                            "chunk_id": row["chunk_id"],
                        },
                    )
                    for row in batch_rows
                ]
                client.upsert(
                    collection_name=COLLECTION_NAME, points=points, wait=False
                )
                pbar.update(len(batch_rows))

            pbar.close()
            save_checkpoint(i)
            print("   File uploaded and checkpoint saved.")

            del df, rows, points
            gc.collect()

        except Exception as e:
            print(f"\n   Failed to upload {filename}: {e}")
            print("   Stopping. Fix the connection and re-run to resume.")
            break

    if get_last_successful_index() == total_files - 1:
        print("\nAll files uploaded successfully to Qdrant!")


if __name__ == "__main__":
    main()
