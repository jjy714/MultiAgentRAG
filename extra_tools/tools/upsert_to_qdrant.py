import os
import gc
import uuid
import glob
import torch
import polars as pl
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys

sys.stdout.reconfigure(line_buffering=True)

INPUT_PARQUET = "data/capstone/kilt_dpr_100w_fixed.parquet"
OUTPUT_DIR = "data/capstone/kilt_vectors_ready"

MODEL_PATH = "/datadrive/data/Qwen3-Embedding-0.6B"

CHUNK_SIZE = 100_000
MICRO_BATCH_SIZE = 512
MAX_SEQ_LENGTH = 512
TEXT_SLICE_LEN = 2000
DEVICE = "cuda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_safely(model, texts, batch_size):
    if batch_size < 1:
        raise RuntimeError("Batch size dropped to 0! Input text might be too massive.")
    try:
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,  # Secondary GPU progress bar
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        new_batch = batch_size // 2
        print(f"⚠️ OOM detected! Reducing batch size: {batch_size} -> {new_batch}")
        return encode_safely(model, texts, new_batch)


def main():
    print("⚡ Loading Model on A100...")
    model = SentenceTransformer(
        MODEL_PATH,
        device=DEVICE,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    model.eval()

    print(f"📂 Scanning Dataset: {INPUT_PARQUET}")
    lf = pl.scan_parquet(INPUT_PARQUET)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"📊 Total rows to process: {total_rows:,}")

    # --- NEW RESUME LOGIC ---
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "vectors_part_*.parquet"))

    if existing_files:
        # Find the highest part number (e.g., vectors_part_0004.parquet -> 4)
        highest_part = max(
            [
                int(os.path.basename(f).split("_part_")[1].split(".parquet")[0])
                for f in existing_files
            ]
        )
        part_idx = highest_part + 1
        offset = part_idx * CHUNK_SIZE
        print(
            f"⏩ Found {len(existing_files)} existing files. Resuming from part {part_idx} (offset: {offset:,})..."
        )
    else:
        part_idx = 0
        offset = 0
    # Start progress bar at the resumed offset
    pbar = tqdm(total=total_rows, initial=offset, desc="🚀 Embedding")

    while offset < total_rows:
        df = lf.slice(offset, CHUNK_SIZE).collect()
        if df.is_empty():
            break

        df = df.with_columns(
            (
                pl.col("title") + ": " + pl.col("text").str.slice(0, TEXT_SLICE_LEN)
            ).alias("input_text")
        )
        texts = df["input_text"].to_list()

        with torch.inference_mode():
            vectors = encode_safely(model, texts, MICRO_BATCH_SIZE)
        torch.cuda.synchronize()

        out_rows = []
        for i, row in enumerate(df.iter_rows(named=True)):
            uid = f"{row['wikipedia_id']}_{row['chunk_id']}"
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, uid))

            out_rows.append(
                {
                    "id": doc_id,
                    "vector": vectors[i].tolist(),
                    "title": row["title"],
                    "text": row["text"],
                    "wikipedia_id": row["wikipedia_id"],
                    "chunk_id": row["chunk_id"],
                }
            )

        out_path = os.path.join(OUTPUT_DIR, f"vectors_part_{part_idx:04d}.parquet")
        pl.DataFrame(out_rows).write_parquet(out_path)

        pbar.update(len(out_rows))
        offset += len(out_rows)
        part_idx += 1

        del df, texts, vectors, out_rows
        gc.collect()

    print("\n🎉 Step 1 Complete! All vectors embedded and saved.")


if __name__ == "__main__":
    main()
