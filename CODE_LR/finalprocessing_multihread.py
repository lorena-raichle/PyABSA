# === Imports ===
import os
import json
import ast
import math
import unicodedata
import re
import warnings
import io
import itertools
import threading
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import pandas as pd
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from pyabsa import AspectTermExtraction as ATEPC
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# === Settings ===
chunk_size = 10000
batch_size_db = 500
max_len = 512
window_sizes = [1, 2, 4]

TEST_MODE = False
N_TEST_CHUNKS = 2
num_workers = 8  # Number of parallel threads

filename = "/Users/lorenaraichle/Developer/ABSA/PyABSA/data/google_reviews_ALL_03june.json"
checkpoint_file = "/Users/lorenaraichle/Developer/ABSA/PyABSA/data/checkpoint_file.txt"

# === Count total reviews ===
with open(filename, "r", encoding="utf-8") as f:
    total_reviews = sum(1 for _ in f)

print(f"Total reviews in file: {total_reviews}")
n_chunks = (total_reviews + chunk_size - 1) // chunk_size

# === Load .env ===
load_dotenv()

conn_params = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# === Helper Functions ===
def normalize_for_matching(text):
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.replace("&", " ").replace("/", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def tokenize(text):
    return word_tokenize(text)

def split_review_into_parts(review, tokenizer, max_tokens=512):
    token_ids = tokenizer.encode(review, add_special_tokens=True)
    parts = []
    for i in range(0, len(token_ids), max_tokens):
        part_ids = token_ids[i:i+max_tokens]
        part_text = tokenizer.decode(part_ids)
        token_offset = i
        parts.append((part_text, token_offset))
    return parts

def correct_position(text, aspect, initial_pos, review_id, tokenizer, unmatched_counter):
    aspect_norm = normalize_for_matching(aspect)
    review_text_norm = normalize_for_matching(text)
    aspect_fused = aspect_norm.replace(" ", "")

    if aspect_norm not in review_text_norm and aspect_fused not in review_text_norm.replace(" ", ""):
        unmatched_counter['count'] += 1
        return None

    tokens = tokenize(text)
    token_sequence = [normalize_for_matching(t) for t in tokens]
    aspect_tokens = aspect_norm.split()
    candidate_positions = []

    for i in range(len(token_sequence)):
        joined_span = " ".join(token_sequence[i:i+len(aspect_tokens)+2])
        fused_span = joined_span.replace(" ", "")

        if aspect_norm in joined_span or aspect_fused in fused_span:
            candidate_positions.append(i)

    if not candidate_positions:
        for i, t in enumerate(token_sequence):
            if aspect_norm in t or aspect_fused in t.replace(" ", ""):
                candidate_positions.append(i)

    if not candidate_positions:
        unmatched_counter['count'] += 1
        return None

    closest_pos = min(candidate_positions, key=lambda x: abs(x - initial_pos))
    return closest_pos

def extract_snippet(text, aspect, pos, review_id, window, tokenizer, unmatched_counter):
    corrected_pos = correct_position(text, aspect, pos, review_id, tokenizer, unmatched_counter)
    if corrected_pos is None:
        return ""

    tokens = tokenize(text)
    start = max(corrected_pos - window, 0)
    end = min(corrected_pos + window + 1, len(tokens))
    return " ".join(tokens[start:end])

def safe_get_pos(pos):
    if isinstance(pos, list) and pos:
        return int(pos[0])
    try:
        return int(pos)
    except Exception:
        return None

# === Insert SQL ===
insert_sql = """
INSERT INTO aspect_sentiment_results (
    review_id, aspect, sentiment, confidence, position,
    snippet_1, snippet_2, snippet_4,
    embedding_1, embedding_2, embedding_4,
    aspect_id,
    truncated
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (aspect_id) DO UPDATE SET
    sentiment = EXCLUDED.sentiment,
    confidence = EXCLUDED.confidence,
    position = EXCLUDED.position,
    snippet_1 = EXCLUDED.snippet_1,
    snippet_2 = EXCLUDED.snippet_2,
    snippet_4 = EXCLUDED.snippet_4,
    embedding_1 = EXCLUDED.embedding_1,
    embedding_2 = EXCLUDED.embedding_2,
    embedding_4 = EXCLUDED.embedding_4,
    truncated = EXCLUDED.truncated;
"""

# === Process One Chunk ===
def process_chunk(df_chunk, unmatched_counter):

    # === Load local extractor and embedding model per thread ===
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_folder = os.path.expanduser("/Users/lorenaraichle/Developer/ABSA/PyABSA/checkpoints/ATEPC_MULTILINGUAL_CHECKPOINT")
    extractor = ATEPC.AspectExtractor(
        checkpoint=checkpoint_folder,
        auto_device=False,
        device=device,
        cal_perplexity=True,
    )
    tokenizer = extractor.tokenizer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    df_chunk['token_count'] = df_chunk['review'].apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=True))
    )
    df_chunk['too_long'] = df_chunk['token_count'] > max_len
    n_too_long = df_chunk['too_long'].sum()
    if n_too_long:
        warnings.warn(f"{n_too_long} reviews exceed {max_len} tokens and will be split.")

    sentences = []
    truncated_flags = []
    review_ids = []
    original_reviews = []
    token_offsets = []

    for i, row in df_chunk.iterrows():
        parts = split_review_into_parts(row['review'], tokenizer)
        for part_text, token_offset in parts:
            sentences.append(part_text)
            truncated_flags.append(len(parts) > 1)
            review_ids.append(row['review_id'])
            original_reviews.append(row['review'])
            token_offsets.append(token_offset)

    batch_size_absa = 64
    all_results = []
    num_batches = math.ceil(len(sentences) / batch_size_absa)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size_absa
        batch = sentences[start : start + batch_size_absa]

        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            batch_out = extractor.predict(
                batch,
                save_result=False,
                print_result=False,
                ignore_error=True
            )

        all_results.extend(batch_out)

    out_df = pd.DataFrame(all_results)[
        ['sentence', 'aspect', 'sentiment', 'confidence', 'position']
    ]

    out_df['truncated'] = truncated_flags
    out_df['review_id'] = review_ids
    out_df['review'] = original_reviews
    out_df['token_offset'] = token_offsets

    for col in ['aspect', 'sentiment', 'confidence', 'position']:
        out_df[col] = out_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    records = []
    for _, row in out_df.iterrows():
        for asp, sent, conf, pos in zip(row['aspect'], row['sentiment'], row['confidence'], row['position']):
            local_position = safe_get_pos(pos)
            if local_position is None:
                continue

            global_position = row['token_offset'] + local_position

            entry = {
                "review_id": row['review_id'],
                "review": row['review'],
                "aspect": asp,
                "sentiment": sent,
                "confidence": float(conf),
                "position": global_position,
                "truncated": row['truncated']
            }

            for win in window_sizes:
                entry[f"snippet_{win}"] = extract_snippet(row['review'], asp, global_position, row['review_id'], win, tokenizer, unmatched_counter)

            records.append(entry)

    df_multi_snippets = pd.DataFrame(records)

    for win in window_sizes:
        col = f"snippet_{win}"
        emb_col = f"embedding_{win}"
        df_multi_snippets[emb_col] = df_multi_snippets[col].apply(lambda x: model.encode(x).tolist())

    df_multi_snippets['aspect_number'] = df_multi_snippets.groupby('review_id').cumcount() + 1
    df_multi_snippets['aspect_id'] = (
        df_multi_snippets['review_id'].astype(str) + '_' + df_multi_snippets['aspect_number'].astype(str)
    )

    return df_multi_snippets

# === Per-thread worker ===
def process_and_save_chunk(chunk_idx, df_chunk):
    unmatched_counter = {'count': 0}

    print(f"\n========== Processing CHUNK {chunk_idx} ==========\n")
    df_multi_snippets = process_chunk(df_chunk, unmatched_counter)

    # Save parquet
    parquet_path = f"/Users/lorenaraichle/Developer/ABSA/PyABSA/results/processed_chunk_{chunk_idx}.parquet"
    df_multi_snippets.to_parquet(parquet_path)
    print(f"Saved CHUNK {chunk_idx} to {parquet_path}")

    # Upsert to DB
    rows = df_multi_snippets.to_dict(orient='records')
    print(f"Upserting {len(rows)} rows into DB...")

    conn = psycopg2.connect(
        **conn_params,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    cur = conn.cursor()

    try:
        for j in range(0, len(rows), batch_size_db):
            batch_rows = rows[j:j+batch_size_db]

            execute_batch(cur, insert_sql, [
                (
                    row['review_id'], row['aspect'], row['sentiment'],
                    float(row['confidence']), int(row['position']),
                    row.get('snippet_1'), row.get('snippet_2'), row.get('snippet_4'),
                    row['embedding_1'], row['embedding_2'], row['embedding_4'],
                    row['aspect_id'],
                    row['truncated']
                )
                for row in batch_rows
            ])

            conn.commit()
            print(f"Inserted batch {j//batch_size_db+1} / {len(rows)//batch_size_db+1} of chunk {chunk_idx}")

    finally:
        conn.close()

    # Return chunk_idx and unmatched count
    return chunk_idx, unmatched_counter['count']

# === Load processed chunks checkpoint ===
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        processed_chunks = set(int(line.strip()) for line in f.readlines())
else:
    processed_chunks = set()

# === Main Loop ===
total_unmatched_aspect_count = 0

try:
    reader = pd.read_json(filename, lines=True, chunksize=chunk_size)

    if TEST_MODE:
        chunk_iterator = itertools.islice(enumerate(reader), N_TEST_CHUNKS)
        total_chunks_to_process = N_TEST_CHUNKS
    else:
        chunk_iterator = enumerate(reader)
        total_chunks_to_process = n_chunks

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        futures = []
        for chunk_idx, df_chunk in tqdm(chunk_iterator, total=total_chunks_to_process, desc="Submitting chunks", unit="chunk"):

            if chunk_idx in processed_chunks:
                print(f"Skipping already processed CHUNK {chunk_idx}")
                continue

            future = executor.submit(process_and_save_chunk, chunk_idx, df_chunk)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing in parallel"):
            try:
                chunk_idx_done, unmatched_count = future.result()

                # Write to checkpoint after successful chunk
                with open(checkpoint_file, "a") as f:
                    f.write(f"{chunk_idx_done}\n")

                print(f"CHUNK {chunk_idx_done} completed and checkpoint updated.")
                total_unmatched_aspect_count += unmatched_count

            except Exception as e:
                print(" Error processing chunk:", e)

    print(f"\n=== ALL CHUNKS DONE ===\n")
    print(f"Total unmatched aspects: {total_unmatched_aspect_count}")

except Exception as e:
    print("Error during processing:", e)
