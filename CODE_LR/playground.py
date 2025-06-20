# import psycopg2
#
#
# from dotenv import load_dotenv
# import os
#
# # === Load .env ===
# load_dotenv()
#
# conn_params = {
#     "host": os.getenv("DB_HOST"),
#     "port": os.getenv("DB_PORT"),
#     "database": os.getenv("DB_NAME"),
#     "user": os.getenv("DB_USER"),
#     "password": os.getenv("DB_PASSWORD")
# }
#
#
#
# def fetch_existing_review_id(review_id):
#     print(f"Checking {review_id} review_id.")
#     conn = psycopg2.connect(**conn_params)
#     cur = conn.cursor()
#     query = "SELECT aspect_id FROM aspect_sentiment_results_sample WHERE review_id = %s;"
#     cur.execute(query, (review_id,))
#     existing_aspects = cur.fetchall()
#     conn.close()
#     print(f"Found {len(existing_aspects)} existing_aspects .")
#     return existing_aspects
#
#
#
#
# fetch_existing_review_id(191820)


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
import stopwordsiso as stopwordsiso
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from pyabsa import AspectTermExtraction as ATEPC
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")



import argparse


chunk_size = 500
batch_size_db = 500
max_len = 512
window_sizes = [4]

TEST_MODE = False
N_TEST_CHUNKS = 3
num_workers = 6

filename = "/Users/lorenaraichle/Developer/ABSA/PyABSA/data/test_reviews.json"
checkpoint_file = "/Users/lorenaraichle/Developer/ABSA/PyABSA/data/test_checkpoint_file.txt"

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

# def fetch_existing_aspect_ids(aspect_ids_to_check):
#     print(f"Checking {len(aspect_ids_to_check)} aspect_ids in DB...")
#     conn = psycopg2.connect(**conn_params)
#     cur = conn.cursor()
#     query = "SELECT aspect_id FROM aspect_sentiment_results_sample WHERE aspect_id = ANY(%s);"
#     cur.execute(query, (list(aspect_ids_to_check),))
#     existing_ids = set(row[0] for row in cur.fetchall())
#     conn.close()
#     print(f"Found {len(existing_ids)} existing aspect_ids.")
#     return existing_ids


def fetch_existing_review_id(review_id):
    print(f"Checking {review_id} review_id.")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    query = "SELECT aspect FROM aspect_sentiment_results_sample WHERE review_id = %s;"
    cur.execute(query, (review_id,))
    existing_aspects = cur.fetchall()
    conn.close()
    print(f"Found {len(existing_aspects)} existing_aspects .")
    return existing_aspects



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


all_stopwords = set()
for lang in ['en', 'de', 'fr', 'it']:
    all_stopwords.update(stopwordsiso.stopwords(lang))

def is_valid_aspect(aspect):
    aspect = aspect.strip().lower()
    if not aspect or len(aspect) < 2:
        return False
    if aspect in all_stopwords:
        return False
    if all(char in "!?,.:;[](){}\"'-" for char in aspect):  # punctuation-only
        return False
    return True


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
        span = token_sequence[i:i + len(aspect_tokens)]
        if span == aspect_tokens:
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

insert_sql = """
INSERT INTO aspect_sentiment_results_sample (
    review_id, aspect, sentiment, confidence, position,
    snippet_4,
    embedding_4,
    aspect_id,
    truncated,
    updated_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (aspect_id) DO UPDATE SET
    sentiment = EXCLUDED.sentiment,
    confidence = EXCLUDED.confidence,
    position = EXCLUDED.position,
    snippet_4 = EXCLUDED.snippet_4,
    embedding_4 = EXCLUDED.embedding_4,
    truncated = EXCLUDED.truncated,
    updated_at = NOW();
"""


# === Process One Chunk ===
def process_chunk(chunk_idx, df_chunk, unmatched_counter):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
  #  print(f"[CHUNK {chunk_idx}] Using device: {device}")

    checkpoint_path = "/Users/lorenaraichle/Developer/ABSA/PyABSA/CODE_LR/checkpoints/ATEPC_ENGLISH_CHECKPOINT/fast_lcf_atepc_English_cdw_apcacc_82.36_apcf1_81.89_atef1_75.43"
    extractor = ATEPC.AspectExtractor(
        checkpoint=checkpoint_path,
        auto_device=False,
        device=device,
        cal_perplexity=False,
    )

    tokenizer = extractor.tokenizer
    print(f"[CHUNK {chunk_idx}] Model and tokenizer loaded. Starting tokenization of {len(df_chunk)} reviews.")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    df_chunk['token_count'] = df_chunk['review'].apply(lambda t: len(tokenizer.encode(t, add_special_tokens=True)))
    df_chunk['too_long'] = df_chunk['token_count'] > max_len
    n_too_long = df_chunk['too_long'].sum()
    if n_too_long:
        warnings.warn(f"{n_too_long} reviews exceed {max_len} tokens and will be split.")

    sentences = []
    truncated_flags = []
    review_ids = []
    original_reviews = []
    token_offsets = []

    for _, row in df_chunk.iterrows():
        parts = split_review_into_parts(row['review'], tokenizer)
        for part_text, token_offset in parts:
            sentences.append(part_text)
            truncated_flags.append(len(parts) > 1)
            review_ids.append(row['review_id'])
            original_reviews.append(row['review'])
            token_offsets.append(token_offset)

    batch_size_absa = 32
    all_results = []
    num_batches = math.ceil(len(sentences) / batch_size_absa)

    print(f"[CHUNK {chunk_idx}] Starting ABSA prediction on {len(sentences)} split reviews in {num_batches} batches...")

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
        # if batch_out:
        #     print(f"[CHUNK {chunk_idx}][BATCH {batch_idx}] Sample output: {json.dumps(batch_out[0], indent=2)}")
        # else:
        #     print(f"[CHUNK {chunk_idx}][BATCH {batch_idx}] No output returned.")

        all_results.extend(batch_out)

    out_df = pd.DataFrame(all_results)[['sentence', 'aspect', 'sentiment', 'confidence', 'position']]
    out_df['truncated'] = truncated_flags
    out_df['review_id'] = review_ids
    out_df['review'] = original_reviews
    out_df['token_offset'] = token_offsets

    for col in ['aspect', 'sentiment', 'confidence', 'position']:
        out_df[col] = out_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    records = []
    snippets = []

    for _, row in out_df.iterrows():
        for asp, sent, conf, pos in zip(row['aspect'], row['sentiment'], row['confidence'], row['position']):
            if not asp or asp.strip().upper() == "CLS" or not is_valid_aspect(asp):
                print(f"[CHUNK {chunk_idx}] Skipping invalid aspect '{asp}' for review_id {row['review_id']}")
                continue

            local_position = safe_get_pos(pos)
            if local_position is None:
                continue

            global_position = row['token_offset'] + local_position

            snippet_4 = extract_snippet(row['review'], asp, global_position, row['review_id'], 4, tokenizer,
                                        unmatched_counter)

            if not asp.lower() in snippet_4.lower():
                print(f"[CHUNK {chunk_idx}] Warning: aspect '{asp}' not found in extracted snippet!")

            entry = {
                "review_id": row['review_id'],
                "review": row['review'],
                "aspect": asp,
                "sentiment": sent,
                "confidence": float(conf),
                "position": global_position,
                "truncated": row['truncated'],
                "snippet_4": snippet_4
            }

            records.append(entry)
            snippets.append(snippet_4)
    print(f"[CHUNK {chunk_idx}] Encoding {len(snippets)} snippets with SentenceTransformer...")
    embeddings = model.encode(snippets, batch_size=64, show_progress_bar=False)

    for i in range(len(records)):
        records[i]['embedding_4'] = embeddings[i].tolist()

    print(f"[CHUNK {chunk_idx}] Finished ABSA prediction. Total valid entries: {len(records)}")

    df_multi_snippets = pd.DataFrame(records)
    df_multi_snippets['aspect_number'] = df_multi_snippets.groupby('review_id').cumcount() + 1
    df_multi_snippets['aspect_id'] = (
            df_multi_snippets['review_id'].astype(str) + '_' + df_multi_snippets['aspect_number'].astype(str)
    )

    return df_multi_snippets



from concurrent.futures import ThreadPoolExecutor

def insert_batches_in_parallel(cur, conn, rows, chunk_idx):
    with ThreadPoolExecutor(max_workers=4) as thread_executor:
        futures = []
        for j in range(0, len(rows), batch_size_db):
            batch_rows = rows[j:j + batch_size_db]
            future = thread_executor.submit(execute_batch, cur, insert_sql, [
                (
                    row['review_id'], row['aspect'], row['sentiment'],
                    float(row['confidence']), int(row['position']),
                    None, None, row.get('snippet_4'),
                    None, None, row.get('embedding_4'),
                    row['aspect_id'],
                    row['truncated']
                ) for row in batch_rows
            ])
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()
            conn.commit()
            print(f"Inserted batch {i+1} / {len(futures)} of chunk {chunk_idx}")


# === Per-thread worker ===
def process_and_save_chunk(chunk_idx, df_chunk):
    unmatched_counter = {'count': 0}

    print(f"\n========== Processing CHUNK {chunk_idx} ==========\n")
    df_multi_snippets = process_chunk(chunk_idx, df_chunk, unmatched_counter)

    # === Filter out already inserted aspects ===
    records = df_multi_snippets.to_dict(orient='records')
    new_records = []
    existing_aspects_cache = {}

    for rec in records:
        review_id = rec['review_id']
        aspect = normalize_for_matching(rec['aspect'])

        if review_id not in existing_aspects_cache:
            existing_aspects_raw = fetch_existing_review_id(review_id)
            existing_aspects_cache[review_id] = set(
                normalize_for_matching(a[0]) for a in existing_aspects_raw
            )

        if aspect not in existing_aspects_cache[review_id]:
            new_records.append(rec)
        else:
            print(f"Skipping duplicate aspect '{rec['aspect']}' for review_id {review_id}")

    print(f"CHUNK {chunk_idx}: {len(new_records)} new aspect(s) will be upserted.")

    # === Reassign aspect IDs ===
    for idx, rec in enumerate(new_records):
        rec['aspect_number'] = idx + 1
        rec['aspect_id'] = f"{rec['review_id']}_{rec['aspect_number']}"

    # === Insert into DB ===
    conn = psycopg2.connect(
        **conn_params,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    cur = conn.cursor()
    try:
        insert_batches_in_parallel(cur, conn, new_records, chunk_idx)
    finally:
        conn.close()

    return chunk_idx, unmatched_counter['count']


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    # === Load processed chunks checkpoint ===
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            with open(checkpoint_file, "r") as f:
                processed_chunks = set(int(line.strip()) for line in f if line.strip())

    else:
        processed_chunks = set()
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

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc="Processing in parallel"):
                try:
                    chunk_idx_done, unmatched_count = future.result()

                    with open(checkpoint_file, "a") as f:
                        f.write(f"{chunk_idx_done}\n")

                    print(f"CHUNK {chunk_idx_done} completed and checkpoint updated.")
                    total_unmatched_aspect_count += unmatched_count

                except Exception as e:
                    print(f"Error processing a chunk: {e}")

        print(f"\n=== ALL CHUNKS DONE ===\n")
        print(f"Total unmatched aspects: {total_unmatched_aspect_count}")

    except Exception as e:
        print("Error during processing:", e)


