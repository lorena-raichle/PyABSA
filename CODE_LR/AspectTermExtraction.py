import torch
import pandas as pd
import warnings
from transformers import DebertaV2Config
from pyabsa import AspectTermExtraction as ATEPC
from tqdm import tqdm
import math
import sys
from contextlib import redirect_stdout
import io
import os
from contextlib import redirect_stdout, redirect_stderr
import io


chunk_size = 10000
reader = pd.read_json(
    "/Users/lorenaraichle/Developer/ABSA/PyABSA/data/google_reviews_ALL_03june.json",
    lines=True,
    chunksize=chunk_size
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint_folder = os.path.expanduser(
    "/PyABSA/checkpoints/ATEPC_MULTILINGUAL_CHECKPOINT"
)

extractor = ATEPC.AspectExtractor(
    checkpoint=checkpoint_folder,
    auto_device=False,
    device=device,
    cal_perplexity=True,
)

max_len = 512


tokenizer = extractor.tokenizer
df['token_count'] = df['review'].apply(
    lambda t: len(tokenizer.encode(t, add_special_tokens=True))
)
df['too_long']    = df['token_count'] > max_len

n_too_long = df['too_long'].sum()
if n_too_long:
    warnings.warn(
        f"{n_too_long} reviews exceed {max_len} tokens and will be truncated."
    )



# Batch‐predict as before
sentences = df['review'].tolist()
batch_size = 64
all_results = []

num_batches = math.ceil(len(sentences) / batch_size)


for batch_idx in tqdm(range(num_batches), desc="Main Batches", unit="batch"):
    start = batch_idx * batch_size
    batch = sentences[start : start + batch_size]

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
df_final = pd.concat([df.reset_index(drop=True), out_df.drop(columns=['sentence'])], axis=1)
df_final.to_csv(
    "/Users/lorenaraichle/Developer/ABSA/PyABSA/results/ate_batch_results_10_000.csv",
    index=False
)

print(f"Processed {len(df_final)} reviews; flagged {n_too_long} as too long.")
