"""Build data/sample.csv: 3,000 reviews per sentiment class from Amazon Reviews 2023.

The category file is 9.3 GB of JSON lines, sorted by user. Reading only the top of the
file would give us a handful of users, so instead we download many small random byte
ranges from across the whole file (HTTP Range requests) and parse the complete lines
inside each one. That gives a random pool of ~200k reviews without downloading 9 GB.

Run from the project root:  python scripts/build_sample.py
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

CATEGORY = "Cell_Phones_and_Accessories"
URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
       f"raw/review_categories/{CATEGORY}.jsonl")
SEED = 42
N_CHUNKS = 400            # number of random slices to read
CHUNK_BYTES = 256 * 1024  # size of each slice (~550 reviews)
PER_CLASS = 3000


def star_to_label(rating):
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def fetch_chunk(start):
    """Download one byte range and return the complete JSON records inside it."""
    headers = {"Range": f"bytes={start}-{start + CHUNK_BYTES - 1}"}
    raw = requests.get(URL, headers=headers, timeout=60).content
    lines = raw.split(b"\n")[1:-1]  # first and last lines are cut in half, drop them
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def to_row(r):
    image_urls = [img["medium_image_url"] for img in r.get("images", [])]
    return {
        "rating": r.get("rating"),
        "title": r.get("title"),
        "text": r.get("text"),
        "image_urls": "|".join(image_urls),
        "n_images": len(image_urls),
        "asin": r.get("asin"),
        "parent_asin": r.get("parent_asin"),
        "user_id": r.get("user_id"),
        "timestamp": r.get("timestamp"),
        "helpful_vote": r.get("helpful_vote"),
        "verified_purchase": r.get("verified_purchase"),
    }


def main():
    file_size = int(requests.head(URL, allow_redirects=True, timeout=60).headers["Content-Length"])
    rng = random.Random(SEED)
    starts = sorted(rng.randrange(0, file_size - CHUNK_BYTES) for _ in range(N_CHUNKS))

    with ThreadPoolExecutor(max_workers=8) as pool:
        chunks = list(pool.map(fetch_chunk, starts))

    pool_df = pd.DataFrame([to_row(r) for chunk in chunks for r in chunk])
    pool_df = pool_df.drop_duplicates(subset=["user_id", "asin", "timestamp"])  # overlapping slices
    pool_df["label"] = pool_df["rating"].apply(star_to_label)

    # Natural class balance of the random pool, kept for the Phase 1 audit.
    stats = {
        "category": CATEGORY,
        "source_url": URL,
        "file_bytes": file_size,
        "chunks_read": N_CHUNKS,
        "bytes_read": N_CHUNKS * CHUNK_BYTES,
        "pool_reviews": len(pool_df),
        "pool_star_counts": {str(k): int(v) for k, v in pool_df["rating"].value_counts().sort_index().items()},
        "pool_label_counts": {k: int(v) for k, v in pool_df["label"].value_counts().items()},
        "pool_pct_with_images": round(100 * (pool_df["n_images"] > 0).mean(), 2),
    }
    with open("data/pool_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    sample = pool_df.groupby("label").sample(n=PER_CLASS, random_state=SEED)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows
    sample.to_csv("data/sample.csv", index=False)

    print(json.dumps(stats, indent=2))
    print("sample rows:", len(sample))


if __name__ == "__main__":
    main()
