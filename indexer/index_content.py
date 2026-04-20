#!/usr/bin/env python3
"""
Reads scraped JSON files from /data, embeds each item via Pinecone inference,
and upserts to the beverly-civic index.

Run after scraping:
    PINECONE_API_KEY=... python indexer/index_content.py
"""

import json
import logging
import os
import re
from pathlib import Path

from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96  # Pinecone inference limit per request


def load_records() -> list[dict]:
    records = {}
    for path in sorted(DATA_DIR.glob("*.json"), reverse=True):
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            key = item.get("detail_url") or item.get("title", "")
            if key and key not in records:
                records[key] = item
    log.info("Loaded %d unique records from %s", len(records), DATA_DIR)
    return list(records.values())


def make_doc_id(item: dict) -> str:
    url = item.get("detail_url", item.get("title", ""))
    return re.sub(r"[^a-zA-Z0-9_-]", "_", url)[:512]


def make_text(item: dict) -> str:
    parts = [item.get("title", "")]
    if item.get("date"):
        parts.append(f"Date: {item['date']}")
    if item.get("time"):
        parts.append(f"Time: {item['time']}")
    if item.get("location"):
        parts.append(f"Location: {item['location']}")
    if item.get("posted_date"):
        parts.append(f"Posted: {item['posted_date']}")
    if item.get("category"):
        parts.append(f"Category: {item['category']}")
    parts.append(item.get("description") or item.get("summary") or "")
    parts.append(item.get("body") or item.get("agenda_text") or "")
    return " ".join(p for p in parts if p).strip()


def make_metadata(item: dict) -> dict:
    return {
        "title": item.get("title", ""),
        "url": item.get("detail_url", ""),
        "date": item.get("date") or item.get("posted_date") or "",
        "type": "news" if item.get("posted_date") else "event",
    }


def index(records: list[dict], pc: Pinecone, idx) -> None:
    texts = [make_text(r) for r in records]

    for i in range(0, len(records), BATCH_SIZE):
        batch_records = records[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]

        log.info("Embedding batch %d-%d...", i + 1, i + len(batch_records))
        embeddings = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=batch_texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        vectors = [
            {
                "id": make_doc_id(r),
                "values": emb["values"],
                "metadata": make_metadata(r),
            }
            for r, emb in zip(batch_records, embeddings)
        ]
        idx.upsert(vectors=vectors)
        log.info("Upserted %d vectors", len(vectors))

    log.info("Done. %d total vectors indexed.", len(records))


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    records = load_records()
    if not records:
        raise SystemExit(f"No JSON files found in {DATA_DIR}")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

    index(records, pc, idx)


if __name__ == "__main__":
    main()
