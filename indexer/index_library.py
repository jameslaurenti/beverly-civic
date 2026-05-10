#!/usr/bin/env python3
"""
Indexes Beverly Public Library events into Pinecone.

Run:
    python indexer/index_library.py            # skip already-indexed events
    python indexer/index_library.py --force    # re-embed everything
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96
BATCH_PAUSE = 20
SNIPPET_MAX = 1500


def load_events(data_dir: Path) -> list[dict]:
    """Load all library JSON files, deduplicate by URL (newest file wins)."""
    by_url: dict[str, dict] = {}
    no_url_events: list[dict] = []
    for path in sorted(data_dir.glob("library_*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            url = item.get("url", "")
            if url:
                by_url[url] = item
            else:
                no_url_events.append(item)
    all_events = list(by_url.values()) + no_url_events
    log.info("Loaded %d unique library events", len(all_events))
    return all_events


def _make_doc_id(event: dict, index: int) -> str:
    url = event.get("url", "")
    if url:
        return "library_" + re.sub(r"[^a-zA-Z0-9_-]", "_", url)[:200]
    title = re.sub(r"[^a-zA-Z0-9_-]", "_", event.get("title", "unknown"))[:80]
    date = re.sub(r"[^0-9]", "", event.get("start_date", ""))[:12]
    return f"library_{title}_{date}_{index}"


def _format_datetime(date_str: str, time_str: str = "") -> str:
    """Format a date + optional time into a human-readable string."""
    from datetime import datetime as dt
    if not date_str:
        return ""
    try:
        if time_str:
            combined = f"{date_str}T{time_str}"
            d = dt.strptime(combined, "%Y-%m-%dT%H:%M:%S")
            hour = d.strftime("%I").lstrip("0") or "12"
            return d.strftime(f"%B {d.day}, %Y {hour}:%M %p")
        else:
            d = dt.strptime(date_str, "%Y-%m-%d")
            return d.strftime(f"%B {d.day}, %Y")
    except Exception:
        return date_str


def build_vectors(events: list[dict]) -> list[dict]:
    vectors = []
    for i, event in enumerate(events):
        title = event.get("title", "").strip()
        start_date = event.get("start_date", "")
        door_time = event.get("door_time", "")
        url = event.get("url", "")
        description = event.get("description", "").strip()
        location = event.get("location", "").strip()

        if not title:
            log.info("[%d/%d] Skipping event with no title", i + 1, len(events))
            continue

        display_date = _format_datetime(start_date, door_time) or start_date

        text_parts = [f"Beverly Public Library Event: {title}"]
        if display_date:
            text_parts.append(f"Date/Time: {display_date}")
        if location:
            text_parts.append(f"Location: {location}")
        if description:
            text_parts.append(f"Description: {description}")
        text = "\n".join(text_parts)

        snippet_parts = [title]
        if display_date:
            snippet_parts.append(display_date)
        if location:
            snippet_parts.append(location)
        if description:
            snippet_parts.append(description)
        snippet = " | ".join(snippet_parts)

        vectors.append({
            "id": _make_doc_id(event, i),
            "text": text,
            "metadata": {
                "title": title,
                "date": display_date or start_date,
                "url": url,
                "type": "library",
                "snippet": snippet[:SNIPPET_MAX],
            },
        })

        log.info("[%d/%d] Prepared: %s (%s)", i + 1, len(events), title, display_date)

    return vectors


def _fetch_existing_ids(idx, ids: list[str]) -> set[str]:
    existing = set()
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        result = idx.fetch(ids=batch)
        existing.update(result.vectors.keys())
    return existing


def _embed_with_retry(pc: Pinecone, texts: list[str]) -> list:
    for attempt in range(4):
        try:
            return pc.inference.embed(
                model=EMBED_MODEL,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
        except Exception as e:
            if "429" in str(e) and "RESOURCE_EXHAUSTED" not in str(e) and attempt < 3:
                wait = 60 * (attempt + 1)
                log.warning("Rate limited — waiting %ds before retry %d/3", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise


def upsert_vectors(vectors: list[dict], pc: Pinecone, idx) -> None:
    texts = [v["text"] for v in vectors]
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i: i + BATCH_SIZE]
        batch_texts = texts[i: i + BATCH_SIZE]
        log.info("Embedding events %d-%d...", i + 1, i + len(batch))
        embeddings = _embed_with_retry(pc, batch_texts)
        records = [
            {"id": v["id"], "values": emb["values"], "metadata": v["metadata"]}
            for v, emb in zip(batch, embeddings)
        ]
        idx.upsert(vectors=records)
        log.info("Upserted %d vectors", len(records))
        if i + BATCH_SIZE < len(vectors):
            time.sleep(BATCH_PAUSE)


def main():
    parser = argparse.ArgumentParser(description="Index Beverly Public Library events")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed all events, even if already indexed")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent / ".env")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise SystemExit("PINECONE_API_KEY not set")

    events = load_events(DATA_DIR)
    if not events:
        raise SystemExit(f"No library_*.json files found in {DATA_DIR}")

    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

    vectors = build_vectors(events)
    log.info("%d library events ready to index", len(vectors))

    if not args.force:
        all_ids = [v["id"] for v in vectors]
        existing = _fetch_existing_ids(idx, all_ids)
        before = len(vectors)
        vectors = [v for v in vectors if v["id"] not in existing]
        log.info("%d new events to index (%d already indexed, skipping)",
                 len(vectors), before - len(vectors))

    if not vectors:
        log.info("Nothing new to index.")
        return

    upsert_vectors(vectors, pc, idx)
    log.info("Done.")


if __name__ == "__main__":
    main()
