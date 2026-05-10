#!/usr/bin/env python3
"""
Indexes Beverly MA city calendar events from JSON scrape files into Pinecone.

Each event becomes one vector. Events without agenda text are still indexed
so metadata-only queries (what meetings are this week?) work.

Run:
    python indexer/index_calendar.py            # skip already-indexed events
    python indexer/index_calendar.py --force    # re-embed everything
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96
BATCH_PAUSE = 20


def _clean(s: str | None) -> str:
    if not s:
        return ""
    return s.replace("\xa0", " ").replace(" ", " ").strip()


def _eid(detail_url: str) -> str | None:
    try:
        qs = parse_qs(urlparse(detail_url).query)
        return qs["EID"][0]
    except (KeyError, IndexError):
        return None


def load_events(data_dir: Path) -> list[dict]:
    """Load all calendar JSON files, deduplicate by EID, newest file wins."""
    by_eid: dict[str, dict] = {}
    for path in sorted(data_dir.glob("calendar_*.json")):
        events = json.loads(path.read_text(encoding="utf-8"))
        for ev in events:
            eid = _eid(ev.get("detail_url", ""))
            if eid:
                by_eid[eid] = ev  # later file overwrites earlier — newest scrape wins
    log.info("Loaded %d unique events from %s", len(by_eid), data_dir)
    return list(by_eid.values())


def _build_text(ev: dict) -> str:
    title = _clean(ev.get("title"))
    date = _clean(ev.get("date"))
    time_str = _clean(ev.get("time"))
    location = _clean(ev.get("location"))
    agenda = _clean(ev.get("agenda_text"))

    parts = [f"Beverly MA City Calendar Event\n\nTitle: {title}"]
    if date:
        parts.append(f"Date: {date}")
    if time_str:
        parts.append(f"Time: {time_str}")
    if location and location.lower() not in ("event location", ""):
        parts.append(f"Location: {location}")
    if agenda:
        parts.append(f"\nAgenda:\n{agenda}")

    return "\n".join(parts)


def build_vectors(events: list[dict]) -> list[dict]:
    vectors = []
    for ev in events:
        eid = _eid(ev.get("detail_url", ""))
        if not eid:
            continue
        text = _build_text(ev)
        if len(text) < 20:
            continue
        vectors.append({
            "id": f"calendar_eid_{eid}",
            "text": text,
            "metadata": {
                "title": _clean(ev.get("title")),
                "date": _clean(ev.get("date")),
                "time": _clean(ev.get("time")),
                "location": _clean(ev.get("location")),
                "url": ev.get("detail_url", ""),
                "agenda_pdf_url": ev.get("agenda_pdf_url") or "",
                "has_agenda": bool(_clean(ev.get("agenda_text"))),
                "type": "calendar",
                "snippet": text[:1500],
            },
        })
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
    parser = argparse.ArgumentParser(description="Index Beverly MA calendar events")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed and upsert all events, even if already indexed")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    events = load_events(DATA_DIR)
    vectors = build_vectors(events)
    log.info("%d events ready to index", len(vectors))

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

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
