#!/usr/bin/env python3
"""
Indexes Beverly MA meeting minutes into Pinecone using per-chunk vectors.

Each meeting produces:
  _c0  — Claude summary of key decisions (overview chunk)
  _c1, _c2, ...  — overlapping fixed-size windows of the raw OCR text

Run:
    python indexer/index_minutes.py                     # skip already-indexed meetings
    python indexer/index_minutes.py --force             # re-embed everything
    python indexer/index_minutes.py --clean             # delete old single-meeting vectors first
    python indexer/index_minutes.py --clean --force     # clean + re-embed all
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 96
BATCH_PAUSE = 20

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_minutes(data_dir: Path) -> list[dict]:
    """Load all minutes JSON files, deduplicate by (committee, meeting_id) — newest file wins."""
    by_id: dict[str, dict] = {}
    for path in sorted(data_dir.glob("minutes_*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            mid = item.get("meeting_id", "")
            committee = item.get("committee", "City Council")
            if mid:
                by_id[f"{committee}|{mid}"] = item
    log.info("Loaded %d unique minutes records", len(by_id))
    return list(by_id.values())


def _base_id(meeting: dict) -> str:
    mid = meeting["meeting_id"]
    committee = meeting.get("committee", "City Council")
    if committee == "City Council":
        return f"minutes_{mid}"
    slug = re.sub(r"[^a-zA-Z0-9]", "_", committee).lower()
    return f"minutes_{slug}_{mid}"


def _old_format_ids(meetings: list[dict]) -> list[str]:
    """Return the pre-chunking vector IDs (no _cN suffix) so they can be deleted."""
    return [_base_id(m) for m in meetings if m.get("meeting_id")]


def delete_old_vectors(meetings: list[dict], idx) -> None:
    ids = _old_format_ids(meetings)
    if not ids:
        return
    log.info("Deleting %d old single-meeting vectors...", len(ids))
    for i in range(0, len(ids), 1000):
        batch = ids[i: i + 1000]
        idx.delete(ids=batch)
    log.info("Deleted old vectors.")


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _summarize(meeting: dict, claude: anthropic.Anthropic) -> str:
    minutes_text = (meeting.get("minutes_text") or "").strip()
    if not minutes_text:
        return ""

    committee = meeting.get("committee", "City Council")
    prompt = (
        f"Beverly MA {committee} meeting minutes from {meeting.get('date', 'unknown date')}:\n\n"
        f"{minutes_text[:4000]}\n\n"
        "Extract the key decisions, votes, and outcomes that Beverly residents would want to know. "
        "Focus on: ordinances passed or rejected, budget approvals, zoning changes, appointments, "
        "major discussions, items tabled. Write each as a plain-English sentence. "
        "Output one item per line, no bullets or numbering."
    )

    try:
        resp = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log.warning("  Summarization failed for %s: %s", meeting.get("date"), e)
        return ""


def build_vectors(meetings: list[dict], claude: anthropic.Anthropic) -> list[dict]:
    vectors = []

    for i, meeting in enumerate(meetings):
        meeting_id = meeting.get("meeting_id", "")
        date = meeting.get("date", "").strip()
        minutes_text = (meeting.get("minutes_text") or "").strip()
        minutes_url = meeting.get("minutes_url", "")
        committee = meeting.get("committee", "City Council")
        meeting_title = meeting.get("title") or f"{committee} Meeting"
        display_title = f"{meeting_title} Minutes — {date}"
        base = _base_id(meeting)

        if not meeting_id or not minutes_text:
            log.info("[%d/%d] Skipping %s %s — no text", i + 1, len(meetings), committee, date)
            continue

        log.info("[%d/%d] Chunking: %s %s", i + 1, len(meetings), committee, date)

        # Chunk 0: Claude summary (overview — answers "what happened at this meeting?")
        summary = _summarize(meeting, claude)
        if summary:
            header = f"Beverly MA {committee} Meeting Minutes — {date}\nKey decisions and outcomes:\n"
            vectors.append({
                "id": f"{base}_c0",
                "text": header + summary,
                "metadata": {
                    "title": display_title,
                    "date": date,
                    "url": minutes_url,
                    "type": "minutes",
                    "snippet": f"Summary of {committee} meeting on {date}:\n{summary}",
                    "chunk": 0,
                },
            })

        # Chunks 1..N: overlapping windows of the raw OCR text
        chunks = chunk_text(minutes_text)
        log.info("  %d chars -> %d chunks", len(minutes_text), len(chunks))
        for n, chunk in enumerate(chunks, start=1):
            header = f"Beverly MA {committee} Meeting Minutes — {date}\n\n"
            vectors.append({
                "id": f"{base}_c{n}",
                "text": header + chunk,
                "metadata": {
                    "title": display_title,
                    "date": date,
                    "url": minutes_url,
                    "type": "minutes",
                    "snippet": chunk,
                    "chunk": n,
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
        log.info("Embedding chunks %d-%d...", i + 1, i + len(batch))
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
    parser = argparse.ArgumentParser(description="Index Beverly MA meeting minutes (chunked)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete old single-meeting vectors before upserting chunks")
    parser.add_argument("--force", action="store_true",
                        help="Re-summarize and re-embed all meetings, even if already indexed")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent / ".env")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not pinecone_key:
        raise SystemExit("PINECONE_API_KEY not set")
    if not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    meetings = load_minutes(DATA_DIR)
    if not meetings:
        raise SystemExit(f"No minutes_*.json files found in {DATA_DIR}")

    claude = anthropic.Anthropic(api_key=anthropic_key)
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

    if args.clean:
        delete_old_vectors(meetings, idx)

    if not args.force:
        # Use _c0 chunk as proxy — if summary chunk exists, full meeting is indexed
        candidate_ids = [
            f"{_base_id(m)}_c0"
            for m in meetings
            if m.get("meeting_id") and (m.get("minutes_text") or "").strip()
        ]
        existing = _fetch_existing_ids(idx, candidate_ids)
        before = len(meetings)
        meetings = [
            m for m in meetings
            if f"{_base_id(m)}_c0" not in existing
        ]
        log.info("%d new meetings to process (%d already indexed, skipping)",
                 len(meetings), before - len(meetings))

    if not meetings:
        log.info("Nothing new to index.")
        return

    vectors = build_vectors(meetings, claude)
    log.info("%d total chunks ready to index", len(vectors))

    upsert_vectors(vectors, pc, idx)
    log.info("Done.")


if __name__ == "__main__":
    main()
