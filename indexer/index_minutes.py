#!/usr/bin/env python3
"""
Indexes Beverly MA City Council meeting minutes into Pinecone.

Uses Claude to extract key decisions, votes, and outcomes so residents
can ask "what happened at last week's council meeting?"

Run:
    python indexer/index_minutes.py
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 96
BATCH_PAUSE = 20
SNIPPET_MAX = 1500


def load_minutes(data_dir: Path) -> list[dict]:
    """Load all minutes JSON files, deduplicate by (committee, meeting_id) (newest file wins)."""
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


def _summarize(meeting: dict, claude: anthropic.Anthropic) -> str:
    """Ask Claude to extract decisions, votes, and outcomes from raw minutes text."""
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

        if not meeting_id or not minutes_text:
            log.info("[%d/%d] Skipping %s — no minutes text", i + 1, len(meetings), date)
            continue

        committee = meeting.get("committee", "City Council")
        log.info("[%d/%d] Summarizing: %s %s", i + 1, len(meetings), committee, date)
        summary = _summarize(meeting, claude)

        text_parts = [f"Beverly MA {committee} Meeting Minutes", f"Date: {date}"]
        if minutes_text:
            text_parts.append(f"\nMinutes:\n{minutes_text[:3000]}")
        if summary:
            text_parts.append(f"\nKey decisions and outcomes:\n{summary}")
        text = "\n".join(text_parts)

        full_content = minutes_text
        if summary:
            full_content = f"{minutes_text}\n\nKey decisions:\n{summary}"

        meeting_title = meeting.get("title") or f"{committee} Meeting"
        # City Council uses legacy ID format for backward compat; other committees include catid prefix
        committee_slug = re.sub(r"[^a-zA-Z0-9]", "_", committee).lower()
        vec_id = (f"minutes_{meeting_id}" if committee == "City Council"
                  else f"minutes_{committee_slug}_{meeting_id}")
        vectors.append({
            "id": vec_id,
            "text": text,
            "metadata": {
                "title": f"{meeting_title} Minutes — {date}",
                "date": date,
                "url": minutes_url,
                "type": "minutes",
                "snippet": full_content[:SNIPPET_MAX],
            },
        })

    return vectors


def _embed_with_retry(pc: Pinecone, texts: list[str]) -> list:
    for attempt in range(4):
        try:
            return pc.inference.embed(
                model=EMBED_MODEL,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )
        except PineconeApiException as e:
            if e.status == 429 and attempt < 3:
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
        log.info("Embedding minutes %d-%d...", i + 1, i + len(batch))
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

    vectors = build_vectors(meetings, claude)
    log.info("%d minutes records ready to index", len(vectors))

    upsert_vectors(vectors, pc, idx)
    log.info("Done.")


if __name__ == "__main__":
    main()
