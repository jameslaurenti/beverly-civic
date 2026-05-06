#!/usr/bin/env python3
"""
Generates a ground-truth eval set for the Beverly Civic Assistant.

Samples real chunks from calendar JSON, news JSON, and Pinecone (budgets),
then asks Claude to write realistic resident questions + expected answers.

Output: eval/eval_set.json

Run:
    python eval/generate_eval.py
"""

import json
import logging
import os
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = Path(__file__).parent / "eval_set.json"
EVAL_MODEL = "claude-haiku-4-5-20251001"

CALENDAR_SAMPLES = 10
NEWS_SAMPLES = 5
BUDGET_SEED_QUERIES = [
    "public works department budget spending",
    "school budget teachers salaries",
    "tax levy revenue",
    "police fire department budget",
    "capital improvements city spending",
]


def _ask_claude(client: anthropic.Anthropic, data_type: str, title: str, date: str, content: str) -> list[dict]:
    prompt = f"""Here is a piece of Beverly MA civic data:

TYPE: {data_type}
TITLE: {title}
DATE: {date}
CONTENT: {content[:1200]}

Generate 2 realistic questions that a Beverly MA resident might ask, based only on this content.
Questions should be natural and specific — like something typed into a chatbot.
Include a concise expected answer for each, drawn only from the content above.

Respond with a JSON array only, no commentary:
[
  {{"question": "...", "expected_answer": "..."}},
  {{"question": "...", "expected_answer": "..."}}
]"""

    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def sample_calendar(n: int) -> list[dict]:
    # Use the most recent full scrape
    paths = sorted(DATA_DIR.glob("calendar_*.json"))
    all_events = []
    seen = set()
    for p in reversed(paths):
        for ev in json.loads(p.read_text(encoding="utf-8")):
            eid = ev.get("detail_url", "")
            agenda = (ev.get("agenda_text") or "").strip()
            if eid not in seen and len(agenda) > 100:
                seen.add(eid)
                all_events.append(ev)
    random.seed(42)
    return random.sample(all_events, min(n, len(all_events)))


def sample_news() -> list[dict]:
    paths = sorted(DATA_DIR.glob("news_*.json"))
    if not paths:
        return []
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def sample_budget_chunks(pc: Pinecone, queries: list[str]) -> list[dict]:
    chunks = []
    seen = set()
    for q in queries:
        emb = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[q],
            parameters={"input_type": "query", "truncate": "END"},
        )[0]["values"]
        results = pc.Index("beverly-civic").query(
            vector=emb, top_k=3, include_metadata=True,
            filter={"type": {"$eq": "budget"}},
        )
        for m in results.matches:
            if m.id not in seen:
                seen.add(m.id)
                chunks.append(m.metadata)
    return chunks


def main():
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise SystemExit("PINECONE_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=anthropic_key)
    pc = Pinecone(api_key=pinecone_key)

    eval_set = []

    # Calendar events
    log.info("Sampling %d calendar events...", CALENDAR_SAMPLES)
    for ev in sample_calendar(CALENDAR_SAMPLES):
        try:
            pairs = _ask_claude(
                client,
                data_type="city calendar event with agenda",
                title=ev.get("title", ""),
                date=ev.get("date", ""),
                content=ev.get("agenda_text", ""),
            )
            for p in pairs:
                eval_set.append({
                    "question": p["question"],
                    "expected_answer": p["expected_answer"],
                    "source_title": ev.get("title", ""),
                    "source_date": ev.get("date", ""),
                    "data_type": "calendar",
                })
            log.info("  ✓ %s", ev.get("title", ""))
        except Exception as e:
            log.warning("  Skipped %s: %s", ev.get("title", ""), e)

    # News items
    log.info("Sampling news items...")
    for item in sample_news()[:NEWS_SAMPLES]:
        content = item.get("body") or item.get("summary") or ""
        if len(content) < 50:
            continue
        try:
            pairs = _ask_claude(
                client,
                data_type="city news announcement",
                title=item.get("title", ""),
                date=item.get("posted_date", ""),
                content=content,
            )
            for p in pairs:
                eval_set.append({
                    "question": p["question"],
                    "expected_answer": p["expected_answer"],
                    "source_title": item.get("title", ""),
                    "source_date": item.get("posted_date", ""),
                    "data_type": "news",
                })
            log.info("  ✓ %s", item.get("title", ""))
        except Exception as e:
            log.warning("  Skipped %s: %s", item.get("title", ""), e)

    # Budget chunks
    log.info("Sampling budget chunks from Pinecone...")
    for chunk in sample_budget_chunks(pc, BUDGET_SEED_QUERIES):
        content = chunk.get("snippet", "")
        if len(content) < 100:
            continue
        try:
            pairs = _ask_claude(
                client,
                data_type="city budget document",
                title=chunk.get("title", ""),
                date=chunk.get("date", ""),
                content=content,
            )
            # Take only 1 question per budget chunk to avoid over-indexing on budget
            eval_set.append({
                "question": pairs[0]["question"],
                "expected_answer": pairs[0]["expected_answer"],
                "source_title": chunk.get("title", ""),
                "source_date": chunk.get("date", ""),
                "data_type": "budget",
            })
            log.info("  ✓ %s", chunk.get("title", ""))
        except Exception as e:
            log.warning("  Skipped budget chunk: %s", e)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(eval_set, indent=2), encoding="utf-8")
    log.info("Saved %d eval questions -> %s", len(eval_set), OUT_PATH)


if __name__ == "__main__":
    main()
