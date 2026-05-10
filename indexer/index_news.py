#!/usr/bin/env python3
"""
Indexes Beverly MA news articles into Pinecone with Claude-enriched text.

For each article, Claude extracts concrete implications (e.g. "City Hall closed
April 20", "trash pickup delayed one day") so colloquial resident queries
surface the right article even when the title is non-obvious.

Run:
    python indexer/index_news.py            # skip already-indexed articles
    python indexer/index_news.py --force    # re-enrich and re-embed everything
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
SNIPPET_MAX = 1500


def load_news(data_dir: Path) -> list[dict]:
    """Load all news JSON files, deduplicate by detail_url (newest file wins)."""
    by_url: dict[str, dict] = {}
    for path in sorted(data_dir.glob("news_*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            url = item.get("detail_url", "")
            if url:
                by_url[url] = item
    log.info("Loaded %d unique news articles from %s", len(by_url), data_dir)
    return list(by_url.values())


def _enrich(article: dict, claude: anthropic.Anthropic) -> str:
    """Ask Claude to extract Q&A pairs a resident would actually search for."""
    title = article.get("title", "")
    body = article.get("body") or article.get("summary") or ""
    date = article.get("posted_date", "")

    prompt = (
        f"Beverly MA news article titled \"{title}\" (posted {date}):\n\n{body}\n\n"
        "Write 4-8 Q&A pairs a Beverly resident might search for when affected by this news. "
        "Use colloquial phrasing (e.g. 'Is City Hall open?', 'Is trash delayed?', 'When does the program start?'). "
        "Include the answer inline. Format: 'Q: <question> A: <short answer>' — one pair per line, no bullets."
    )

    try:
        resp = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log.warning("  Enrichment failed for '%s': %s", title, e)
        return ""


def _make_doc_id(url: str) -> str:
    return "news_" + re.sub(r"[^a-zA-Z0-9_-]", "_", url)[:500]


def build_vectors(articles: list[dict], claude: anthropic.Anthropic) -> list[dict]:
    vectors = []
    for i, article in enumerate(articles):
        title = article.get("title", "").strip()
        body = (article.get("body") or article.get("summary") or "").strip()
        date = article.get("posted_date", "").strip()
        url = article.get("detail_url", "")

        if not url or not title:
            continue

        log.info("[%d/%d] Enriching: %s", i + 1, len(articles), title)
        implications = _enrich(article, claude)

        text_parts = [f"Beverly MA News: {title}"]
        if date:
            text_parts.append(f"Posted: {date}")
        if body:
            text_parts.append(body)
        if implications:
            text_parts.append(f"\nKey facts:\n{implications}")
        text = "\n".join(text_parts)

        full_content = f"{body}\n\nKey facts:\n{implications}" if implications else body
        vectors.append({
            "id": _make_doc_id(url),
            "text": text,
            "metadata": {
                "title": title,
                "url": url,
                "date": date,
                "type": "news",
                "snippet": full_content[:SNIPPET_MAX],
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
        log.info("Embedding articles %d-%d...", i + 1, i + len(batch))
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
    parser = argparse.ArgumentParser(description="Index Beverly MA news articles")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich and re-embed all articles, even if already indexed")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent / ".env")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not pinecone_key:
        raise SystemExit("PINECONE_API_KEY not set")
    if not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    articles = load_news(DATA_DIR)
    if not articles:
        raise SystemExit(f"No news_*.json files found in {DATA_DIR}")

    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

    if not args.force:
        candidate_ids = [_make_doc_id(a["detail_url"]) for a in articles if a.get("detail_url")]
        existing = _fetch_existing_ids(idx, candidate_ids)
        before = len(articles)
        articles = [a for a in articles if _make_doc_id(a.get("detail_url", "")) not in existing]
        log.info("%d new articles to process (%d already indexed, skipping)",
                 len(articles), before - len(articles))

    if not articles:
        log.info("Nothing new to index.")
        return

    claude = anthropic.Anthropic(api_key=anthropic_key)
    vectors = build_vectors(articles, claude)
    log.info("%d news articles ready to index", len(vectors))

    upsert_vectors(vectors, pc, idx)
    log.info("Done.")


if __name__ == "__main__":
    main()
