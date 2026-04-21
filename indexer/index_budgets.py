#!/usr/bin/env python3
"""
Indexes Beverly MA fiscal budget PDFs into Pinecone, one vector per page.

Add a new budget PDF to data/budgets/ and re-run to index it.
Already-indexed pages are overwritten in place (idempotent by doc ID).

Run:
    PINECONE_API_KEY=... python indexer/index_budgets.py
"""

import logging
import os
import re
import time
from pathlib import Path

import pdfplumber
from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUDGETS_DIR = Path(__file__).parent.parent / "data" / "budgets"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
BATCH_SIZE = 96
MIN_PAGE_CHARS = 80  # skip pages that are mostly images/charts


def parse_year(filename: str) -> str:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else "unknown"


def extract_pages(pdf_path: Path) -> list[dict]:
    year = parse_year(pdf_path.name)
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_PAGE_CHARS:
                continue
            pages.append({
                "id": f"budget_{year}_p{i + 1:04d}",
                "text": f"Beverly MA FY{year} Budget — Page {i + 1}\n\n{text}",
                "metadata": {
                    "title": f"Beverly MA FY{year} Proposed Operating Budget — Page {i + 1}",
                    "url": "",
                    "date": year,
                    "type": "budget",
                    "snippet": text[:1500],
                },
            })
    log.info("%s: %d/%d pages have extractable text", pdf_path.name, len(pages), i + 1)
    return pages


BATCH_PAUSE = 20  # seconds between batches to stay under free-tier rate limit


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


def index_pages(pages: list[dict], pc: Pinecone, idx) -> None:
    texts = [p["text"] for p in pages]
    for i in range(0, len(pages), BATCH_SIZE):
        batch = pages[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]
        log.info("Embedding pages %d-%d...", i + 1, i + len(batch))
        embeddings = _embed_with_retry(pc, batch_texts)
        vectors = [
            {"id": p["id"], "values": emb["values"], "metadata": p["metadata"]}
            for p, emb in zip(batch, embeddings)
        ]
        idx.upsert(vectors=vectors)
        log.info("Upserted %d vectors", len(vectors))
        if i + BATCH_SIZE < len(pages):
            time.sleep(BATCH_PAUSE)


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    pdfs = sorted(BUDGETS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {BUDGETS_DIR}")

    pc = Pinecone(api_key=api_key)
    idx = pc.Index(INDEX_NAME)
    log.info("Connected to Pinecone index: %s", INDEX_NAME)

    for pdf_path in pdfs:
        log.info("Processing %s...", pdf_path.name)
        pages = extract_pages(pdf_path)
        if pages:
            index_pages(pages, pc, idx)

    log.info("All budgets indexed.")


if __name__ == "__main__":
    main()
