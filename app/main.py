#!/usr/bin/env python3
"""
Beverly Civic chat app.

RAG pipeline: embed question via Pinecone inference → query index → Claude answer.

Run:
    PINECONE_API_KEY=... ANTHROPIC_API_KEY=... python app/main.py
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pinecone import Pinecone
from pydantic import BaseModel

STATIC_DIR = Path(__file__).parent / "static"
INDEX_NAME = "beverly-civic"
EMBED_MODEL = "multilingual-e5-large"
TOP_K = 10         # results returned to Claude
RETRIEVAL_K = 15   # candidates fetched per query variant before merging
MIN_SCORE = 0.75
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

_pc: Pinecone | None = None
_index: Any = None
_claude: anthropic.Anthropic | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pc, _index, _claude
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    _pc = Pinecone(api_key=pinecone_key)
    _index = _pc.Index(INDEX_NAME)
    _claude = anthropic.Anthropic(api_key=anthropic_key)
    yield


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    text: str
    history: list[dict] = []


def _query_variants(question: str) -> list[str]:
    resp = _claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content":
            f"Given this question about Beverly MA civic information, generate 2 search query variants:\n"
            f"1. A direct rephrasing of the question\n"
            f"2. A broader conceptual query capturing the underlying civic topic "
            f"(e.g. holiday closures, service delays, budget line items, meeting agendas)\n\n"
            f"Output only the 2 variants, one per line, no numbering or labels:\n\n{question}"}],
    )
    variants = [q.strip() for q in resp.content[0].text.strip().splitlines() if q.strip()]
    return [question] + variants[:2]


class _EmbedQuotaError(Exception):
    pass


def _embed(text: str) -> list[float]:
    try:
        return _pc.inference.embed(
            model=EMBED_MODEL,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )[0]["values"]
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
            raise _EmbedQuotaError() from e
        raise


ORIGINAL_QUERY_BOOST = 0.015  # boosts original query results vs. LLM-generated variants
MIN_CALENDAR = 1              # always surface at least 1 calendar event if any exist above threshold


def retrieve(question: str, history: list[dict] = []) -> list[dict]:
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    base_query = f"{last_assistant}\n{question}".strip() if last_assistant else question

    queries = _query_variants(base_query)

    seen: dict[str, dict] = {}
    for qi, q in enumerate(queries):
        boost = ORIGINAL_QUERY_BOOST if qi == 0 else 0.0
        results = _index.query(vector=_embed(q), top_k=RETRIEVAL_K, include_metadata=True)
        for m in results.matches:
            if m.score < MIN_SCORE:
                continue
            effective = round(m.score + boost, 3)
            if m.id not in seen or effective > seen[m.id]["score"]:
                seen[m.id] = {
                    "title": m.metadata.get("title", ""),
                    "url": m.metadata.get("url", ""),
                    "date": m.metadata.get("date", ""),
                    "type": m.metadata.get("type", ""),
                    "snippet": m.metadata.get("snippet", ""),
                    "score": effective,
                }

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # Cap per unique title: prevents recurring calendar events from flooding results,
    # while allowing up to 3 chunks from the same meeting to surface together.
    MAX_PER_TITLE = 3
    title_counts: dict[str, int] = {}
    results = []
    for r in ranked:
        key = r["title"]
        if title_counts.get(key, 0) < MAX_PER_TITLE:
            title_counts[key] = title_counts.get(key, 0) + 1
            results.append(r)

    # Guarantee at least MIN_CALENDAR calendar results so specific agenda questions
    # aren't completely crowded out by meeting minutes chunks. Calendar events may rank
    # outside TOP_K — reserve the last slot(s) for them rather than letting sort push
    # them back out.
    top_window = results[:TOP_K]
    cal_count = sum(1 for r in top_window if r.get("type") == "calendar")
    if cal_count < MIN_CALENDAR:
        top_ids = {id(r) for r in top_window}
        calendar_extras = [r for r in results if r.get("type") == "calendar" and id(r) not in top_ids]
        needed = MIN_CALENDAR - cal_count
        if calendar_extras:
            # Replace the lowest-scoring non-calendar slots with the best calendar extras
            non_cal_slots = [r for r in top_window if r.get("type") != "calendar"]
            cal_slots = [r for r in top_window if r.get("type") == "calendar"]
            top_window = non_cal_slots[: TOP_K - len(cal_slots) - needed] + cal_slots + calendar_extras[:needed]
            top_window.sort(key=lambda x: x["score"], reverse=True)

    return top_window[:TOP_K]


def answer(question: str, sources: list[dict], history: list[dict] = []) -> str:
    if not sources:
        return "I couldn't find anything relevant in the Beverly civic data."

    today = datetime.now().strftime("%A, %B %d, %Y").replace(" 0", " ")

    context_parts = []
    for s in sources:
        entry = f"[{s['type'].upper()}] {s['title']} ({s['date']})\nURL: {s['url']}"
        if s.get("snippet"):
            entry += f"\nContent: {s['snippet']}"
        context_parts.append(entry)
    context = "\n\n".join(context_parts)

    system = (
        f"You are a helpful assistant for residents of Beverly, MA. Today's date is {today}. "
        "You have access to six types of Beverly civic data: (1) city calendar events and meeting agendas, "
        "(2) city news and announcements, (3) city operating budgets for FY2024, FY2025, and FY2026, "
        "(4) Beverly Public Schools budgets for FY2024, FY2025, and FY2026, "
        "(5) meeting minutes from 2024-2026 covering City Council votes and ordinances, Planning Board "
        "decisions and zoning approvals, and Zoning Board of Appeals variances, and "
        "(6) Beverly Public Library events including programs, story times, and author talks. "
        "Answer questions using only the civic data provided in the user's message. "
        "Be concise and specific. Use plain prose — never use markdown headers (# or ##). "
        "Use bold text for key terms if helpful. "
        "Translate civic or government jargon into plain English — "
        "for example, explain what a committee does, what a variance means, or what a line item covers. "
        "If the retrieved data doesn't answer the question, say what data type "
        "would have the answer (e.g. 'this would be in the city calendar') and suggest the user try rephrasing. "
        "Never tell the user you don't have a type of data if it's one of the six types listed above. "
        "Always include relevant links from the data."
    )

    messages = list(history) + [{
        "role": "user",
        "content": f"Civic data:\n{context}\n\nQuestion: {question}",
    }]

    resp = _claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return resp.content[0].text.strip()


@app.post("/ask")
async def ask(q: Question):
    try:
        sources = retrieve(q.text, q.history)
    except _EmbedQuotaError:
        return JSONResponse({
            "answer": (
                "The search service is temporarily unavailable — the monthly usage limit for "
                "the underlying search index has been reached. It resets on June 1. "
                "In the meantime, you can browse Beverly civic information directly at "
                "beverlyma.gov."
            ),
            "sources": [],
        })
    response = answer(q.text, sources, q.history)
    return JSONResponse({"answer": response, "sources": sources})


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
