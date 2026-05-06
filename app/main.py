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
TOP_K = 8          # results returned to Claude
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


def _embed(text: str) -> list[float]:
    return _pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"},
    )[0]["values"]


def retrieve(question: str, history: list[dict] = []) -> list[dict]:
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    base_query = f"{last_assistant}\n{question}".strip() if last_assistant else question

    queries = _query_variants(base_query)

    seen: dict[str, dict] = {}
    for q in queries:
        results = _index.query(vector=_embed(q), top_k=RETRIEVAL_K, include_metadata=True)
        for m in results.matches:
            if m.score < MIN_SCORE:
                continue
            if m.id not in seen or m.score > seen[m.id]["score"]:
                seen[m.id] = {
                    "title": m.metadata.get("title", ""),
                    "url": m.metadata.get("url", ""),
                    "date": m.metadata.get("date", ""),
                    "type": m.metadata.get("type", ""),
                    "snippet": m.metadata.get("snippet", ""),
                    "score": round(m.score, 3),
                }

    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:TOP_K]


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
        "You have access to five types of Beverly civic data: (1) city calendar events and meeting agendas, "
        "(2) city news and announcements, (3) city operating budgets for FY2024, FY2025, and FY2026, "
        "(4) Beverly Public Schools budgets for FY2024, FY2025, and FY2026, and "
        "(5) City Council meeting minutes from 2025-2026 covering votes, ordinances, and decisions. "
        "Answer questions using only the civic data provided in the user's message. "
        "Be concise and specific. Translate civic or government jargon into plain English — "
        "for example, explain what a committee does, what a warrant article means, or what a line item covers. "
        "If the retrieved data doesn't answer the question, say what data type "
        "would have the answer (e.g. 'this would be in the city calendar') and suggest the user try rephrasing. "
        "Never tell the user you don't have a type of data if it's one of the five types listed above. "
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
    sources = retrieve(q.text, q.history)
    response = answer(q.text, sources, q.history)
    return JSONResponse({"answer": response, "sources": sources})


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
