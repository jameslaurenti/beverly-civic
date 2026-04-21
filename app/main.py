#!/usr/bin/env python3
"""
Beverly Civic chat app.

RAG pipeline: embed question via Pinecone inference → query index → Claude answer.

Run:
    PINECONE_API_KEY=... ANTHROPIC_API_KEY=... python app/main.py
"""

import os
from contextlib import asynccontextmanager
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
TOP_K = 5
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


def retrieve(question: str) -> list[dict]:
    embedding = _pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[question],
        parameters={"input_type": "query", "truncate": "END"},
    )[0]["values"]
    results = _index.query(vector=embedding, top_k=TOP_K, include_metadata=True)
    return [
        {
            "title": m.metadata.get("title", ""),
            "url": m.metadata.get("url", ""),
            "date": m.metadata.get("date", ""),
            "type": m.metadata.get("type", ""),
            "score": round(m.score, 3),
        }
        for m in results.matches
    ]


def answer(question: str, sources: list[dict]) -> str:
    if not sources:
        return "I couldn't find anything relevant in the Beverly civic data."

    context = "\n\n".join(
        f"[{s['type'].upper()}] {s['title']} ({s['date']})\nURL: {s['url']}"
        for s in sources
    )

    prompt = f"""You are a helpful assistant for residents of Beverly, MA. Answer the question below using only the civic data provided. Be concise and specific. If the data doesn't fully answer the question, say so and share what you do know. Always include relevant links from the data.

Civic data:
{context}

Question: {question}"""

    resp = _claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


@app.post("/ask")
async def ask(q: Question):
    sources = retrieve(q.text)
    response = answer(q.text, sources)
    return JSONResponse({"answer": response, "sources": sources})


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
