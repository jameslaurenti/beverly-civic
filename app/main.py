#!/usr/bin/env python3
"""
Beverly Civic chat app.

RAG pipeline: embed question → query Pinecone → generate answer.
LLM is stubbed until Anthropic billing is set up; the UI shows retrieved
sources so you can validate the pipeline end-to-end.

Run:
    PINECONE_API_KEY=... uvicorn app.main:app --reload
or:
    PINECONE_API_KEY=... python app/main.py
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

STATIC_DIR = Path(__file__).parent / "static"

INDEX_NAME = "beverly-civic"
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"

# Loaded once at startup, reused across requests
_model: SentenceTransformer | None = None
_index: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _index
    _model = SentenceTransformer(EMBED_MODEL)
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    _index = Pinecone(api_key=api_key).Index(INDEX_NAME)
    yield


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    text: str


def retrieve(question: str) -> list[dict]:
    embedding = _model.encode([question])[0].tolist()
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
    # Placeholder — swap this function body for a Claude API call when ready
    if not sources:
        return "I couldn't find anything relevant in the Beverly civic data."
    lines = [f"- **{s['title']}** ({s['date']}) — [link]({s['url']})" for s in sources]
    return (
        "_(AI answer coming soon — Claude API not yet connected)_\n\n"
        "**Most relevant results for your question:**\n\n"
        + "\n".join(lines)
    )


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
