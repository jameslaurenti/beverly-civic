# Beverly Civic Assistant

An AI-powered tool that lets Beverly, MA residents chat with their city's public record.

Beverly's civic information is scattered across an outdated city website — meeting agendas buried
in PDFs, budget data that requires knowing where to look, news that's hard to search. This tool
scrapes, indexes, and surfaces that information through a conversational interface.

**Live demo:** [beverly-civic.onrender.com](https://beverly-civic.onrender.com) *(first load may
take ~30s due to Render cold start)*

## What it knows

- City calendar events and meeting agendas
- City news and announcements
- Beverly city operating budgets (FY2024, FY2025)
- Beverly Public Schools budgets (FY2024, FY2025, FY2026)

## Architecture

```
scraper/          # Pulls calendar + news from beverlycityma.iod.com
indexer/          # Parses budget PDFs; embeds all content into Pinecone
data/             # Local JSON/text data between scrape and index
app/              # FastAPI chat server + static frontend
```

**RAG pipeline:** question → Pinecone embedding (multilingual-e5-large) → top-12 retrieval →
Claude Haiku generates answer with source links. Conversation history is carried forward so
follow-up questions resolve correctly.

## Stack

| Layer | Technology |
|-------|------------|
| Scraping | Python, requests, BeautifulSoup |
| PDF parsing | pdfplumber, PyMuPDF |
| Vector store | Pinecone (multilingual-e5-large embeddings) |
| API | FastAPI + uvicorn |
| LLM | Claude Haiku (Anthropic) |

## Running locally

```bash
pip install -r requirements.txt

export PINECONE_API_KEY=...
export ANTHROPIC_API_KEY=...

python app/main.py
```

Open [http://localhost:8000](http://localhost:8000).

To rebuild the index from scratch, run the scrapers first, then the indexers:

```bash
python scraper/scrape_calendar.py
python scraper/scrape_news.py
python indexer/index_content.py
python indexer/index_budgets.py
```

## Built with

Developed with [Claude Code](https://claude.ai/code).
