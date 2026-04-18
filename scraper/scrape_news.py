#!/usr/bin/env python3
"""
Scrapes recent news/announcements from the Beverly MA city website.

Outputs a timestamped JSON file to /data each run.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.beverlyma.gov"
NEWS_URL = f"{BASE_URL}/civicalerts.aspx"
DATA_DIR = Path(__file__).parent.parent / "data"
HEADERS = {
    "User-Agent": "beverly-civic-bot/1.0 (civic data aggregator; contact james.laurenti@gmail.com)"
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _fetch_body(session: requests.Session, detail_url: str) -> str:
    soup = fetch_html(session, detail_url)
    el = soup.select_one(".article-content.fr-view, .article-content")
    return el.get_text(separator=" ", strip=True) if el else ""


def scrape_news(session) -> list[dict]:
    log.info("Fetching city news...")
    soup = fetch_html(session, NEWS_URL)

    seen = set()
    articles = []

    for item in soup.select(".carousel-item"):
        a = item.select_one("a.article-title-link")
        if not a:
            continue

        href = a["href"]
        if href in seen:
            continue
        seen.add(href)

        title = a.get_text(strip=True)
        detail_url = BASE_URL + href if href.startswith("/") else href

        summary_el = item.select_one(".article-preview")
        summary = summary_el.get_text(strip=True) if summary_el else None

        date_el = item.select_one(".fst-italic")
        posted_date = date_el.get_text(strip=True).removeprefix("Posted on").strip() if date_el else None

        category_el = item.select_one(".badge")
        category = category_el.get_text(strip=True) if category_el else None

        log.info("Article: %s", title)
        body = ""
        try:
            body = _fetch_body(session, detail_url)
        except Exception as e:
            log.warning("  Could not fetch detail: %s", e)

        articles.append({
            "title": title,
            "detail_url": detail_url,
            "category": category,
            "posted_date": posted_date,
            "summary": summary,
            "body": body,
        })

    return articles


def save(articles: list[dict]) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"news_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log.info("Saved %d articles -> %s", len(articles), out_path)
    return out_path


def main():
    session = make_session()
    articles = scrape_news(session)
    save(articles)


if __name__ == "__main__":
    main()
