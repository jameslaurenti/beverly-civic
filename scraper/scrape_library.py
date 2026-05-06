#!/usr/bin/env python3
"""
Scrapes Beverly Public Library events from the embedded Assabet calendar.

Usage:
    python scraper/scrape_library.py                    # current month + 2 ahead
    python scraper/scrape_library.py --months-ahead 6
    python scraper/scrape_library.py --months-back 1 --months-ahead 3
"""

import argparse
import html
import json
import logging
from calendar import month_name
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://beverlypubliclibrary.assabetinteractive.com"
DATA_DIR = Path(__file__).parent.parent / "data"
HEADERS = {
    "User-Agent": "beverly-civic-bot/1.0 (civic data aggregator; contact james.laurenti@gmail.com)"
}

CANCELLED_STATUS = "http://schema.org/EventCancelled"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _month_slug(year: int, month: int) -> str:
    return f"{year}-{month_name[month].lower()}"


def fetch_month(session: requests.Session, year: int, month: int) -> BeautifulSoup:
    slug = _month_slug(year, month)
    url = f"{BASE_URL}/calendar/{slug}/"
    log.info("Fetching %s", url)
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_events(soup: BeautifulSoup) -> list[dict]:
    events = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        if data.get("@type") != "Event":
            continue

        status = data.get("eventStatus", "")
        if status == CANCELLED_STATUS:
            continue

        location_list = data.get("location") or []
        if isinstance(location_list, dict):
            location_list = [location_list]
        location_name = location_list[0].get("name", "") if location_list else ""

        raw_desc = data.get("description", "") or ""
        description = BeautifulSoup(raw_desc, "html.parser").get_text(separator=" ", strip=True)

        events.append({
            "title": html.unescape(data.get("name", "")).strip(),
            "start_date": data.get("startDate", ""),
            "end_date": data.get("endDate", ""),
            "door_time": data.get("doorTime", ""),
            "duration": data.get("duration", ""),
            "url": data.get("url", ""),
            "description": description,
            "location": html.unescape(location_name),
        })

    log.info("  %d events parsed", len(events))
    return events


def _month_range(months_back: int, months_ahead: int) -> list[tuple[int, int]]:
    now = datetime.now()
    year, month = now.year, now.month
    months = []
    for delta in range(-months_back, months_ahead + 1):
        m = month + delta
        y = year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        months.append((y, m))
    return months


def save(events: list[dict]) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"library_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    log.info("Saved %d events -> %s", len(events), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Scrape Beverly Public Library events")
    parser.add_argument("--months-back", type=int, default=0,
                        help="Months of history to include (default: 0)")
    parser.add_argument("--months-ahead", type=int, default=2,
                        help="Months ahead to include (default: 2)")
    args = parser.parse_args()

    session = make_session()
    all_events = []

    for year, month in _month_range(args.months_back, args.months_ahead):
        try:
            soup = fetch_month(session, year, month)
            events = parse_events(soup)
            all_events.extend(events)
        except Exception as e:
            log.error("Failed to fetch %s: %s", _month_slug(year, month), e)

    seen_urls: set[str] = set()
    deduped = []
    for ev in all_events:
        if ev["url"] and ev["url"] not in seen_urls:
            seen_urls.add(ev["url"])
            deduped.append(ev)
        elif not ev["url"]:
            deduped.append(ev)

    log.info("Total: %d unique events across %d months", len(deduped),
             args.months_back + args.months_ahead + 1)

    if deduped:
        save(deduped)
    else:
        log.warning("No library events found")


if __name__ == "__main__":
    main()
