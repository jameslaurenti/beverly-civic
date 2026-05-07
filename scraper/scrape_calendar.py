#!/usr/bin/env python3
"""
Scrapes Beverly MA city calendar events across any date range, including agenda PDFs.

Usage:
    python scraper/scrape_calendar.py                     # current + 2 months ahead
    python scraper/scrape_calendar.py --months-back 6     # also pull 6 months of history
    python scraper/scrape_calendar.py --months-ahead 0    # current month only
"""

import argparse
import io
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.beverlyma.gov"
CALENDAR_BASE = f"{BASE_URL}/calendar.aspx"
CALENDAR_IDS = ["47", "23", "25", "26", "27", "46"]
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


def _month_range(months_back: int, months_ahead: int):
    today = date.today()
    base = today.month - 1  # 0-indexed
    for delta in range(-months_back, months_ahead + 1):
        idx = base + delta
        month = idx % 12 + 1
        year = today.year + idx // 12
        yield month, year



def fetch_month_soup(session: requests.Session, month: int, year: int) -> BeautifulSoup:
    """Fetch the calendar HTML for a specific month/year."""
    url = f"{CALENDAR_BASE}?month={month}&year={year}&CID={','.join(CALENDAR_IDS)}"
    return fetch_html(session, url)


def _ocr_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    log.info("  OCR: running Tesseract on %d pages", len(doc))
    texts = []
    for page in doc:
        png = page.get_pixmap(dpi=200).tobytes("png")
        text = pytesseract.image_to_string(Image.open(io.BytesIO(png)))
        if text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


def extract_pdf_text(session: requests.Session, pdf_url: str) -> str:
    resp = session.get(pdf_url, timeout=30)
    resp.raise_for_status()
    pdf_bytes = resp.content

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()

    if text:
        return text

    log.info("  PDF has no embedded text — falling back to Tesseract OCR")
    return _ocr_pdf(pdf_bytes)


def _find_agenda_pdf_url(session, detail_soup) -> str | None:
    for a in detail_soup.find_all("a", href=True):
        href = a["href"]
        if "AgendaCenter" in href and "ViewFile" not in href:
            agenda_page_url = BASE_URL + href if href.startswith("/") else href
            try:
                agenda_soup = fetch_html(session, agenda_page_url)
                for b in agenda_soup.find_all("a", href=True):
                    if "ViewFile/Agenda" in b["href"]:
                        pdf_path = b["href"]
                        return BASE_URL + pdf_path if pdf_path.startswith("/") else pdf_path
            except Exception as e:
                log.warning("  Could not load agenda page: %s", e)
    return None


def _parse_detail(session, detail_url: str) -> dict:
    result = {
        "date": None, "time": None, "location": None, "description": None,
        "agenda_pdf_url": None, "agenda_text": None,
    }
    soup = fetch_html(session, detail_url)

    for header in soup.select(".specificDetailHeader"):
        label = header.get_text(strip=True).lower()
        item = header.find_next_sibling(class_="specificDetailItem") or header.find_next(class_="specificDetailItem")
        value = item.get_text(strip=True) if item else None
        if not value:
            continue
        if "date" in label:
            result["date"] = value
        elif "time" in label:
            result["time"] = value
        elif "location" in label:
            result["location"] = value

    for selector in (".field-items", ".description", "#CivicAlerts-TargetID", ".fr-view"):
        el = soup.select_one(selector)
        if el:
            result["description"] = el.get_text(separator=" ", strip=True)
            break

    pdf_url = _find_agenda_pdf_url(session, soup)
    if pdf_url:
        result["agenda_pdf_url"] = pdf_url
        log.info("  Extracting agenda PDF: %s", pdf_url)
        try:
            result["agenda_text"] = extract_pdf_text(session, pdf_url)
        except Exception as e:
            log.warning("  PDF extraction failed: %s", e)

    return result


def scrape_events(session: requests.Session, soup: BeautifulSoup) -> list[dict]:
    seen = set()
    events = []

    for a in soup.select("a[href*='Calendar.aspx?EID=']"):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)

        title = a.get_text(strip=True)
        if not title:
            continue

        detail_url = BASE_URL + href if href.startswith("/") else href
        log.info("Event: %s", title)

        event = {"title": title, "detail_url": detail_url}
        try:
            event.update(_parse_detail(session, detail_url))
        except Exception as e:
            log.warning("  Skipping detail page: %s", e)

        events.append(event)

    return events


def save(events: list[dict]) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"calendar_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    log.info("Saved %d events -> %s", len(events), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Scrape Beverly MA city calendar")
    parser.add_argument("--months-back", type=int, default=0,
                        help="Months of history to backfill (default: 0)")
    parser.add_argument("--months-ahead", type=int, default=2,
                        help="Months ahead to scrape (default: 2)")
    args = parser.parse_args()

    session = make_session()
    all_events = []

    for month, year in _month_range(args.months_back, args.months_ahead):
        log.info("--- Fetching %d/%d ---", month, year)
        try:
            soup = fetch_month_soup(session, month, year)
            events = scrape_events(session, soup)
            log.info("  %d events found", len(events))
            all_events.extend(events)
        except Exception as e:
            log.error("  Failed to scrape %d/%d: %s", month, year, e)

    save(all_events)


if __name__ == "__main__":
    main()
