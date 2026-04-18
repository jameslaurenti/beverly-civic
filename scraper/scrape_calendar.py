#!/usr/bin/env python3
"""
Scrapes upcoming events from the Beverly MA city calendar, including agenda PDFs.

Outputs a timestamped JSON file to /data each run.
"""

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf
import pdfplumber
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.beverlyma.gov"
CALENDAR_URL = f"{BASE_URL}/calendar.aspx?CID=47,23,25,26,27,46&showPastEvents=false"
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


OCR_MAX_PAGES = 5


def _ocr_pdf_with_claude(pdf_bytes: bytes) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("  ANTHROPIC_API_KEY not set — skipping OCR")
        return ""

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_to_ocr = min(len(doc), OCR_MAX_PAGES)
    log.info("  OCR: sending %d/%d pages to Claude vision", pages_to_ocr, len(doc))

    content = []
    for i in range(pages_to_ocr):
        page = doc[i]
        png_bytes = page.get_pixmap(dpi=150).tobytes("png")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png_bytes).decode(),
            },
        })

    content.append({
        "type": "text",
        "text": (
            "These are pages from a Beverly MA government meeting agenda. "
            "Transcribe all text exactly as it appears. "
            "Preserve structure (headings, numbered items, bullet points). "
            "Output only the transcribed text, no commentary."
        ),
    })

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


def extract_pdf_text(session: requests.Session, pdf_url: str) -> str:
    resp = session.get(pdf_url, timeout=30)
    resp.raise_for_status()
    pdf_bytes = resp.content

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()

    if text:
        return text

    # OCR fallback disabled for v1 (scanned PDFs return empty — deferred until Anthropic billing set up)
    log.info("  PDF has no embedded text — OCR not enabled, skipping")
    return ""


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
        "agenda_pdf_url": None, "agenda_text": None
    }
    soup = fetch_html(session, detail_url)

    headers = soup.select(".specificDetailHeader")
    for header in headers:
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


def scrape_events(session) -> list[dict]:
    log.info("Fetching city calendar...")
    soup = fetch_html(session, CALENDAR_URL)

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
    session = make_session()
    events = scrape_events(session)
    save(events)


if __name__ == "__main__":
    main()
