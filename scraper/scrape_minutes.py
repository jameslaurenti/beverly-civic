#!/usr/bin/env python3
"""
Scrapes Beverly MA meeting minutes from AgendaCenter.

Usage:
    python scraper/scrape_minutes.py                              # City Council, 2024-present
    python scraper/scrape_minutes.py --years 2023 2024 2025 2026
    python scraper/scrape_minutes.py --catid 16 --committee "Planning Board"
    python scraper/scrape_minutes.py --catid 17 --committee "Zoning Board of Appeals"

Common catIDs:
    49 = City Council (default)
    16 = Planning Board
    17 = Zoning Board of Appeals
    10 = Conservation Commission
    23 = Historic District Commission
    15 = Community Preservation Committee
"""

import argparse
import io
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.beverlyma.gov"
DATA_DIR = Path(__file__).parent.parent / "data"
HEADERS = {
    "User-Agent": "beverly-civic-bot/1.0 (civic data aggregator; contact james.laurenti@gmail.com)"
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_year(session: requests.Session, year: int, catid: int) -> BeautifulSoup:
    log.info("Fetching year %d (catID=%d)", year, catid)
    resp = session.post(f"{BASE_URL}/AgendaCenter/UpdateCategoryList", data={
        "year": year,
        "catID": catid,
        "startDate": "",
        "endDate": "",
        "term": "",
        "prevVersionScreen": "false",
    }, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _extract_meeting_id(href: str) -> str | None:
    m = re.search(r"_([0-9]+-[0-9]+)", href)
    return m.group(1) if m else None


def _parse_date(text: str) -> str:
    """Extract 'Month D, YYYY' from TD text like 'Dec 15, 2025<NBSP>Posted...'"""
    m = re.match(r"([A-Za-z]+ \d{1,2}, \d{4})", text.strip())
    return m.group(1) if m else text.split()[0]


def parse_meetings(soup: BeautifulSoup, committee: str) -> list[dict]:
    meetings = []
    for row in soup.select("tr.catAgendaRow"):
        minutes_td = row.find("td", class_="minutes")
        if not minutes_td:
            continue
        minutes_a = minutes_td.find("a", href=True)
        if not minutes_a:
            continue  # no minutes posted for this meeting

        minutes_href = minutes_a["href"]
        minutes_url = BASE_URL + minutes_href if minutes_href.startswith("/") else minutes_href
        meeting_id = _extract_meeting_id(minutes_href)
        if not meeting_id:
            continue

        first_td = row.find("td")
        row_text = first_td.get_text(" ", strip=True) if first_td else ""
        date = _parse_date(row_text)

        agenda_a = (first_td.find("a", href=lambda h: h and "ViewFile/Agenda" in h)
                    if first_td else None)
        agenda_url = (BASE_URL + agenda_a["href"] if agenda_a and agenda_a["href"].startswith("/")
                      else (agenda_a["href"] if agenda_a else None))

        title = ""
        if agenda_a:
            raw = agenda_a.get_text(strip=True)
            title = re.sub(r"\s*Agenda\s*\(PDF\)\s*$", "", raw).strip()

        meetings.append({
            "meeting_id": meeting_id,
            "committee": committee,
            "date": date,
            "title": title,
            "minutes_url": minutes_url,
            "agenda_url": agenda_url,
        })
        log.info("  Found minutes: %s - %s", date, title)

    return meetings


def _ocr_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    log.info("  OCR: %d pages", len(doc))
    texts = []
    for page in doc:
        png = page.get_pixmap(dpi=200).tobytes("png")
        text = pytesseract.image_to_string(Image.open(io.BytesIO(png)))
        if text.strip():
            texts.append(text.strip())
    return "\n\n".join(texts)


def extract_pdf_text(session: requests.Session, pdf_url: str) -> str:
    resp = session.get(pdf_url, timeout=60)
    resp.raise_for_status()
    pdf_bytes = resp.content

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages).strip()

    if text:
        return text

    log.info("  No embedded text - falling back to OCR")
    return _ocr_pdf(pdf_bytes)


def _committee_slug(committee: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", committee).strip("_").lower()


def save(meetings: list[dict], committee: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _committee_slug(committee)
    out_path = DATA_DIR / f"minutes_{slug}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meetings, f, indent=2, ensure_ascii=False)
    log.info("Saved %d meetings -> %s", len(meetings), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Scrape Beverly MA meeting minutes")
    current_year = datetime.now().year
    parser.add_argument("--years", type=int, nargs="+",
                        default=list(range(2024, current_year + 1)),
                        help="Years to scrape (default: 2024 to current year)")
    parser.add_argument("--catid", type=int, default=49,
                        help="AgendaCenter category ID (default: 49 = City Council)")
    parser.add_argument("--committee", type=str, default="City Council",
                        help="Human-readable committee name (default: City Council)")
    args = parser.parse_args()

    session = make_session()
    all_meetings = []

    for year in args.years:
        log.info("--- Year %d ---", year)
        try:
            soup = fetch_year(session, year, args.catid)
            meetings = parse_meetings(soup, args.committee)
            log.info("  %d meetings with minutes", len(meetings))
        except Exception as e:
            log.error("  Failed to fetch year %d: %s", year, e)
            continue

        for meeting in meetings:
            log.info("Extracting minutes: %s", meeting["date"])
            try:
                meeting["minutes_text"] = extract_pdf_text(session, meeting["minutes_url"])
                log.info("  %d chars extracted", len(meeting["minutes_text"]))
            except Exception as e:
                log.warning("  PDF extraction failed: %s", e)
                meeting["minutes_text"] = None
            all_meetings.append(meeting)

    if all_meetings:
        save(all_meetings, args.committee)
    else:
        log.warning("No meetings with minutes found")


if __name__ == "__main__":
    main()
