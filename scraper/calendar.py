#!/usr/bin/env python3
"""
Scrapes upcoming events from the Beverly MA city calendar, including agenda PDFs.

Outputs a timestamped JSON file to /data each run.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from bev_utils import BASE_URL, extract_pdf_text, fetch_html, make_session

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CALENDAR_URL = (
    f"{BASE_URL}/calendar.aspx?CID=47,23,25,26,27,46&showPastEvents=false"
)
DATA_DIR = Path(__file__).parent.parent / "data"


def _find_agenda_pdf_url(session, detail_soup) -> str | None:
    """
    Beverly event detail pages link to an AgendaCenter landing page.
    That page then has the actual PDF download link.
    Pattern: /AgendaCenter/ViewFile/Agenda/_MMDDYYYY-NNNN
    """
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
    """Fetch an event detail page and extract structured fields."""
    result = {
        "date": None, "time": None, "location": None, "description": None,
        "agenda_pdf_url": None, "agenda_text": None
    }
    soup = fetch_html(session, detail_url)

    # CivicPlus renders event metadata in a two-column table
    for row in soup.select("table tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) == 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if "date" in label:
                result["date"] = value
            elif "time" in label:
                result["time"] = value
            elif "location" in label:
                result["location"] = value

    # Description lives in varying containers across CivicPlus versions
    for selector in (".field-items", ".description", "#CivicAlerts-TargetID", ".fr-view"):
        el = soup.select_one(selector)
        if el:
            result["description"] = el.get_text(separator=" ", strip=True)
            break

    # Agenda PDF (may not exist yet if event is more than ~14 days out)
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
