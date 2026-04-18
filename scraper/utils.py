import io
import logging

import pdfplumber
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE_URL = "https://www.beverlyma.gov"
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
    return BeautifulSoup(resp.text, "lxml")


def extract_pdf_text(session: requests.Session, pdf_url: str) -> str:
    """Download a PDF and return its full text."""
    resp = session.get(pdf_url, timeout=30)
    resp.raise_for_status()
    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        return "\n\n".join(
            page.extract_text() or "" for page in pdf.pages
        ).strip()
