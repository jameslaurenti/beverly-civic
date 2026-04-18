#!/usr/bin/env python3
"""Fetches one event detail page and saves the raw HTML so we can inspect the structure."""

import requests
from pathlib import Path

URL = "https://www.beverlyma.gov/Calendar.aspx?EID=3727&month=4&year=2026&day=17&calType=0"
HEADERS = {"User-Agent": "beverly-civic-bot/1.0 (contact james.laurenti@gmail.com)"}

resp = requests.get(URL, headers=HEADERS, timeout=15)
out = Path(__file__).parent.parent / "data" / "debug_event.html"
out.write_text(resp.text, encoding="utf-8")
print(f"Saved to {out}")
