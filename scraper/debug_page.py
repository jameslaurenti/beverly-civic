#!/usr/bin/env python3
"""Prints the HTML structure around date/time/location fields on an event detail page."""

import requests
from bs4 import BeautifulSoup

URL = "https://www.beverlyma.gov/Calendar.aspx?EID=3727&month=4&year=2026&day=17&calType=0"
HEADERS = {"User-Agent": "beverly-civic-bot/1.0 (contact james.laurenti@gmail.com)"}

resp = requests.get(URL, headers=HEADERS, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")

print("=== DT/DD pairs ===")
for dt in soup.find_all("dt"):
    dd = dt.find_next_sibling("dd")
    print(f"  DT: {repr(dt.get_text(strip=True))}  =>  DD: {repr(dd.get_text(strip=True) if dd else None)}")

print("\n=== Tables ===")
for table in soup.find_all("table"):
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if cells:
            print("  ROW:", [c.get_text(strip=True) for c in cells])

print("\n=== Any tag containing 'April' ===")
for tag in soup.find_all(string=lambda t: t and "April" in t):
    print(f"  <{tag.parent.name} class={tag.parent.get('class')}> {repr(tag.strip())}")

print("\n=== Any tag containing '1:00' ===")
for tag in soup.find_all(string=lambda t: t and "1:00" in t):
    print(f"  <{tag.parent.name} class={tag.parent.get('class')}> {repr(tag.strip())}")
