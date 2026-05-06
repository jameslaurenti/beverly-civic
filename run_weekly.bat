@echo off
cd /d C:\Laurenti-Claude\beverly-civic
call venv\Scripts\activate
python scraper/scrape_calendar.py
python indexer/index_calendar.py
