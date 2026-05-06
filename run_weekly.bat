@echo off
cd /d C:\Laurenti-Claude\beverly-civic
call venv\Scripts\activate
python scraper/scrape_calendar.py
python indexer/index_calendar.py
python scraper/scrape_news.py
python indexer/index_news.py
python scraper/scrape_minutes.py
python indexer/index_minutes.py
