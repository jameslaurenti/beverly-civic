@echo off
cd /d C:\Laurenti-Claude\beverly-civic
call venv\Scripts\activate

set LOGFILE=logs\run_weekly_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%.log
if not exist logs mkdir logs

echo ===== Beverly Civic Weekly Run %DATE% %TIME% ===== >> %LOGFILE%

echo [1/8] Calendar scraper >> %LOGFILE%
python scraper/scrape_calendar.py --months-ahead 3 >> %LOGFILE% 2>&1
python indexer/index_calendar.py >> %LOGFILE% 2>&1

echo [2/8] News scraper >> %LOGFILE%
python scraper/scrape_news.py >> %LOGFILE% 2>&1
python indexer/index_news.py >> %LOGFILE% 2>&1

echo [3/8] City Council minutes >> %LOGFILE%
python scraper/scrape_minutes.py --catid 49 --committee "City Council" >> %LOGFILE% 2>&1

echo [4/8] Planning Board minutes >> %LOGFILE%
python scraper/scrape_minutes.py --catid 16 --committee "Planning Board" >> %LOGFILE% 2>&1

echo [5/8] Zoning Board of Appeals minutes >> %LOGFILE%
python scraper/scrape_minutes.py --catid 17 --committee "Zoning Board of Appeals" >> %LOGFILE% 2>&1

echo [6/8] Indexing all minutes >> %LOGFILE%
python indexer/index_minutes.py >> %LOGFILE% 2>&1

echo [7/8] Library events >> %LOGFILE%
python scraper/scrape_library.py --months-back 1 --months-ahead 2 >> %LOGFILE% 2>&1
python indexer/index_library.py >> %LOGFILE% 2>&1

echo [8/8] Done >> %LOGFILE%
echo ===== Completed %DATE% %TIME% ===== >> %LOGFILE%

echo Run complete. Log: %LOGFILE%
