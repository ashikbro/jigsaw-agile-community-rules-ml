@echo off
call .\.venv_new\Scripts\activate
python -m uvicorn simple_api:app --host 127.0.0.1 --port 8000 --reload
pause
