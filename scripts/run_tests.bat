@echo off
echo Running tests for Jigsaw ML project...

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Running tests with pytest...
python -m pytest tests/ -v

echo Test run complete!