@echo off
echo Setting up environment for Jigsaw ML project...

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Setting up NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo Environment setup complete!
echo To activate the environment, run: .venv\Scripts\activate.bat
