@echo off
setlocal

echo Installing required packages...
pip install -r requirements_simple.txt

echo Training the model...
python train_simple_model.py

echo Starting the API server...
python simple_api.py
