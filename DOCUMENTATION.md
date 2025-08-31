# Jigsaw Agile Community Rules Classification - Documentation

## Overview

This project implements machine learning models to classify Reddit comments according to whether they violate specific community rules. It was developed for the Kaggle competition [Jigsaw – Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules).

## Project Components

### Data Processing

#### `src/preprocess.py`

The `DataPreprocessor` class handles text preprocessing tasks:

- Text cleaning (removing URLs, user mentions, HTML tags)
- Tokenization
- Feature extraction
- Train/validation/test splitting

Usage:
```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor()
clean_text = preprocessor.clean_text(raw_text)
```

### Model Training

#### `src/train_baseline.py`

Implements a baseline model using TF-IDF vectorization and Logistic Regression:

- TF-IDF vectorization with n-grams
- Logistic Regression with class balancing
- Basic hyperparameter tuning
- Model evaluation and metrics reporting

#### `src/train_improved.py`

Implements an improved model using sentence transformers:

- Sentence embeddings from pre-trained models
- Multiple classifier options (Logistic Regression, Random Forest)
- Cross-validation
- Data augmentation techniques
- Advanced evaluation metrics

#### `src/fine_tune_model.py`

Implements hyperparameter optimization:

- Grid search for optimal parameters
- Feature engineering
- Model comparison and selection

### Prediction

#### `src/predict.py` and `src/predict_improved.py`

These modules handle making predictions with trained models:

- Loading trained models
- Processing new data
- Generating predictions
- Creating submission files

### API

#### `src/api.py`

Provides a FastAPI-based REST API for model serving:

- `/predict` endpoint for making predictions
- Request/response validation with Pydantic
- Health check endpoint
- Swagger documentation

### Experiment Tracking

#### `src/experiment_tracking.py`

Integrates MLflow for experiment tracking:

- Parameter logging
- Metric tracking
- Model versioning
- Artifact storage

## Testing

The `tests/` directory contains unit tests for the project:

- `test_utils.py`: Tests for utility functions
- `test_preprocess.py`: Tests for preprocessing functions

Run tests with pytest:
```bash
python -m pytest tests/
```

## Containerization

The project includes Docker support:

- `Dockerfile`: Defines the container image
- `docker-compose.yml`: Defines services for the application

## CI/CD

GitHub Actions workflows in `.github/workflows/` automate testing and deployment:

- `ci.yml`: Runs tests and linting on push and pull requests

## Code Quality

The project uses several tools to maintain code quality:

- Flake8 for linting (configuration in `.flake8`)
- Black for code formatting (configuration in `pyproject.toml`)
- isort for import sorting (configuration in `pyproject.toml`)

## Environment Setup

- `setup_env.bat`: Sets up the Python virtual environment on Windows
- `requirements.txt`: Lists all project dependencies

## Running the Pipeline

- `run_ml_pipeline.bat` or `src/run_ml_pipeline.py`: Runs the complete ML pipeline
- `run_api.bat`: Starts the API server
- `run_tests.bat`: Runs the test suite