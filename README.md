# 🚀 Explainable AI Text Classification Framework

A comprehensive framework for text classification with explainable AI (XAI) techniques, featuring multi-dataset integration, transfer learning, and automated benchmarking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Academic Materials](#academic-materials)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides a complete framework for understanding and implementing explainable AI techniques in text classification. It includes hands-on lab exercises, multi-dataset integration, transfer learning capabilities, and automated benchmarking tools.

### Key Objectives

- **Educational**: Learn XAI techniques through hands-on exercises
- **Practical**: Apply XAI to real-world text classification problems
- **Comprehensive**: Compare models across multiple datasets
- **Scalable**: Automated benchmarking and evaluation framework

## ✨ Features

### 🔬 Explainable AI Techniques
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- **Feature Importance Analysis**
- **Decision Boundary Visualization**
- **Model Comparison and Analysis**

### 📊 Dataset Integration
- **IMDB Movie Reviews** - Sentiment analysis
- **AG News** - News categorization
- **Jigsaw Toxic Comments** - Toxicity detection
- **Twitter Hate Speech** - Hate speech detection
- **Fake News Detection** - News authenticity
- **Extensible framework** for adding new datasets

### 🧠 Machine Learning Models
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Transfer Learning** with pre-trained embeddings

### 🔄 Advanced Features
- **Transfer Learning Framework**
- **Cross-Dataset Evaluation**
- **Domain Adaptation Techniques**
- **Automated Benchmarking System**
- **RESTful API with XAI Integration**

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd jigsaw-agile-community-rules-ml

# Install basic requirements
pip install -r requirements.txt
```

### Full Installation (with XAI libraries)

```bash
# Install all dependencies including optional XAI packages
pip install -r requirements.txt
pip install lime shap sentence-transformers transformers torch
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 jupyter
```

## 🚀 Quick Start

### 1. Run Complete Demo

```bash
# Run the comprehensive demonstration
python run_complete_demo.py
```

### 2. XAI Lab Exercises

```python
from src.xai_lab_exercises import XAILabExercises

# Initialize and run all exercises
xai_lab = XAILabExercises(dataset_name='imdb', sample_size=1000)
xai_lab.run_all_exercises()
```

### 3. Dataset Integration

```python
from src.dataset_integration import DatasetManager

# Load and explore datasets
manager = DatasetManager()
train_df, test_df = manager.load_dataset('imdb', sample_size=500)
print(manager.get_dataset_info('imdb'))
```

### 4. Transfer Learning

```python
from src.transfer_learning import TransferLearningFramework

# Run transfer learning analysis
framework = TransferLearningFramework()
framework.load_datasets(['imdb', 'ag_news'])
results = framework.run_complete_evaluation()
```

### 5. Start API Server

```bash
# Start the FastAPI server
uvicorn src.api:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
```

## 📁 Project Structure

```
jigsaw-agile-community-rules-ml/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 run_complete_demo.py              # Complete demonstration script
├── 📄 ACADEMIC_REPORT_TEMPLATE.md       # Academic report template
├── 📄 EXPLAINABLE_AI_PRESENTATION.md    # Presentation outline
├── 📄 ACADEMIC_QUIZ.md                  # Knowledge assessment quiz
├── 📄 ACADEMIC_LAB_EXERCISE.md          # Structured lab exercises
├── 📁 src/                              # Source code
│   ├── 📄 __init__.py
│   ├── 📄 api.py                        # FastAPI with XAI integration
│   ├── 📄 dataset_integration.py        # Multi-dataset loader
│   ├── 📄 explainable_ai.py            # Core XAI techniques
│   ├── 📄 xai_lab_exercises.py         # Hands-on lab exercises
│   ├── 📄 transfer_learning.py         # Transfer learning framework
│   └── 📄 automated_benchmarking.py    # Benchmarking system
├── 📁 notebooks/                        # Jupyter notebooks
│   ├── 📄 cross_dataset_evaluation.ipynb
│   ├── 📄 explainable_ai_demo.ipynb
│   └── 📄 ml_fundamentals_tutorial.ipynb
├── 📁 tests/                           # Unit tests
│   ├── 📄 __init__.py
│   └── 📄 test_explainable_ai.py
├── 📁 outputs/                         # Generated results (created during execution)
│   ├── 📁 xai_lab_exercises/
│   ├── 📁 transfer_learning/
│   └── 📁 benchmarks/
└── 📁 data/                            # Datasets (if available)
```

## 💡 Usage Examples

### Example 1: Basic XAI Analysis

```python
from src.explainable_ai import ExplainableAI
from src.dataset_integration import DatasetManager

# Load dataset
manager = DatasetManager()
train_df, test_df = manager.load_dataset('imdb', sample_size=1000)

# Initialize XAI
xai = ExplainableAI()
X_train, X_test, y_train, y_test = xai.prepare_data(train_df, test_df)

# Train model
model = xai.train_model(X_train, y_train, model_type='logistic_regression')

# Generate explanations
lime_explanation = xai.explain_prediction_lime(model, X_test[0], feature_names)
shap_values = xai.explain_prediction_shap(model, X_test[:100])

# Visualize results
xai.visualize_lime_explanation(lime_explanation)
xai.visualize_shap_summary(shap_values, X_test[:100])
```

### Example 2: Cross-Dataset Evaluation

```python
from src.transfer_learning import TransferLearningFramework

# Initialize framework
framework = TransferLearningFramework(embedding_type='sentence_transformer')

# Load multiple datasets
datasets = ['imdb', 'ag_news', 'jigsaw_toxic']
framework.load_datasets(datasets, sample_size=500)

# Run cross-domain evaluation
results = framework.cross_domain_evaluation()

# Visualize transfer learning performance
framework.visualize_transfer_matrix(results)

# Generate comprehensive report
report = framework.generate_comprehensive_report()
print(report)
```

### Example 3: Automated Benchmarking

```python
from src.automated_benchmarking import AutomatedBenchmark

# Initialize benchmark system
benchmark = AutomatedBenchmark(output_dir='outputs/my_benchmark')

# Define experimental configuration
datasets = ['imdb', 'ag_news']
models = ['logistic_regression', 'random_forest', 'svm']
embeddings = ['tfidf', 'sentence_transformer']

# Run comprehensive benchmark
results = benchmark.run_comprehensive_benchmark(
    dataset_names=datasets,
    model_names=models,
    embedding_types=embeddings,
    sample_size=1000
)

# Analyze results
benchmark.visualize_results()
best_configs = benchmark.get_best_configurations(top_k=5)
insights = benchmark.generate_academic_insights()
```

### Example 4: API Usage

```python
import requests

# Start the API server first: uvicorn src.api:app --reload

# Basic prediction
response = requests.post(
    "http://localhost:8000/predict/",
    json={"text": "This movie was absolutely fantastic!"}
)
print(response.json())

# Prediction with explanation
response = requests.post(
    "http://localhost:8000/explain/",
    json={
        "text": "This movie was absolutely fantastic!",
        "rule_text": "Positive sentiment analysis"
    }
)
print(response.json())
```

## 📚 Academic Materials

### Lab Exercises
- **ACADEMIC_LAB_EXERCISE.md**: Structured learning path with exercises
- **XAI Lab Exercises**: Hands-on coding exercises with step-by-step guidance
- **Jupyter Notebooks**: Interactive tutorials and demonstrations

### Assessment
- **ACADEMIC_QUIZ.md**: Knowledge assessment with multiple-choice questions
- **Practical Exercises**: Code-based assignments and projects

### Research
- **ACADEMIC_REPORT_TEMPLATE.md**: Template for academic papers
- **EXPLAINABLE_AI_PRESENTATION.md**: Presentation outline for conferences

## 🌐 API Documentation

### Endpoints

#### POST `/predict/`
Basic text classification prediction.

**Request Body:**
```json
{
  "text": "Your text to classify"
}
```

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.85,
  "model_info": {...}
}
```

#### POST `/explain/`
Text classification with XAI explanations.

**Request Body:**
```json
{
  "text": "Your text to classify",
  "rule_text": "Description of the classification rule"
}
```

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.85,
  "explanations": {
    "lime": {...},
    "shap": {...},
    "feature_importance": [...]
  }
}
```

### Interactive Documentation
Visit `http://localhost:8000/docs` when the server is running for interactive API documentation.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_explainable_ai.py
```

## 📈 Performance Considerations

### Memory Usage
- Use `sample_size` parameter to limit dataset size for testing
- Consider using TF-IDF instead of transformer embeddings for large datasets
- Monitor memory usage during benchmarking

### Computational Requirements
- XAI techniques (especially SHAP) can be computationally intensive
- Transfer learning with transformers requires significant GPU/CPU resources
- Consider using smaller models for initial experimentation

### Optimization Tips
- Use `embedding_type='tfidf'` for faster experimentation
- Reduce `sample_size` for quicker iterations
- Enable multiprocessing where available
- Cache embeddings for repeated experiments

## 🤝 Contributing

### Development Setup

```bash
# Fork and clone the repository
git clone <your-fork-url>
cd jigsaw-agile-community-rules-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .
pip install pytest black flake8 jupyter
```

### Code Style

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run tests
pytest
```

### Adding New Datasets

1. Create a new loader class in `src/dataset_integration.py`
2. Inherit from `DatasetLoader` base class
3. Implement required methods: `download_data()`, `load_data()`, `get_info()`
4. Add the dataset to `DatasetManager.AVAILABLE_DATASETS`
5. Update tests and documentation

### Adding New XAI Techniques

1. Add new methods to `src/explainable_ai.py`
2. Follow the existing pattern for explanation methods
3. Add visualization functions
4. Update lab exercises and notebooks
5. Add unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- **Scikit-learn**: Machine learning library
- **Transformers**: Hugging Face transformers library
- **FastAPI**: Modern web framework for APIs

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Open an issue on GitHub
2. **Discussions**: Use GitHub Discussions for questions
3. **Documentation**: Check the academic materials and notebooks
4. **Examples**: Run `python run_complete_demo.py` for comprehensive examples

---

**Happy Learning with Explainable AI! 🎉**
