# 🚀 Jigsaw Agile Community Rules ML

A comprehensive machine learning framework for text classification with explainable AI techniques, featuring multi-dataset integration and automated benchmarking.

## 🎯 Overview

This project provides a complete framework for text classification with explainable AI (XAI) capabilities. It includes support for multiple datasets, various ML models, and comprehensive evaluation tools.

## ✨ Key Features

- **Explainable AI**: LIME, SHAP, feature importance analysis
- **Multi-Dataset Support**: IMDB, AG News, Jigsaw Toxic Comments, Twitter Hate Speech
- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, Naive Bayes
- **Automated Benchmarking**: Performance comparison across models and datasets
- **API Server**: RESTful API for model predictions
- **Interactive Demos**: Web-based interfaces for testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jigsaw-agile-community-rules-ml.git
cd jigsaw-agile-community-rules-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Train a simple model
python train_simple_model.py

# Run API server
python api_server.py

# Start interactive demo
python streamlit_demo.py
```

## 📁 Project Structure

```
├── src/                    # Core source code
├── data/                   # Dataset storage
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
├── demos/                  # Interactive demos
├── utils/                  # Helper utilities
├── eda/                    # Exploratory data analysis
├── config/                 # Configuration files
└── reports/                # Generated reports
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters
- Dataset paths
- API settings
- Evaluation metrics

## 📊 Supported Datasets

- **IMDB Movie Reviews** - Sentiment analysis
- **AG News** - News categorization  
- **Jigsaw Toxic Comments** - Toxicity detection
- **Twitter Hate Speech** - Hate speech detection

## 🤖 Available Models

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Ensemble Methods

## 📈 Explainable AI Features

- **LIME**: Local model explanations
- **SHAP**: Feature importance analysis
- **Visualization**: Decision boundary plots
- **Comparison**: Model performance analysis

## 🌐 API Usage

```python
import requests

# Predict text classification
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Your text here'})
result = response.json()
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_preprocess.py
```

## 📚 Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Detailed technical documentation
- [DEMO_GUIDE.md](DEMO_GUIDE.md) - Interactive demo guide
- [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Docker deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Jigsaw/Conversation AI for the toxic comments dataset
- The open-source ML community for tools and libraries
- Contributors and maintainers

---

**Made with ❤️ for the ML community**
