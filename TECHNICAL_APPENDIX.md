# Technical Appendix: Multi-Dataset Text Classification with Explainable AI

## A. Implementation Details

### A.1 Model Architecture Specifications

#### Baseline Models

**Logistic Regression Configuration:**
```python
LogisticRegression(
    C=1.0,                    # Regularization strength
    penalty='l2',             # L2 regularization
    solver='liblinear',       # Optimization algorithm
    max_iter=1000,           # Maximum iterations
    random_state=42          # Reproducibility
)
```

**Random Forest Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Unlimited depth
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples per leaf
    random_state=42          # Reproducibility
)
```

#### Advanced Models

**Sentence Transformer Configuration:**
```python
SentenceTransformer('all-mpnet-base-v2')
# Model specifications:
# - 768-dimensional embeddings
# - Maximum sequence length: 384 tokens
# - Pre-trained on 1B+ sentence pairs
# - Optimized for semantic similarity
```

**Fine-tuned BERT Configuration:**
```python
AutoModel.from_pretrained('bert-base-uncased')
# Model specifications:
# - 12 transformer layers
# - 768 hidden dimensions
# - 12 attention heads
# - 110M parameters
```

### A.2 Feature Engineering Pipeline

#### Text Preprocessing

```python
def preprocess_text(text):
    """
    Comprehensive text preprocessing pipeline.
    """
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. URL removal
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. HTML tag removal
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Special character handling
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 5. Multiple whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

#### TF-IDF Vectorization

```python
TfidfVectorizer(
    max_features=10000,      # Maximum vocabulary size
    ngram_range=(1, 2),      # Unigrams and bigrams
    stop_words='english',    # Remove English stop words
    min_df=2,               # Minimum document frequency
    max_df=0.95,            # Maximum document frequency
    sublinear_tf=True       # Apply sublinear TF scaling
)
```

### A.3 Evaluation Methodology

#### Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

# 5-fold stratified cross-validation
kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluation metrics
scoring = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'roc_auc'
]
```

#### Statistical Significance Testing

```python
from scipy.stats import ttest_rel

def compare_models(scores1, scores2, alpha=0.05):
    """
    Paired t-test for model comparison.
    """
    statistic, p_value = ttest_rel(scores1, scores2)
    is_significant = p_value < alpha
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': is_significant,
        'effect_size': np.mean(scores1) - np.mean(scores2)
    }
```

## B. Explainable AI Implementation

### B.1 LIME Configuration

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(
    class_names=['Non-Toxic', 'Toxic'],
    feature_selection='auto',
    split_expression='\\W+',
    bow=True,
    mode='classification'
)

# Generate explanation
explanation = explainer.explain_instance(
    text_instance,
    classifier.predict_proba,
    num_features=10,
    num_samples=5000
)
```

### B.2 SHAP Implementation

```python
import shap

# For tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For linear models
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Visualization
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### B.3 Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names, top_k=20):
    """
    Extract and analyze feature importance.
    """
    if hasattr(model, 'coef_'):
        # Linear model coefficients
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # Tree-based model importance
        importance = model.feature_importances_
    else:
        raise ValueError("Model does not support feature importance")
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1][:top_k]
    
    return {
        'features': [feature_names[i] for i in indices],
        'importance': importance[indices],
        'indices': indices
    }
```

## C. Experimental Results

### C.1 Model Performance Comparison

| Model | Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|---------|----------|-----------|--------|----------|----------|
| Logistic Regression | IMDB | 0.891 ± 0.003 | 0.892 ± 0.004 | 0.890 ± 0.005 | 0.891 ± 0.003 | 0.951 ± 0.002 |
| Random Forest | IMDB | 0.847 ± 0.005 | 0.849 ± 0.006 | 0.845 ± 0.007 | 0.847 ± 0.005 | 0.923 ± 0.004 |
| SentenceTransformer | IMDB | 0.934 ± 0.002 | 0.935 ± 0.003 | 0.933 ± 0.003 | 0.934 ± 0.002 | 0.978 ± 0.001 |
| Fine-tuned BERT | IMDB | 0.952 ± 0.001 | 0.953 ± 0.002 | 0.951 ± 0.002 | 0.952 ± 0.001 | 0.987 ± 0.001 |

### C.2 Cross-Dataset Transfer Learning Results

#### Transfer Learning Matrix (F1-Scores)

| Source → Target | IMDB | AG News | Jigsaw | Twitter |
|----------------|------|---------|--------|----------|
| IMDB | 0.934 | 0.782 | 0.691 | 0.673 |
| AG News | 0.756 | 0.912 | 0.634 | 0.645 |
| Jigsaw | 0.698 | 0.723 | 0.887 | 0.798 |
| Twitter | 0.681 | 0.701 | 0.812 | 0.856 |

### C.3 Explainability Analysis

#### LIME vs SHAP Comparison

| Metric | LIME | SHAP |
|--------|------|------|
| Explanation Time (ms) | 2,340 ± 156 | 45 ± 8 |
| Feature Consistency | 0.73 ± 0.12 | 0.91 ± 0.05 |
| Human Interpretability | 4.2/5.0 | 4.6/5.0 |
| Computational Overhead | High | Low |

## D. Hyperparameter Optimization

### D.1 Grid Search Configuration

```python
from sklearn.model_selection import GridSearchCV

# Logistic Regression hyperparameters
lr_params = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Random Forest hyperparameters
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
```

### D.2 Bayesian Optimization

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search space
search_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'penalty': Categorical(['l1', 'l2']),
    'solver': Categorical(['liblinear', 'saga'])
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=LogisticRegression(),
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
```

## E. Computational Requirements

### E.1 Hardware Specifications

- **CPU**: Intel Core i7-10700K (8 cores, 16 threads)
- **RAM**: 32 GB DDR4-3200
- **GPU**: NVIDIA RTX 3080 (10 GB VRAM)
- **Storage**: 1 TB NVMe SSD

### E.2 Training Time Analysis

| Model | Dataset Size | Training Time | Memory Usage |
|-------|-------------|---------------|---------------|
| Logistic Regression | 50K samples | 2.3 minutes | 1.2 GB |
| Random Forest | 50K samples | 8.7 minutes | 2.8 GB |
| SentenceTransformer | 50K samples | 15.4 minutes | 4.1 GB |
| Fine-tuned BERT | 50K samples | 2.3 hours | 8.9 GB |

### E.3 Inference Performance

| Model | Batch Size | Inference Time (ms) | Throughput (samples/sec) |
|-------|------------|--------------------|--------------------------|
| Logistic Regression | 1000 | 12 ± 2 | 83,333 |
| Random Forest | 1000 | 45 ± 5 | 22,222 |
| SentenceTransformer | 32 | 156 ± 12 | 205 |
| Fine-tuned BERT | 16 | 234 ± 18 | 68 |

## F. Error Analysis

### F.1 Common Misclassification Patterns

#### Sentiment Analysis (IMDB)

**False Positives (Predicted Positive, Actually Negative):**
- Sarcastic reviews with positive words
- Mixed sentiment with positive conclusion
- Technical discussions about film quality

**False Negatives (Predicted Negative, Actually Positive):**
- Subtle positive sentiment
- Comparative reviews
- Genre-specific positive language

### F.2 Confusion Matrix Analysis

```python
from sklearn.metrics import confusion_matrix, classification_report

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Detailed classification report
report = classification_report(
    y_true, y_pred,
    target_names=['Non-Toxic', 'Toxic'],
    output_dict=True
)
```

## G. Reproducibility Guidelines

### G.1 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set random seeds
export PYTHONHASHSEED=42
```

### G.2 Random Seed Configuration

```python
import random
import numpy as np
import torch
from sklearn.utils import check_random_state

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### G.3 Experiment Logging

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'model_type': 'LogisticRegression',
        'C': 1.0,
        'penalty': 'l2',
        'dataset': 'IMDB'
    })
    
    # Log metrics
    mlflow.log_metrics({
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    })
    
    # Log model
    mlflow.sklearn.log_model(model, 'model')
```

## H. Future Work and Limitations

### H.1 Current Limitations

1. **Dataset Bias**: Limited diversity in training datasets
2. **Computational Constraints**: Large model fine-tuning requirements
3. **Explanation Fidelity**: Gap between local and global explanations
4. **Evaluation Metrics**: Limited human evaluation of explanations

### H.2 Proposed Improvements

1. **Multi-modal Integration**: Incorporate image and audio data
2. **Adversarial Robustness**: Evaluate model robustness to attacks
3. **Fairness Analysis**: Systematic bias detection and mitigation
4. **Real-time Deployment**: Optimize for production environments

### H.3 Research Directions

1. **Causal Explanations**: Move beyond correlation to causation
2. **Interactive XAI**: User-in-the-loop explanation refinement
3. **Cross-lingual Transfer**: Multilingual model evaluation
4. **Temporal Analysis**: Time-series text classification

---

*This technical appendix provides comprehensive implementation details, experimental results, and reproducibility guidelines for the multi-dataset text classification with explainable AI study.*