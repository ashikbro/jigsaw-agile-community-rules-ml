# 🚀 XAI Framework Upgrade Guide

## 📋 Overview

This guide will help you upgrade your existing XAI text classification framework to a state-of-the-art academic research platform with advanced models, comprehensive datasets, and cutting-edge explainability techniques.

## 🎯 What You'll Get After Upgrading

### 🤖 Advanced Models
- **Transformer Models**: BERT, RoBERTa, DistilBERT, ELECTRA, DeBERTa, ALBERT
- **Ensemble Methods**: Voting classifiers, stacking ensembles
- **Few-Shot Learning**: Prototypical networks for low-resource scenarios
- **Domain-Specific Models**: SciBERT, ClinicalBERT, LegalBERT, FinBERT

### 📊 Comprehensive Datasets
- **Academic Benchmarks**: SST-2, IMDB, AG News, Yelp, Amazon reviews
- **Specialized Domains**: Scientific papers, legal documents, financial texts
- **Multilingual Support**: Cross-lingual evaluation datasets
- **Challenging Sets**: Adversarial examples, out-of-distribution data

### 🔍 Advanced XAI Techniques
- **Attention Visualization**: Multi-head attention heatmaps
- **Integrated Gradients**: Attribution-based explanations
- **Counterfactual Explanations**: What-if analysis
- **Causal Methods**: Mediation analysis, instrumental variables
- **Probing Tasks**: Layer-wise representation analysis

### 📈 Enhanced Evaluation
- **XAI-Specific Metrics**: Faithfulness, stability, comprehensiveness
- **Cross-Domain Analysis**: Transfer learning evaluation
- **Human Evaluation Framework**: Structured assessment protocols
- **Robustness Testing**: Adversarial and out-of-distribution evaluation

## 🛠️ Installation Guide

### Step 1: Environment Setup

#### Option A: Create New Conda Environment (Recommended)
```bash
# Create new environment with Python 3.9
conda create -n xai-advanced python=3.9
conda activate xai-advanced

# Install PyTorch (choose based on your system)
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU (CUDA 11.7):
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### Option B: Use Existing Environment
```bash
# Activate your existing environment
conda activate your-env-name
# or
source venv/bin/activate  # for virtualenv
```

### Step 2: Install Core Dependencies

#### Basic Installation (Essential packages)
```bash
# Install core ML libraries
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Install transformers and datasets
pip install transformers datasets tokenizers huggingface-hub

# Install XAI libraries
pip install shap lime captum

# Install additional utilities
pip install tqdm rich click
```

#### Full Installation (All advanced features)
```bash
# Install all advanced dependencies
pip install -r requirements_advanced.txt
```

#### Selective Installation (Choose what you need)
```bash
# For transformer models only
pip install torch transformers datasets tokenizers

# For advanced XAI only
pip install captum shap lime alibi

# For ensemble methods only
pip install scikit-learn xgboost lightgbm

# For visualization only
pip install plotly bokeh altair
```

### Step 3: Verify Installation

```python
# Run this in Python to verify key components
import torch
import transformers
import datasets
import captum
import shap
import lime

print("✅ All core libraries installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
```

## 🚀 Quick Start: Running the Upgrade

### Option 1: Full Automatic Upgrade
```bash
# Run complete upgrade (recommended for first-time users)
python upgrade_framework.py --mode full
```

### Option 2: Selective Upgrades
```bash
# Upgrade models only
python upgrade_framework.py --mode models_only

# Upgrade datasets only
python upgrade_framework.py --mode datasets_only

# Upgrade XAI techniques only
python upgrade_framework.py --mode xai_only
```

### Option 3: Custom Output Directory
```bash
# Specify custom output directory
python upgrade_framework.py --mode full --output_dir "my_upgraded_framework"
```

## 📊 Step-by-Step Manual Upgrade

### Step 1: Advanced Model Integration

```python
from upgrade_framework import AdvancedModelManager

# Initialize model manager
model_manager = AdvancedModelManager()

# Check available models
print("Available models:", model_manager.available_models)

# Load specific models
model_manager.load_model('bert-base')
model_manager.load_model('roberta-base')
model_manager.load_model('distilbert')

# Get embeddings
texts = ["This is a great movie!", "I didn't like this film."]
embeddings = model_manager.get_embeddings('bert-base', texts)
print(f"Embeddings shape: {embeddings.shape}")
```

### Step 2: Academic Dataset Integration

```python
from upgrade_framework import AcademicDatasetLoader

# Initialize dataset loader
dataset_loader = AcademicDatasetLoader()

# Check available datasets
print("Available datasets:", dataset_loader.available_datasets)

# Load specific datasets
imdb_data = dataset_loader.load_dataset('imdb', max_samples=1000)
ag_news_data = dataset_loader.load_dataset('ag_news', max_samples=1000)

print(f"IMDB data shape: {imdb_data.shape}")
print(f"AG News data shape: {ag_news_data.shape}")
```

### Step 3: Advanced XAI Implementation

```python
from upgrade_framework import AdvancedXAI

# Assuming you have a loaded model and tokenizer
model = model_manager.models['bert-base']
tokenizer = model_manager.tokenizers['bert-base']

# Initialize XAI
xai = AdvancedXAI(model, tokenizer)

# Get attention weights
test_text = "This movie has excellent acting and plot."
attention_weights = xai.get_attention_weights(test_text)

# Visualize attention
xai.visualize_attention(test_text, save_path='attention_viz.png')

# Integrated gradients analysis
ig_results = xai.integrated_gradients_analysis(test_text)
if ig_results:
    print(f"Tokens: {ig_results['tokens'][:10]}")
    print(f"Attributions shape: {ig_results['attributions'].shape}")
```

### Step 4: Ensemble Methods

```python
from upgrade_framework import EnsembleFramework
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Initialize ensemble framework
ensemble_framework = EnsembleFramework()

# Prepare data (using IMDB as example)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(imdb_data['text']).toarray()
y = imdb_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and evaluate voting ensemble
voting_ensemble = ensemble_framework.create_voting_ensemble()
voting_results = ensemble_framework.evaluate_ensemble(
    voting_ensemble, X_train, y_train, X_test, y_test
)

print(f"Voting Ensemble Accuracy: {voting_results['accuracy']:.3f}")
print(f"Voting Ensemble F1-Score: {voting_results['f1_score']:.3f}")

# Create and evaluate stacking ensemble
stacking_ensemble = ensemble_framework.create_stacking_ensemble()
stacking_results = ensemble_framework.evaluate_ensemble(
    stacking_ensemble, X_train, y_train, X_test, y_test
)

print(f"Stacking Ensemble Accuracy: {stacking_results['accuracy']:.3f}")
print(f"Stacking Ensemble F1-Score: {stacking_results['f1_score']:.3f}")
```

### Step 5: Advanced Evaluation

```python
from upgrade_framework import AdvancedEvaluationMetrics

# Initialize evaluation metrics
evaluation_metrics = AdvancedEvaluationMetrics()

# Compute XAI metrics (simplified example)
dummy_explanations = [np.random.random(X.shape[1]) for _ in range(len(X_test))]
xai_metrics = evaluation_metrics.compute_xai_metrics(
    voting_ensemble, X_test, dummy_explanations
)

print("XAI Metrics:")
for metric_name, metric_value in xai_metrics.items():
    print(f"  {metric_name}: {metric_value}")
```

## 🎯 Usage Examples

### Example 1: Complete Research Pipeline

```python
# Complete research pipeline example
from upgrade_framework import UpgradeFramework

# Initialize upgrade framework
upgrader = UpgradeFramework('outputs/research_pipeline')

# Run full upgrade
success = upgrader.run_full_upgrade()

if success:
    print("🎉 Research pipeline ready!")
    
    # Generate comprehensive report
    report_path = upgrader.generate_upgrade_report()
    print(f"📋 Detailed report: {report_path}")
    
    # Access results
    print(f"Models loaded: {len(upgrader.results['models'])}")
    print(f"Datasets loaded: {len(upgrader.results['datasets'])}")
    print(f"XAI techniques: {len(upgrader.results['xai_analyses'])}")
else:
    print("⚠️ Upgrade completed with issues. Check logs for details.")
```

### Example 2: Cross-Domain Evaluation

```python
# Cross-domain evaluation example
from upgrade_framework import AcademicDatasetLoader, AdvancedModelManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets from different domains
dataset_loader = AcademicDatasetLoader()

# Sentiment analysis datasets
imdb_data = dataset_loader.load_dataset('imdb', max_samples=500)
yelp_data = dataset_loader.load_dataset('yelp_polarity', max_samples=500)

# News classification datasets
ag_news_data = dataset_loader.load_dataset('ag_news', max_samples=500)

if imdb_data is not None and yelp_data is not None:
    # Cross-domain evaluation: Train on IMDB, test on Yelp
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Train on IMDB
    X_train = vectorizer.fit_transform(imdb_data['text'])
    y_train = imdb_data['label']
    
    # Test on Yelp
    X_test = vectorizer.transform(yelp_data['text'])
    y_test = yelp_data['label']
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Cross-domain accuracy (IMDB → Yelp): {accuracy:.3f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
```

### Example 3: Advanced XAI Analysis

```python
# Advanced XAI analysis example
import matplotlib.pyplot as plt
import numpy as np

# Load a transformer model
model_manager = AdvancedModelManager()
model_manager.load_model('bert-base')

model = model_manager.models['bert-base']
tokenizer = model_manager.tokenizers['bert-base']

# Initialize XAI
xai = AdvancedXAI(model, tokenizer)

# Analyze multiple texts
texts = [
    "This movie is absolutely fantastic with great acting!",
    "The film was boring and poorly executed.",
    "An average movie with decent plot but weak characters."
]

for i, text in enumerate(texts):
    print(f"\n--- Analysis for Text {i+1} ---")
    print(f"Text: {text}")
    
    # Attention analysis
    attention_weights = xai.get_attention_weights(text)
    if attention_weights is not None:
        print(f"Attention shape: {attention_weights.shape}")
        
        # Save attention visualization
        xai.visualize_attention(
            text, 
            save_path=f'outputs/attention_text_{i+1}.png'
        )
    
    # Integrated gradients analysis
    ig_results = xai.integrated_gradients_analysis(text)
    if ig_results:
        tokens = ig_results['tokens']
        attributions = ig_results['attributions']
        
        # Find most important tokens
        top_indices = np.argsort(np.abs(attributions))[-5:]
        print("Top 5 important tokens:")
        for idx in reversed(top_indices):
            if idx < len(tokens):
                print(f"  {tokens[idx]}: {attributions[idx]:.3f}")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA/GPU Problems
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA issues, install CPU-only version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: Memory Issues
```python
# Reduce batch size and model size
model_manager = AdvancedModelManager()
# Use smaller models
model_manager.load_model('distilbert')  # Instead of 'bert-large'

# Reduce dataset size
dataset = dataset_loader.load_dataset('imdb', max_samples=100)  # Instead of 1000
```

#### Issue 3: Missing Dependencies
```bash
# Install missing packages individually
pip install transformers
pip install datasets
pip install captum
pip install shap
pip install lime

# Or install from requirements
pip install -r requirements_advanced.txt
```

#### Issue 4: Hugging Face Hub Issues
```bash
# Login to Hugging Face (if needed for some models)
huggingface-cli login

# Or set cache directory
export HF_HOME=/path/to/cache
```

### Performance Optimization

#### For Limited Resources
```python
# Use smaller models
SMALL_MODELS = ['distilbert', 'albert-base']

# Reduce dataset sizes
MAX_SAMPLES = 200

# Use CPU-only mode
device = 'cpu'
```

#### For High Performance
```python
# Use larger models
LARGE_MODELS = ['bert-large', 'roberta-large']

# Increase dataset sizes
MAX_SAMPLES = 5000

# Use GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 📊 Expected Results

After successful upgrade, you should see:

### ✅ Models Loaded
- 3-5 transformer models successfully loaded
- Ensemble methods available
- Model embeddings working

### ✅ Datasets Integrated
- 5-8 academic datasets loaded
- Cross-domain evaluation possible
- Standardized data format

### ✅ XAI Enhanced
- Attention visualization working
- Integrated gradients analysis available
- Advanced explanation methods ready

### ✅ Evaluation Framework
- XAI-specific metrics computed
- Cross-domain analysis possible
- Comprehensive reporting available

## 🎓 Academic Usage

### For Research Papers
1. **Baseline Comparison**: Use multiple transformer models
2. **Cross-Domain Analysis**: Evaluate transfer learning
3. **XAI Evaluation**: Apply faithfulness and stability metrics
4. **Human Studies**: Use evaluation framework templates

### For Course Projects
1. **Start Simple**: Use basic models and small datasets
2. **Add Complexity**: Gradually introduce advanced features
3. **Compare Methods**: Evaluate different XAI techniques
4. **Document Results**: Use automated reporting

### For Thesis Work
1. **Comprehensive Analysis**: Use full framework capabilities
2. **Novel Contributions**: Extend existing methods
3. **Rigorous Evaluation**: Apply all evaluation metrics
4. **Reproducible Results**: Use version control and documentation

## 🚀 Next Steps

After upgrading, consider:

1. **Explore Advanced Features**: Try causal XAI methods
2. **Add Custom Datasets**: Integrate domain-specific data
3. **Develop Novel Methods**: Extend the framework
4. **Publish Results**: Share your findings with the community
5. **Contribute Back**: Submit improvements to the project

## 📞 Support

If you encounter issues:

1. **Check Logs**: Look for error messages in console output
2. **Review Documentation**: Read the implementation plan and enhancement roadmap
3. **Test Components**: Run individual components to isolate issues
4. **Reduce Complexity**: Start with basic features and add gradually
5. **Community Support**: Share your experience and get help from others

---

**🎉 Congratulations!** You now have a state-of-the-art XAI framework ready for academic research and publication-worthy experiments!