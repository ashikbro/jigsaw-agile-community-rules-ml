#!/usr/bin/env python3
"""
Project Setup Script for Reddit Comment Rule Violation Detection

This script helps users set up the project environment, install dependencies,
download required models, and verify the installation.

Usage:
    python project_setup.py --install-deps    # Install Python dependencies
    python project_setup.py --download-models # Download required NLP models
    python project_setup.py --verify          # Verify installation
    python project_setup.py --all             # Do everything

Author: AI Assistant
Date: 2024
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectSetup:
    """
    Handles the setup and configuration of the Reddit moderation project.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # Create necessary directories
        self.create_directories()
    
    def create_directories(self) -> None:
        """
        Create necessary project directories.
        """
        directories = [
            self.src_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "submissions",
            self.models_dir,
            self.project_root / "notebooks",
            self.project_root / "logs",
            self.project_root / "outputs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self) -> bool:
        """
        Install Python dependencies from requirements.txt.
        """
        logger.info("Installing Python dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            logger.info("Upgraded pip successfully")
            
            # Install requirements
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True, capture_output=True, text=True
            )
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def download_nlp_models(self) -> bool:
        """
        Download required NLP models and data.
        """
        logger.info("Downloading NLP models and data...")
        
        models_to_download = [
            # NLTK data
            ("nltk", "punkt"),
            ("nltk", "stopwords"),
            ("nltk", "vader_lexicon"),
            ("nltk", "wordnet"),
            ("nltk", "averaged_perceptron_tagger"),
            
            # spaCy models
            ("spacy", "en_core_web_sm"),
        ]
        
        success = True
        
        for library, model in models_to_download:
            try:
                if library == "nltk":
                    self._download_nltk_data(model)
                elif library == "spacy":
                    self._download_spacy_model(model)
                    
                logger.info(f"Downloaded {library} model: {model}")
                
            except Exception as e:
                logger.error(f"Failed to download {library} model {model}: {e}")
                success = False
        
        return success
    
    def _download_nltk_data(self, data_name: str) -> None:
        """
        Download NLTK data.
        """
        import nltk
        nltk.download(data_name, quiet=True)
    
    def _download_spacy_model(self, model_name: str) -> None:
        """
        Download spaCy model.
        """
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True, capture_output=True, text=True
        )
    
    def verify_installation(self) -> bool:
        """
        Verify that all components are properly installed.
        """
        logger.info("Verifying installation...")
        
        verification_tests = [
            self._test_core_imports,
            self._test_ml_libraries,
            self._test_nlp_libraries,
            self._test_visualization_libraries,
            self._test_project_modules,
            self._test_model_loading
        ]
        
        all_passed = True
        
        for test in verification_tests:
            try:
                test()
                logger.info(f"✓ {test.__name__} passed")
            except Exception as e:
                logger.error(f"✗ {test.__name__} failed: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("🎉 All verification tests passed!")
        else:
            logger.error("❌ Some verification tests failed")
        
        return all_passed
    
    def _test_core_imports(self) -> None:
        """Test core Python libraries."""
        import pandas as pd
        import numpy as np
        import sklearn
        assert pd.__version__ >= "1.3.0"
        assert np.__version__ >= "1.21.0"
    
    def _test_ml_libraries(self) -> None:
        """Test machine learning libraries."""
        import torch
        import transformers
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        import numpy as np
        
        # Test basic functionality
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        clf.fit(X, y)
        pred = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, pred)
        assert 0 <= auc <= 1
    
    def _test_nlp_libraries(self) -> None:
        """Test NLP libraries."""
        import nltk
        import spacy
        from transformers import AutoTokenizer
        
        # Test NLTK
        from nltk.corpus import stopwords
        stops = stopwords.words('english')
        assert len(stops) > 0
        
        # Test spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp("This is a test sentence.")
            assert len(doc) > 0
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found, but spaCy is installed")
        
        # Test transformers
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokens = tokenizer("Test sentence")
        assert 'input_ids' in tokens
    
    def _test_visualization_libraries(self) -> None:
        """Test visualization libraries."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        
        # Test basic plotting
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
    
    def _test_project_modules(self) -> None:
        """Test project-specific modules."""
        sys.path.append(str(self.src_dir))
        
        try:
            from reddit_data_pipeline import RedditDataPipeline
            from reddit_classifier import RedditTransformerClassifier
            from multi_rule_classifier import MultiRuleClassifier
            from evaluation_metrics import KaggleEvaluator
            from xai_integration import XAIIntegrator
            from kaggle_submission import KaggleSubmissionPipeline
            
            logger.info("All project modules imported successfully")
        except ImportError as e:
            raise ImportError(f"Failed to import project modules: {e}")
    
    def _test_model_loading(self) -> None:
        """Test model loading capabilities."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Test loading a small model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Test tokenization
        text = "This is a test sentence for model loading."
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Test model inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.last_hidden_state.shape[0] == 1  # batch size
        assert outputs.last_hidden_state.shape[2] == 768  # hidden size for DistilBERT
    
    def create_sample_config(self) -> None:
        """
        Create a sample configuration file.
        """
        config_content = """
# Reddit Comment Moderation - Configuration File
# Modify these settings according to your needs

# Model Configuration
model:
  name: "distilbert-base-uncased"
  max_length: 128
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  warmup_steps: 100

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42

# Training Configuration
training:
  output_dir: "./models"
  logging_steps: 100
  save_steps: 500
  eval_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_auc"
  greater_is_better: true

# Evaluation Configuration
evaluation:
  metrics: ["auc", "precision", "recall", "f1"]
  threshold: 0.5

# XAI Configuration
xai:
  methods: ["lime", "shap"]
  num_samples: 1000
  max_features: 20

# Kaggle Submission Configuration
kaggle:
  submission_file: "submission.csv"
  max_runtime_hours: 12
  memory_limit_gb: 16
"""
        
        config_file = self.project_root / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created sample configuration file: {config_file}")
    
    def create_gitignore(self) -> None:
        """
        Create a .gitignore file for the project.
        """
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/
data/processed/
*.csv
*.json
*.pkl
*.h5

# Model files
models/
*.bin
*.pt
*.pth
*.onnx

# Logs
logs/
*.log

# Outputs
outputs/
results/
submissions/

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp

# API keys and secrets
.env
*.key
*.secret
config/secrets.yaml

# Large files
*.zip
*.tar.gz
*.rar
"""
        
        gitignore_file = self.project_root / ".gitignore"
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        
        logger.info(f"Created .gitignore file: {gitignore_file}")
    
    def print_setup_summary(self) -> None:
        """
        Print a summary of the setup process.
        """
        print("\n" + "="*60)
        print("🎉 REDDIT COMMENT MODERATION PROJECT SETUP COMPLETE! 🎉")
        print("="*60)
        print()
        print("Project Structure:")
        print(f"  📁 Project Root: {self.project_root}")
        print(f"  📁 Source Code: {self.src_dir}")
        print(f"  📁 Data: {self.data_dir}")
        print(f"  📁 Models: {self.models_dir}")
        print()
        print("Next Steps:")
        print("  1. 🏃 Run the demo: python demo_reddit_moderation.py")
        print("  2. 📊 Prepare your Reddit dataset in data/raw/")
        print("  3. ⚙️  Modify config.yaml for your specific needs")
        print("  4. 🚀 Train your models and submit to Kaggle!")
        print()
        print("Useful Commands:")
        print("  • Run demo: python demo_reddit_moderation.py")
        print("  • Install deps: pip install -r requirements.txt")
        print("  • Verify setup: python project_setup.py --verify")
        print()
        print("Documentation:")
        print("  📖 Competition Guide: REDDIT_COMPETITION_GUIDE.md")
        print("  📋 Requirements: requirements.txt")
        print("  ⚙️  Configuration: config.yaml")
        print()
        print("Happy coding! 🚀")
        print("="*60)

def main():
    """
    Main function to handle command line arguments and run setup.
    """
    parser = argparse.ArgumentParser(
        description="Setup script for Reddit Comment Rule Violation Detection Project"
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Install Python dependencies from requirements.txt"
    )
    parser.add_argument(
        "--download-models", action="store_true",
        help="Download required NLP models and data"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify installation and run tests"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all setup steps (install, download, verify)"
    )
    parser.add_argument(
        "--project-root", type=str,
        help="Specify project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.install_deps, args.download_models, args.verify, args.all]):
        parser.print_help()
        return
    
    # Initialize setup
    setup = ProjectSetup(args.project_root)
    
    success = True
    
    try:
        # Run requested setup steps
        if args.all or args.install_deps:
            logger.info("Step 1: Installing dependencies...")
            if not setup.install_dependencies():
                success = False
        
        if args.all or args.download_models:
            logger.info("Step 2: Downloading NLP models...")
            if not setup.download_nlp_models():
                success = False
        
        if args.all or args.verify:
            logger.info("Step 3: Verifying installation...")
            if not setup.verify_installation():
                success = False
        
        # Create additional files
        if args.all:
            logger.info("Creating additional configuration files...")
            setup.create_sample_config()
            setup.create_gitignore()
        
        # Print summary
        if success:
            setup.print_setup_summary()
        else:
            logger.error("Setup completed with some errors. Please check the logs above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()