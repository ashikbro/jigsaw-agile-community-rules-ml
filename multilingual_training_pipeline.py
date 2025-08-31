#!/usr/bin/env python3
"""
Multilingual Training Pipeline for Content Moderation

This module provides a comprehensive multilingual training pipeline that:
1. Creates separate models for different languages
2. Uses cross-lingual embeddings (mBERT, XLM-R)
3. Implements Bengali-specific preprocessing
4. Ensures balanced datasets across languages
5. Provides robust translation validation

Author: AI Assistant
Date: 2025-01-31
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Import our custom modules
from bengali_preprocessor import BengaliPreprocessor
from bengali_dataset_generator import BengaliDatasetGenerator
from enhanced_translation_service import EnhancedTranslationService

# Deep Learning Libraries
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        XLMRobertaTokenizer, XLMRobertaModel,
        BertTokenizer, BertModel,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using classical ML only.")

# Language Processing
try:
    import langdetect
    from googletrans import Translator
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    LANG_PROCESSING_AVAILABLE = True
except ImportError:
    LANG_PROCESSING_AVAILABLE = False
    logging.warning("Language processing libraries not fully available.")

# Bengali Processing
try:
    import bnlp
    from bnlp import NLTKTokenizer
    from bnlp.corpus import stopwords as bengali_stopwords
    from bnlp.corpus import punctuations as bengali_punctuations
    BENGALI_PROCESSING_AVAILABLE = True
except ImportError:
    BENGALI_PROCESSING_AVAILABLE = False
    logging.warning("Bengali processing library (bnlp) not available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultilingualConfig:
    """Configuration for multilingual training pipeline."""
    
    # Model configurations
    classical_models: List[str] = None
    transformer_models: List[str] = None
    
    # Language configurations
    supported_languages: List[str] = None
    primary_language: str = "en"
    
    # Training configurations
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Data configurations
    min_samples_per_language: int = 100
    balance_classes: bool = True
    
    # Output configurations
    model_output_dir: str = "models/multilingual"
    results_output_dir: str = "outputs/multilingual"
    
    def __post_init__(self):
        if self.classical_models is None:
            self.classical_models = [
                "logistic_regression",
                "random_forest",
                "svm",
                "naive_bayes"
            ]
        
        if self.transformer_models is None:
            self.transformer_models = [
                "xlm-roberta-base",
                "bert-base-multilingual-cased",
                "distilbert-base-multilingual-cased"
            ]
        
        if self.supported_languages is None:
            self.supported_languages = [
                "en",  # English
                "bn",  # Bengali
                "hi",  # Hindi
                "es",  # Spanish
                "fr",  # French
                "de",  # German
                "ar",  # Arabic
                "zh",  # Chinese
            ]

class BengaliPreprocessor:
    """Bengali-specific text preprocessing."""
    
    def __init__(self):
        self.tokenizer = None
        self.stopwords = set()
        self.punctuations = set()
        
        if BENGALI_PROCESSING_AVAILABLE:
            try:
                self.tokenizer = NLTKTokenizer()
                self.stopwords = set(bengali_stopwords())
                self.punctuations = set(bengali_punctuations())
            except Exception as e:
                logger.warning(f"Failed to initialize Bengali processor: {e}")
        
        # Fallback Bengali stopwords if bnlp is not available
        if not self.stopwords:
            self.stopwords = {
                'আমি', 'তুমি', 'সে', 'আমরা', 'তোমরা', 'তারা',
                'এই', 'সেই', 'যে', 'কি', 'কে', 'কোথায়', 'কখন',
                'এবং', 'বা', 'কিন্তু', 'তবে', 'যদি', 'তাহলে',
                'না', 'নয়', 'নেই', 'আছে', 'ছিল', 'হবে', 'করা',
                'হওয়া', 'দেওয়া', 'নেওয়া', 'বলা', 'দেখা'
            }
    
    def clean_text(self, text: str) -> str:
        """Clean Bengali text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuations
        for punct in self.punctuations:
            text = text.replace(punct, ' ')
        
        # Remove English characters and numbers (optional)
        # text = re.sub(r'[a-zA-Z0-9]', ' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Bengali stopwords."""
        if not text:
            return ""
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Bengali text."""
        if self.tokenizer:
            try:
                return self.tokenizer.word_tokenize(text)
            except Exception as e:
                logger.warning(f"Bengali tokenization failed: {e}")
        
        # Fallback tokenization
        return text.split()
    
    def preprocess(self, text: str) -> str:
        """Complete Bengali preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class TranslationValidator:
    """Robust translation validation with multiple fallback services."""
    
    def __init__(self):
        self.translators = []
        
        # Primary translator: Google Translate
        try:
            self.translators.append(('google', Translator()))
        except Exception as e:
            logger.warning(f"Google Translator initialization failed: {e}")
        
        # Add more translation services here as fallbacks
        # self.translators.append(('microsoft', MicrosoftTranslator()))
        # self.translators.append(('deepl', DeepLTranslator()))
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score."""
        try:
            if LANG_PROCESSING_AVAILABLE:
                detected = langdetect.detect_langs(text)
                if detected:
                    return detected[0].lang, detected[0].prob
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return "unknown", 0.0
    
    def translate_text(self, text: str, target_lang: str = "en", source_lang: str = None) -> Optional[str]:
        """Translate text with fallback services."""
        for service_name, translator in self.translators:
            try:
                if service_name == 'google':
                    result = translator.translate(text, dest=target_lang, src=source_lang)
                    if result and result.text:
                        return result.text
            except Exception as e:
                logger.warning(f"Translation failed with {service_name}: {e}")
                continue
        
        logger.error("All translation services failed")
        return None
    
    def validate_translation(self, original: str, translated: str, threshold: float = 0.3) -> bool:
        """Validate translation quality (basic implementation)."""
        if not translated or len(translated.strip()) == 0:
            return False
        
        # Basic validation: check if translation is not just the original text
        if original.strip().lower() == translated.strip().lower():
            return False
        
        # Check minimum length ratio
        length_ratio = len(translated) / max(len(original), 1)
        if length_ratio < threshold or length_ratio > (1 / threshold):
            return False
        
        return True

class MultilingualTrainingPipeline:
    """Comprehensive multilingual training pipeline."""
    
    def __init__(self, config: MultilingualConfig):
        self.config = config
        self.bengali_preprocessor = BengaliPreprocessor()
        self.translation_validator = TranslationValidator()
        
        # Initialize Bengali dataset generator
        self.bengali_dataset_generator = BengaliDatasetGenerator(
            output_dir=str(Path(self.config.model_output_dir).parent / "bengali_generated")
        )
        
        # Initialize enhanced translation service
        self.translation_service = EnhancedTranslationService()
        
        # Create output directories
        Path(self.config.model_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.results_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize models storage
        self.trained_models = {}
        self.vectorizers = {}
        self.results = {}
    
    def load_multilingual_dataset(self, data_path: str) -> pd.DataFrame:
        """Load and prepare multilingual dataset."""
        logger.info(f"Loading dataset from {data_path}")
        
        # Load existing dataset
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Detect languages for each text
        logger.info("Detecting languages...")
        df['detected_language'] = df['comment_text'].apply(
            lambda x: self.translation_validator.detect_language(str(x))[0]
        )
        df['language_confidence'] = df['comment_text'].apply(
            lambda x: self.translation_validator.detect_language(str(x))[1]
        )
        
        return df
    
    def create_balanced_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balanced dataset across languages and classes."""
        logger.info("Creating balanced dataset...")
        
        balanced_dfs = []
        
        for lang in self.config.supported_languages:
            lang_df = df[df['detected_language'] == lang].copy()
            
            if len(lang_df) < self.config.min_samples_per_language:
                logger.warning(f"Insufficient data for language {lang}: {len(lang_df)} samples")
                continue
            
            if self.config.balance_classes:
                # Balance classes within each language
                min_class_size = lang_df['label'].value_counts().min()
                balanced_lang_df = lang_df.groupby('label').apply(
                    lambda x: x.sample(min_class_size, random_state=self.config.random_state)
                ).reset_index(drop=True)
                balanced_dfs.append(balanced_lang_df)
            else:
                balanced_dfs.append(lang_df)
        
        if not balanced_dfs:
            raise ValueError("No sufficient data for any supported language")
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Balanced dataset created with {len(balanced_df)} samples")
        
        return balanced_df
    
    def preprocess_text_by_language(self, text: str, language: str) -> str:
        """Apply language-specific preprocessing."""
        if not text or pd.isna(text):
            return ""
        
        # Check if text is a sparse matrix (this should never happen)
        if hasattr(text, 'toarray'):
            logger.error(f"ERROR: Sparse matrix passed to preprocess_text_by_language: {type(text)}")
            raise ValueError(f"Sparse matrix passed to preprocess_text_by_language: {type(text)}")
        
        text = str(text)  # Ensure it's a string
        
        if language == 'bn':  # Bengali
            return self.bengali_preprocessor.preprocess(text)
        elif language == 'en':  # English
            # Apply English preprocessing
            text = text.lower()
            # Remove punctuation, stopwords, etc.
            if LANG_PROCESSING_AVAILABLE:
                try:
                    nltk.download('stopwords', quiet=True)
                    stop_words = set(stopwords.words('english'))
                    words = text.split()
                    text = ' '.join([word for word in words if word not in stop_words])
                except:
                    pass
            return text
        else:
            # Generic preprocessing for other languages
            return text.lower().strip()
    
    def preprocess_text(self, text: str, language: str = 'en') -> str:
        """Preprocess text based on language."""
        if not text or pd.isna(text):
            return ""
        
        # Language-specific preprocessing
        if language == 'bn' or language == 'bengali':
            # Use Bengali preprocessor
            return self.bengali_preprocessor.preprocess_for_ml(
                text, 
                normalize=True, 
                clean=True, 
                remove_stops=True, 
                stem=False,  # Stemming can reduce accuracy for some models
                preserve_bengali=True
            )
        else:
            # Default preprocessing for other languages
            text = str(text).lower().strip()
            # Remove URLs, mentions, hashtags, numbers, punctuation
            import re
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)     # Remove mentions
            text = re.sub(r'#\w+', '', text)     # Remove hashtags
            text = re.sub(r'\d+', '', text)      # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        
        return text.strip()
    
    def create_cross_lingual_features(self, texts: List[str], languages: List[str]) -> np.ndarray:
        """Create cross-lingual features using multilingual embeddings."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using TF-IDF features")
            return self.create_tfidf_features(texts)
        
        try:
            # Use XLM-RoBERTa for cross-lingual embeddings
            model_name = "xlm-roberta-base"
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            model = XLMRobertaModel.from_pretrained(model_name)
            
            embeddings = []
            
            for text in texts:
                # Tokenize and encode
                inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    embeddings.append(embedding)
            
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Cross-lingual feature extraction failed: {e}")
            return self.create_tfidf_features(texts)
    
    def create_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF features as fallback."""
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,  # We handle stopwords in preprocessing
            lowercase=False   # We handle lowercasing in preprocessing
        )
        
        features = vectorizer.fit_transform(texts)
        self.vectorizers['tfidf'] = vectorizer
        
        return features.toarray()
    
    def train_classical_models(self, X: np.ndarray, y: np.ndarray, language: str) -> Dict[str, Any]:
        """Train classical ML models for a specific language."""
        logger.info(f"Training classical models for language: {language}")
        
        models = {
            'logistic_regression': LogisticRegression(random_state=self.config.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.config.random_state),
            'svm': SVC(probability=True, random_state=self.config.random_state),
            'naive_bayes': MultinomialNB()
        }
        
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        for model_name, model in models.items():
            if model_name not in self.config.classical_models:
                continue
            
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=self.config.cross_validation_folds)
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_transformer_models(self, texts: List[str], labels: List[int], language: str) -> Dict[str, Any]:
        """Train transformer models for a specific language."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, skipping transformer training")
            return {}
        
        logger.info(f"Training transformer models for language: {language}")
        
        results = {}
        
        for model_name in self.config.transformer_models:
            try:
                logger.info(f"Training {model_name}...")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=2
                )
                
                # Tokenize texts
                encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
                
                # Create dataset (simplified - in practice, use torch.utils.data.Dataset)
                # This is a placeholder for transformer training
                # Full implementation would require proper dataset creation and training loop
                
                results[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'status': 'initialized'  # Placeholder
                }
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        return results
    
    def prepare_bengali_data(self, sample_count: int = 2000) -> pd.DataFrame:
        """Prepare Bengali training data with balanced samples."""
        logger.info(f"Generating {sample_count} Bengali samples...")
        
        # Generate balanced Bengali dataset
        bengali_samples = self.bengali_dataset_generator.generate_balanced_dataset(sample_count)
        
        # Convert to DataFrame
        data = []
        for sample in bengali_samples:
            data.append({
                'comment_text': sample.text,
                'label': sample.label,
                'language': 'bengali',
                'category': sample.category,
                'subcategory': sample.subcategory,
                'severity': sample.severity
            })
        
        df = pd.DataFrame(data)
        
        # Validate dataset
        validation_report = self.bengali_dataset_generator.validate_dataset(bengali_samples)
        logger.info(f"Bengali dataset validation: {validation_report}")
        
        # Export for future use
        output_path = Path(self.config.model_output_dir).parent / "bengali_training_data.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Bengali training data saved to {output_path}")
        
        return df
    
    def train_multilingual_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Train the complete multilingual pipeline."""
        logger.info("Starting multilingual training pipeline...")
        
        # Load and prepare dataset
        df = self.load_multilingual_dataset(data_path)
        
        # Generate and add Bengali data if needed
        bengali_data = df[df['detected_language'] == 'bn']
        if len(bengali_data) < self.config.min_samples_per_language:
            logger.info("Generating additional Bengali training data...")
            bengali_df = self.prepare_bengali_data()
            bengali_df['detected_language'] = 'bn'
            bengali_df['language_confidence'] = 1.0
            df = pd.concat([df, bengali_df], ignore_index=True)
        
        balanced_df = self.create_balanced_dataset(df)
        
        # Train models for each language
        all_results = {}
        
        for language in self.config.supported_languages:
            lang_df = balanced_df[balanced_df['detected_language'] == language]
            
            if len(lang_df) < self.config.min_samples_per_language:
                logger.warning(f"Skipping {language} due to insufficient data")
                continue
            
            logger.info(f"Training models for language: {language}")
            
            # Preprocess texts
            processed_texts = [
                self.preprocess_text_by_language(text, language) 
                for text in lang_df['comment_text']
            ]
            
            # Create features
            if language == 'en' or len(processed_texts) > 1000:  # Use cross-lingual for English or large datasets
                X = self.create_cross_lingual_features(processed_texts, [language] * len(processed_texts))
            else:
                X = self.create_tfidf_features(processed_texts)
            
            y = lang_df['label'].values
            
            # Train classical models
            classical_results = self.train_classical_models(X, y, language)
            
            # Train transformer models
            transformer_results = self.train_transformer_models(processed_texts, y.tolist(), language)
            
            # Store results
            all_results[language] = {
                'classical': classical_results,
                'transformer': transformer_results,
                'data_size': len(lang_df),
                'class_distribution': lang_df['label'].value_counts().to_dict()
            }
            
            # Save models
            self.save_language_models(language, classical_results, transformer_results)
        
        # Train cross-lingual model with all data
        logger.info("Training cross-lingual model...")
        all_texts = [self.preprocess_text_by_language(text, lang) 
                    for text, lang in zip(balanced_df['comment_text'], balanced_df['detected_language'])]
        
        X_cross = self.create_cross_lingual_features(all_texts, balanced_df['detected_language'].tolist())
        y_cross = balanced_df['label'].values
        
        cross_lingual_results = self.train_classical_models(X_cross, y_cross, 'cross_lingual')
        all_results['cross_lingual'] = {
            'classical': cross_lingual_results,
            'data_size': len(balanced_df),
            'languages': balanced_df['detected_language'].value_counts().to_dict()
        }
        
        self.save_language_models('cross_lingual', cross_lingual_results, {})
        
        # Save overall results
        self.save_training_results(all_results)
        
        logger.info("Multilingual training pipeline completed!")
        return all_results
    
    def save_language_models(self, language: str, classical_results: Dict, transformer_results: Dict):
        """Save trained models for a specific language."""
        lang_dir = Path(self.config.model_output_dir) / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classical models
        for model_name, result in classical_results.items():
            model_path = lang_dir / f"{model_name}.joblib"
            import joblib
            joblib.dump(result['model'], model_path)
        
        # Save vectorizers
        if 'tfidf' in self.vectorizers:
            vectorizer_path = lang_dir / "vectorizer.joblib"
            import joblib
            joblib.dump(self.vectorizers['tfidf'], vectorizer_path)
        
        # Save transformer models (if any)
        for model_name, result in transformer_results.items():
            if 'model' in result:
                model_dir = lang_dir / model_name
                model_dir.mkdir(exist_ok=True)
                try:
                    result['model'].save_pretrained(model_dir)
                    result['tokenizer'].save_pretrained(model_dir)
                except Exception as e:
                    logger.error(f"Failed to save transformer model {model_name}: {e}")
    
    def save_training_results(self, results: Dict):
        """Save training results and metadata."""
        results_path = Path(self.config.results_output_dir) / "training_results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for lang, lang_results in results.items():
            json_results[lang] = {
                'data_size': lang_results.get('data_size', 0),
                'classical_models': {},
                'transformer_models': list(lang_results.get('transformer', {}).keys())
            }
            
            # Extract classical model metrics
            for model_name, model_result in lang_results.get('classical', {}).items():
                json_results[lang]['classical_models'][model_name] = {
                    'accuracy': model_result['accuracy'],
                    'cv_mean': model_result['cv_mean'],
                    'cv_std': model_result['cv_std']
                }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training results saved to {results_path}")

def main():
    """Main function to run the multilingual training pipeline."""
    # Configuration
    config = MultilingualConfig(
        supported_languages=["en", "bn", "hi", "es"],
        min_samples_per_language=50,
        model_output_dir="models/multilingual",
        results_output_dir="outputs/multilingual"
    )
    
    # Initialize pipeline
    pipeline = MultilingualTrainingPipeline(config)
    
    # Train models
    try:
        results = pipeline.train_multilingual_pipeline("data/raw/sample_data.csv")
        print("\nTraining completed successfully!")
        print(f"Results saved to: {config.results_output_dir}")
        print(f"Models saved to: {config.model_output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()