#!/usr/bin/env python3
"""
Multilingual Content Moderation Model Training Script

This script trains a comprehensive multilingual content moderation system
with enhanced Bengali support, balanced datasets, and robust translation.

Features:
1. Multilingual training pipeline
2. Bengali-specific preprocessing and dataset generation
3. Enhanced translation validation
4. Balanced dataset creation
5. Cross-lingual model evaluation
6. Model performance monitoring

Author: AI Assistant
Date: 2025-01-31
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our custom modules
from multilingual_training_pipeline import MultilingualTrainingPipeline
from bengali_preprocessor import BengaliPreprocessor
from bengali_dataset_generator import BengaliDatasetGenerator
from enhanced_translation_service import EnhancedTranslationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multilingual_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultilingualModelTrainer:
    """Comprehensive multilingual model trainer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the multilingual model trainer."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        from multilingual_training_pipeline import MultilingualConfig
        
        # Create multilingual config
        ml_config = MultilingualConfig(
            model_output_dir=self.config.get('models_path', 'models'),
            results_output_dir=self.config.get('results_path', 'outputs'),
            supported_languages=['en', 'bn'],
            primary_language='en',
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            cross_validation_folds=self.config.get('cv_folds', 5),
            min_samples_per_language=self.config.get('min_samples_per_language', 500),
            balance_classes=True
        )
        
        self.pipeline = MultilingualTrainingPipeline(ml_config)
        
        self.bengali_preprocessor = BengaliPreprocessor()
        self.bengali_generator = BengaliDatasetGenerator(
            output_dir=self.config.get('bengali_data_path', 'data/bengali_generated')
        )
        self.translation_service = EnhancedTranslationService()
        
        # Training configuration
        self.training_config = {
            'bengali_samples': self.config.get('bengali_samples', 2000),
            'test_size': self.config.get('test_size', 0.2),
            'random_state': self.config.get('random_state', 42),
            'cross_validation_folds': self.config.get('cv_folds', 5),
            'models_to_train': self.config.get('models_to_train', [
                'logistic_regression', 'random_forest', 'svm', 'naive_bayes'
            ])
        }
        
        logger.info("Multilingual model trainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'data_path': 'data',
            'models_path': 'models',
            'bengali_data_path': 'data/bengali_generated',
            'bengali_samples': 2000,
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'models_to_train': ['logistic_regression', 'random_forest', 'svm', 'naive_bayes'],
            'min_samples_per_language': 500,
            'balance_threshold': 0.1
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare comprehensive training dataset."""
        logger.info("Preparing training data...")
        
        # Load existing data if available
        existing_data_path = Path(self.config['data_path']) / 'sample_data.csv'
        if existing_data_path.exists():
            logger.info(f"Loading existing data from {existing_data_path}")
            df = pd.read_csv(existing_data_path)
            
            # Ensure labels are integers
            if 'label' in df.columns:
                df['label'] = pd.to_numeric(df['label'], errors='coerce').astype(int)
            
            # Add language detection if not present
            if 'language' not in df.columns:
                df['language'] = 'en'  # Assume English for existing data
                df['detected_language'] = 'en'
                df['language_confidence'] = 1.0
        else:
            # Create empty DataFrame with required columns
            df = pd.DataFrame(columns=[
                'comment_text', 'label', 'language', 'detected_language', 'language_confidence'
            ])
        
        # Generate Bengali training data
        logger.info("Generating Bengali training data...")
        bengali_samples = self.bengali_generator.generate_balanced_dataset(
            self.training_config['bengali_samples']
        )
        
        # Convert Bengali samples to DataFrame
        bengali_data = []
        for sample in bengali_samples:
            bengali_data.append({
                'comment_text': sample.text,
                'label': int(sample.label),  # Ensure label is integer
                'language': 'bn',
                'detected_language': 'bn',
                'language_confidence': 1.0,
                'category': sample.category,
                'subcategory': sample.subcategory,
                'severity': sample.severity
            })
        
        bengali_df = pd.DataFrame(bengali_data)
        
        # Combine datasets
        combined_df = pd.concat([df, bengali_df], ignore_index=True)
        
        # Ensure all labels are integers
        combined_df['label'] = pd.to_numeric(combined_df['label'], errors='coerce').astype(int)
        
        # Remove any rows with invalid labels
        combined_df = combined_df.dropna(subset=['label']).reset_index(drop=True)
        
        # Validate and balance dataset
        validation_report = self._validate_dataset(combined_df)
        logger.info(f"Dataset validation: {validation_report}")
        
        # Save combined dataset
        output_path = Path(self.config['data_path']) / 'multilingual_training_data.csv'
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Combined training data saved to {output_path}")
        
        return combined_df
    
    def _validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset balance and quality."""
        total_samples = len(df)
        
        # Language distribution
        lang_dist = df['language'].value_counts().to_dict()
        
        # Label distribution
        label_dist = df['label'].value_counts().to_dict()
        
        # Balance by language
        balance_by_lang = {}
        for lang in df['language'].unique():
            lang_data = df[df['language'] == lang]
            if len(lang_data) > 0:
                violation_ratio = len(lang_data[lang_data['label'] == 1]) / len(lang_data)
                balance_by_lang[lang] = {
                    'total': len(lang_data),
                    'violations': len(lang_data[lang_data['label'] == 1]),
                    'safe': len(lang_data[lang_data['label'] == 0]),
                    'violation_ratio': violation_ratio
                }
        
        return {
            'total_samples': total_samples,
            'language_distribution': lang_dist,
            'label_distribution': label_dist,
            'balance_by_language': balance_by_lang
        }
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train multilingual models."""
        logger.info("Starting model training...")
        
        results = {
            'training_start': datetime.now().isoformat(),
            'dataset_info': self._validate_dataset(df),
            'model_results': {},
            'best_model': None,
            'best_accuracy': 0.0
        }
        
        # Preprocess text data
        logger.info("Preprocessing text data...")
        df['processed_text'] = df.apply(
            lambda row: self._preprocess_text(row['comment_text'], row['language']), 
            axis=1
        )
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.training_config['test_size'],
            random_state=self.training_config['random_state'],
            stratify=y
        )
        
        # Train models
        for model_name in self.training_config['models_to_train']:
            logger.info(f"Training {model_name}...")
            
            try:
                model_result = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                results['model_results'][model_name] = model_result
                
                # Track best model
                if model_result['test_accuracy'] > results['best_accuracy']:
                    results['best_accuracy'] = model_result['test_accuracy']
                    results['best_model'] = model_name
                
                logger.info(f"{model_name} - Accuracy: {model_result['test_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results['model_results'][model_name] = {'error': str(e)}
        
        results['training_end'] = datetime.now().isoformat()
        
        # Save results
        results_path = Path(self.config['models_path']) / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training completed. Best model: {results['best_model']} (Accuracy: {results['best_accuracy']:.4f})")
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text based on language."""
        if pd.isna(text) or not text:
            return ""
        
        if language == 'bn' or language == 'bengali':
            return self.bengali_preprocessor.preprocess_for_ml(
                text,
                normalize=True,
                clean=True,
                remove_stops=True,
                stem=False,
                preserve_bengali=True
            )
        else:
            # Default preprocessing for other languages
            import re
            text = str(text).lower().strip()
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)     # Remove mentions
            text = re.sub(r'#\w+', '', text)     # Remove hashtags
            text = re.sub(r'\d+', '', text)      # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
            return text.strip()
    
    def _train_single_model(self, model_name: str, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train a single model and return results."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        from sklearn.model_selection import cross_val_score
        import joblib
        
        # Model configurations
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=None  # We handle stop words in preprocessing
            )),
            ('classifier', models[model_name])
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps['classifier'], 'predict_proba') else None
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=self.training_config['cross_validation_folds'],
            scoring='accuracy'
        )
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Save model
        model_path = Path(self.config['models_path']) / f'{model_name}_multilingual.joblib'
        joblib.dump(pipeline, model_path)
        
        # Save vectorizer separately for compatibility
        vectorizer_path = Path(self.config['models_path']) / f'{model_name}_vectorizer.joblib'
        joblib.dump(pipeline.named_steps['tfidf'], vectorizer_path)
        
        return {
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'model_path': str(model_path),
            'vectorizer_path': str(vectorizer_path),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def test_multilingual_prediction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Test the trained models with multilingual examples."""
        logger.info("Testing multilingual predictions...")
        
        # Test cases
        test_cases = [
            # Bengali violation
            {
                'text': 'তুমি একটা বোকা এবং আমি তোমাকে মারব',
                'language': 'bn',
                'expected_label': 1,
                'description': 'Bengali violation (threat)'
            },
            # Bengali safe
            {
                'text': 'আজকের আবহাওয়া খুব ভালো, আমি খুশি',
                'language': 'bn',
                'expected_label': 0,
                'description': 'Bengali safe (weather comment)'
            },
            # English violation
            {
                'text': 'You are stupid and I hate you',
                'language': 'en',
                'expected_label': 1,
                'description': 'English violation (insult)'
            },
            # English safe
            {
                'text': 'The weather is nice today',
                'language': 'en',
                'expected_label': 0,
                'description': 'English safe (weather comment)'
            }
        ]
        
        best_model_name = results['best_model']
        if not best_model_name:
            logger.error("No best model found")
            return {'error': 'No best model available'}
        
        # Load best model
        model_path = Path(self.config['models_path']) / f'{best_model_name}_multilingual.joblib'
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return {'error': f'Model file not found: {model_path}'}
        
        import joblib
        model = joblib.load(model_path)
        
        test_results = {
            'model_used': best_model_name,
            'test_cases': []
        }
        
        for test_case in test_cases:
            try:
                # Preprocess text
                processed_text = self._preprocess_text(test_case['text'], test_case['language'])
                
                # Make prediction
                prediction = model.predict([processed_text])[0]
                prediction_proba = model.predict_proba([processed_text])[0] if hasattr(model, 'predict_proba') else None
                
                # Calculate confidence
                confidence = max(prediction_proba) if prediction_proba is not None else 0.5
                
                test_result = {
                    'original_text': test_case['text'],
                    'processed_text': processed_text,
                    'language': test_case['language'],
                    'description': test_case['description'],
                    'expected_label': int(test_case['expected_label']),
                    'predicted_label': int(prediction),
                    'confidence': float(confidence),
                    'correct': bool(prediction == test_case['expected_label'])
                }
                
                if prediction_proba is not None:
                    test_result['probabilities'] = {
                        'safe': float(prediction_proba[0]),
                        'violation': float(prediction_proba[1])
                    }
                
                test_results['test_cases'].append(test_result)
                
                status = "✓" if test_result['correct'] else "✗"
                logger.info(f"{status} {test_case['description']}: {prediction} (confidence: {confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to test case '{test_case['description']}': {e}")
                test_results['test_cases'].append({
                    'description': test_case['description'],
                    'error': str(e)
                })
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for case in test_results['test_cases'] if case.get('correct', False))
        total_predictions = len([case for case in test_results['test_cases'] if 'correct' in case])
        
        if total_predictions > 0:
            test_results['accuracy'] = correct_predictions / total_predictions
            logger.info(f"Multilingual test accuracy: {test_results['accuracy']:.3f} ({correct_predictions}/{total_predictions})")
        
        return test_results
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete multilingual training pipeline."""
        logger.info("Starting complete multilingual training pipeline...")
        
        try:
            # Step 1: Prepare training data
            df = self.prepare_training_data()
            
            # Step 2: Train models
            training_results = self.train_models(df)
            
            # Step 3: Test multilingual predictions
            test_results = self.test_multilingual_prediction(training_results)
            
            # Combine results
            complete_results = {
                'pipeline_start': datetime.now().isoformat(),
                'training_results': training_results,
                'test_results': test_results,
                'pipeline_end': datetime.now().isoformat()
            }
            
            # Save complete results
            results_path = Path(self.config['models_path']) / 'complete_training_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Complete training pipeline finished. Results saved to {results_path}")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main training function."""
    print("Multilingual Content Moderation Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MultilingualModelTrainer()
    
    # Run complete training
    results = trainer.run_complete_training()
    
    # Print summary
    print("\nTraining Summary:")
    print("-" * 30)
    
    if 'training_results' in results:
        training_results = results['training_results']
        print(f"Best Model: {training_results.get('best_model', 'N/A')}")
        print(f"Best Accuracy: {training_results.get('best_accuracy', 0):.4f}")
        
        dataset_info = training_results.get('dataset_info', {})
        print(f"Total Training Samples: {dataset_info.get('total_samples', 0)}")
        
        lang_dist = dataset_info.get('language_distribution', {})
        print("Language Distribution:")
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count}")
    
    if 'test_results' in results:
        test_results = results['test_results']
        test_accuracy = test_results.get('accuracy', 0)
        print(f"\nMultilingual Test Accuracy: {test_accuracy:.3f}")
        
        print("\nTest Cases:")
        for case in test_results.get('test_cases', []):
            if 'correct' in case:
                status = "✓" if case['correct'] else "✗"
                print(f"  {status} {case['description']}: {case['predicted_label']} (conf: {case.get('confidence', 0):.3f})")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()