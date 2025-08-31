#!/usr/bin/env python3
"""
Complete Training Pipeline for Binary Text Classification
Integrates preprocessing, feature extraction, and classical ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from text_preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from classical_models import ClassicalMLPipeline

class MLTrainingPipeline:
    """
    Complete pipeline for training classical ML models on text data
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_extractor = None
        self.ml_pipeline = None
        self.label_mapping = None
        self.training_history = {}
        
    def load_data(self, 
                  train_path: str = None,
                  text_column: str = 'comment_text',
                  label_column: str = 'target',
                  sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load training data from file or create sample data
        """
        if train_path and os.path.exists(train_path):
            print(f"Loading data from {train_path}")
            df = pd.read_csv(train_path)
            
            if text_column not in df.columns or label_column not in df.columns:
                raise ValueError(f"Required columns {text_column}, {label_column} not found")
            
            # Sample data if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=self.random_state)
                print(f"Sampled {sample_size} rows from dataset")
                
        else:
            print("No data file found. Creating sample dataset...")
            df = self._create_sample_data()
            text_column = 'comment_text'
            label_column = 'target'
        
        # Ensure we have the right columns
        if text_column != 'comment_text':
            df = df.rename(columns={text_column: 'comment_text'})
        if label_column != 'target':
            df = df.rename(columns={label_column: 'target'})
        
        # Clean and validate data
        df = df.dropna(subset=['comment_text', 'target'])
        df['comment_text'] = df['comment_text'].astype(str)
        
        # Ensure binary labels
        unique_labels = df['target'].unique()
        if len(unique_labels) != 2:
            raise ValueError(f"Expected binary classification, found {len(unique_labels)} classes")
        
        # Map labels to 0/1 if needed
        if not all(label in [0, 1] for label in unique_labels):
            self.label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            df['target'] = df['target'].map(self.label_mapping)
            print(f"Mapped labels: {self.label_mapping}")
        
        print(f"Loaded {len(df)} samples")
        print(f"Class distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample training data for demonstration
        """
        # Violation examples
        violations = [
            "You're such an idiot, go kill yourself",
            "I hate you so much, you should die",
            "Stupid moron, nobody likes you",
            "You're worthless trash, get lost",
            "I hope you get cancer and suffer",
            "You're a pathetic loser, kill yourself",
            "I wish you were dead, you piece of garbage",
            "You're so ugly, no wonder nobody loves you",
            "Go jump off a bridge, you waste of space",
            "I hope your family dies in a car crash",
            "You're a disgusting pig, go away",
            "I want to punch you in the face repeatedly",
            "You should be banned from existing",
            "Your mother should have aborted you",
            "I hope you get hit by a truck",
            "You're a complete failure at life",
            "Nobody cares about your stupid opinion",
            "You're mentally retarded and should be locked up",
            "I hope you suffer for the rest of your life",
            "You're a waste of oxygen, please disappear",
            "Shut up you annoying piece of crap",
            "I hate everything about you, you're disgusting",
            "You deserve to be alone forever",
            "Your existence is a mistake",
            "I hope you get what's coming to you",
            "You're the worst person I've ever met",
            "I wish I could make you disappear",
            "You're so stupid it hurts to read your posts",
            "Go crawl back under the rock you came from",
            "You're a toxic waste of human flesh"
        ]
        
        # Safe examples
        safe = [
            "I disagree with your opinion but respect your right to have it",
            "That's an interesting perspective, thanks for sharing",
            "I would like to kill some time before the movie starts",
            "This game is killing my battery life",
            "The suspense is killing me, I can't wait to find out",
            "I need to kill this process on my computer",
            "That joke absolutely killed at the comedy show",
            "I'm trying to kill two birds with one stone",
            "The heat is killing me today, it's so hot",
            "I love this song, it never gets old",
            "Thanks for the helpful advice, I appreciate it",
            "Great job on your project, very impressive work",
            "I'm looking forward to the weekend",
            "This is a really good discussion, thanks everyone",
            "I learned something new today, that's awesome",
            "Hope everyone has a great day",
            "This community is really supportive and friendly",
            "I'm excited to try this new recipe",
            "The weather is beautiful today",
            "I enjoy reading different perspectives on this topic",
            "Thanks for taking the time to explain that",
            "I'm grateful for all the help I've received",
            "This is exactly what I was looking for",
            "I appreciate your patience with my questions",
            "Looking forward to hearing more about this",
            "This is a really well-written post",
            "I'm happy to help if you need anything",
            "Great question, I'm curious about this too",
            "Thanks for sharing your experience",
            "I hope this information is helpful"
        ]
        
        # Create DataFrame
        data = []
        for text in violations:
            data.append({'comment_text': text, 'target': 1})
        for text in safe:
            data.append({'comment_text': text, 'target': 0})
        
        df = pd.DataFrame(data)
        return df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def split_data(self, 
                   df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=df['target']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_val['target']
        )
        
        print(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(f"Train class distribution: {train['target'].value_counts().to_dict()}")
        print(f"Val class distribution: {val['target'].value_counts().to_dict()}")
        print(f"Test class distribution: {test['target'].value_counts().to_dict()}")
        
        return train, val, test
    
    def setup_preprocessing(self, 
                           remove_stopwords: bool = True,
                           use_lemmatization: bool = True) -> TextPreprocessor:
        """
        Setup text preprocessing
        """
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            use_lemmatization=use_lemmatization
        )
        return self.preprocessor
    
    def setup_feature_extraction(self,
                               max_features_word: int = 10000,
                               max_features_char: int = 5000,
                               word_ngram_range: Tuple[int, int] = (1, 3),
                               char_ngram_range: Tuple[int, int] = (3, 6),
                               feature_selection_k: Optional[int] = None) -> FeatureExtractor:
        """
        Setup feature extraction
        """
        self.feature_extractor = FeatureExtractor(
            max_features_word=max_features_word,
            max_features_char=max_features_char,
            word_ngram_range=word_ngram_range,
            char_ngram_range=char_ngram_range,
            feature_selection_k=feature_selection_k
        )
        return self.feature_extractor
    
    def train_pipeline(self,
                      df: pd.DataFrame,
                      models_to_train: Optional[List[str]] = None,
                      use_grid_search: bool = True,
                      preprocessing_config: Optional[Dict] = None,
                      feature_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train the complete pipeline
        """
        print("=" * 60)
        print("STARTING ML TRAINING PIPELINE")
        print("=" * 60)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Setup preprocessing
        if preprocessing_config is None:
            preprocessing_config = {}
        self.setup_preprocessing(**preprocessing_config)
        
        # Setup feature extraction
        if feature_config is None:
            feature_config = {}
        self.setup_feature_extraction(**feature_config)
        
        # Preprocess text
        print("\nPreprocessing text...")
        train_texts = [self.preprocessor.preprocess_text(text) for text in train_df['comment_text']]
        val_texts = [self.preprocessor.preprocess_text(text) for text in val_df['comment_text']]
        test_texts = [self.preprocessor.preprocess_text(text) for text in test_df['comment_text']]
        
        # Extract features
        print("Extracting features...")
        X_train = self.feature_extractor.fit_transform(train_texts, train_df['target'])
        X_val = self.feature_extractor.transform(val_texts)
        X_test = self.feature_extractor.transform(test_texts)
        
        print(f"Feature matrix shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Setup ML pipeline
        self.ml_pipeline = ClassicalMLPipeline(random_state=self.random_state)
        
        # Train models
        print("\nTraining models...")
        results = self.ml_pipeline.train_all_models(
            X_train, train_df['target'].values,
            X_val, val_df['target'].values,
            models_to_train=models_to_train,
            use_grid_search=use_grid_search
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self._evaluate_on_test_set(X_test, test_df['target'].values)
        
        # Store training history
        self.training_history = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'feature_shape': X_train.shape,
            'preprocessing_config': preprocessing_config,
            'feature_config': feature_config,
            'validation_results': results,
            'test_results': test_results
        }
        
        return self.training_history
    
    def _evaluate_on_test_set(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set
        """
        from sklearn.metrics import accuracy_score
        test_results = {}
        
        for model_name in self.ml_pipeline.models.keys():
            try:
                y_pred = self.ml_pipeline.predict(model_name, X_test)
                y_pred_proba = None
                
                try:
                    y_pred_proba = self.ml_pipeline.predict_proba(model_name, X_test)
                except:
                    pass
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                test_results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'auc_score': auc
                }
                
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                print(f"{model_name} Test - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_str}")
                
            except Exception as e:
                print(f"Error evaluating {model_name} on test set: {str(e)}")
                continue
        
        return test_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all models including test results
        """
        if not self.ml_pipeline or not self.training_history:
            raise ValueError("Pipeline not trained yet")
        
        comparison_data = []
        val_results = self.training_history['validation_results']
        test_results = self.training_history['test_results']
        
        for model_name in val_results.keys():
            val_result = val_results[model_name]
            test_result = test_results.get(model_name, {})
            
            comparison_data.append({
                'Model': model_name,
                'Val F1': val_result['f1_score'],
                'Val AUC': val_result['auc_score'] if val_result['auc_score'] else 'N/A',
                'Test F1': test_result.get('f1_score', 'N/A'),
                'Test AUC': test_result.get('auc_score', 'N/A'),
                'Best Params': str(val_result['best_params'])
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Test F1', ascending=False)
    
    def predict_text(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """
        Predict on new text
        """
        if not self.preprocessor or not self.feature_extractor or not self.ml_pipeline:
            raise ValueError("Pipeline not trained yet")
        
        # Use best model if not specified
        if model_name is None:
            model_name, _ = self.ml_pipeline.get_best_model()
        
        # Preprocess and extract features
        processed_text = self.preprocessor.preprocess_text(text)
        features = self.feature_extractor.transform([processed_text])
        
        # Make prediction
        prediction = self.ml_pipeline.predict(model_name, features)[0]
        
        try:
            probability = self.ml_pipeline.predict_proba(model_name, features)[0]
        except:
            probability = None
        
        return {
            'text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'model_used': model_name
        }
    
    def save_pipeline(self, directory: str) -> None:
        """
        Save the complete pipeline
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save components
        if self.preprocessor:
            joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.joblib'))
        
        if self.feature_extractor:
            self.feature_extractor.save(os.path.join(directory, 'feature_extractor.joblib'))
        
        if self.ml_pipeline:
            self.ml_pipeline.save_models(os.path.join(directory, 'models'))
        
        # Save training history and metadata
        metadata = {
            'training_history': self.training_history,
            'label_mapping': self.label_mapping,
            'random_state': self.random_state
        }
        joblib.dump(metadata, os.path.join(directory, 'metadata.joblib'))
        
        print(f"Pipeline saved to {directory}")
    
    def load_pipeline(self, directory: str) -> None:
        """
        Load a previously trained pipeline
        """
        # Load metadata
        metadata = joblib.load(os.path.join(directory, 'metadata.joblib'))
        self.training_history = metadata['training_history']
        self.label_mapping = metadata['label_mapping']
        self.random_state = metadata['random_state']
        
        # Load components
        self.preprocessor = joblib.load(os.path.join(directory, 'preprocessor.joblib'))
        
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load(os.path.join(directory, 'feature_extractor.joblib'))
        
        self.ml_pipeline = ClassicalMLPipeline(random_state=self.random_state)
        self.ml_pipeline.load_models(os.path.join(directory, 'models'))
        
        print(f"Pipeline loaded from {directory}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Load data (will create sample data if no file found)
    df = pipeline.load_data()
    
    # Configure training
    preprocessing_config = {
        'remove_stopwords': True,
        'use_lemmatization': True
    }
    
    feature_config = {
        'feature_type': 'tfidf_word',
        'ngram_range': (1, 3),
        'max_features': 5000,
        'use_feature_selection': True,
        'k_best': 3000
    }
    
    # Train pipeline
    results = pipeline.train_pipeline(
        df,
        models_to_train=['logistic_regression', 'multinomial_nb', 'svm_linear'],
        use_grid_search=True,
        preprocessing_config=preprocessing_config,
        feature_config=feature_config
    )
    
    # Get model comparison
    comparison = pipeline.get_model_comparison()
    print("\nFinal Model Comparison:")
    print(comparison)
    
    # Test prediction
    test_text = "You're such an idiot, go kill yourself"
    prediction = pipeline.predict_text(test_text)
    print(f"\nPrediction for '{test_text}':")
    print(f"Prediction: {prediction['prediction']} (Probability: {prediction['probability']:.4f})")
    
    # Save pipeline
    pipeline.save_pipeline('trained_pipeline')