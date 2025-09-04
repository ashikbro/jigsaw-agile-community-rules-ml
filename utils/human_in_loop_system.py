#!/usr/bin/env python3
"""
Human-in-the-Loop ML Moderation System

This module implements a self-improving explainable moderation system with human feedback loops,
including feedback storage, incremental learning, and adaptive model updating.

Novelty: A self-improving explainable moderation system with human feedback loops
using classical ML models with incremental learning capabilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, deque
import warnings
import logging
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackStorage:
    """
    Manages storage and retrieval of human feedback for model improvement.
    """
    
    def __init__(self, db_path: str = "feedback_storage.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Initialize SQLite database for feedback storage.
        """
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                original_prediction INTEGER NOT NULL,
                original_confidence REAL NOT NULL,
                human_label INTEGER NOT NULL,
                moderator_id TEXT,
                feedback_type TEXT,
                explanation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                language TEXT,
                features_hash TEXT
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS model_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                update_type TEXT NOT NULL,
                feedback_count INTEGER NOT NULL,
                accuracy_before REAL,
                accuracy_after REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                update_details TEXT
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS moderator_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                moderator_id TEXT NOT NULL,
                total_feedback INTEGER DEFAULT 0,
                agreement_rate REAL DEFAULT 0.0,
                avg_response_time REAL DEFAULT 0.0,
                expertise_score REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        logger.info(f"Feedback database initialized at {self.db_path}")
    
    def store_feedback(self, text: str, original_prediction: int, original_confidence: float,
                      human_label: int, moderator_id: str = None, feedback_type: str = "correction",
                      explanation: str = None, model_version: str = None, language: str = "en",
                      features_hash: str = None) -> int:
        """
        Store human feedback in the database.
        
        Args:
            text: Original text that was moderated
            original_prediction: Model's original prediction (0 or 1)
            original_confidence: Model's confidence score
            human_label: Human moderator's label (0 or 1)
            moderator_id: ID of the human moderator
            feedback_type: Type of feedback ('correction', 'confirmation', 'uncertain')
            explanation: Optional explanation from moderator
            model_version: Version of the model that made the prediction
            language: Language of the text
            features_hash: Hash of the feature vector for deduplication
            
        Returns:
            ID of the stored feedback record
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO feedback (
                text, original_prediction, original_confidence, human_label,
                moderator_id, feedback_type, explanation, model_version,
                language, features_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            text, original_prediction, original_confidence, human_label,
            moderator_id, feedback_type, explanation, model_version,
            language, features_hash
        ))
        
        feedback_id = cursor.lastrowid
        self.connection.commit()
        
        # Update moderator performance
        if moderator_id:
            self._update_moderator_performance(moderator_id, original_prediction, human_label)
        
        logger.info(f"Stored feedback with ID {feedback_id}")
        return feedback_id
    
    def get_recent_feedback(self, hours: int = 24, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve recent feedback for model updating.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of records to retrieve
            
        Returns:
            List of feedback records
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
            LIMIT ?
        '''.format(hours), (limit,))
        
        columns = [description[0] for description in cursor.description]
        feedback_records = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            feedback_records.append(record)
        
        return feedback_records
    
    def get_feedback_by_type(self, feedback_type: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve feedback by type.
        
        Args:
            feedback_type: Type of feedback to retrieve
            limit: Maximum number of records
            
        Returns:
            List of feedback records
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE feedback_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (feedback_type, limit))
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_disagreement_cases(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve cases where model and human disagreed.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of disagreement cases
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE original_prediction != human_label
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _update_moderator_performance(self, moderator_id: str, original_prediction: int, human_label: int):
        """
        Update moderator performance metrics.
        
        Args:
            moderator_id: ID of the moderator
            original_prediction: Model's prediction
            human_label: Human's label
        """
        cursor = self.connection.cursor()
        
        # Check if moderator exists
        cursor.execute('SELECT * FROM moderator_performance WHERE moderator_id = ?', (moderator_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            total_feedback = existing[2] + 1
            current_agreement = existing[3]
            
            # Calculate new agreement rate
            agreement = 1 if original_prediction == human_label else 0
            new_agreement_rate = ((current_agreement * (total_feedback - 1)) + agreement) / total_feedback
            
            cursor.execute('''
                UPDATE moderator_performance 
                SET total_feedback = ?, agreement_rate = ?, last_updated = CURRENT_TIMESTAMP
                WHERE moderator_id = ?
            ''', (total_feedback, new_agreement_rate, moderator_id))
        else:
            # Create new record
            agreement_rate = 1.0 if original_prediction == human_label else 0.0
            cursor.execute('''
                INSERT INTO moderator_performance (moderator_id, total_feedback, agreement_rate)
                VALUES (?, 1, ?)
            ''', (moderator_id, agreement_rate))
        
        self.connection.commit()
    
    def get_moderator_stats(self, moderator_id: str = None) -> Dict[str, Any]:
        """
        Get moderator performance statistics.
        
        Args:
            moderator_id: Specific moderator ID (None for all moderators)
            
        Returns:
            Moderator statistics
        """
        cursor = self.connection.cursor()
        
        if moderator_id:
            cursor.execute('SELECT * FROM moderator_performance WHERE moderator_id = ?', (moderator_id,))
            result = cursor.fetchone()
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return {}
        else:
            cursor.execute('SELECT * FROM moderator_performance ORDER BY total_feedback DESC')
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def log_model_update(self, model_version: str, update_type: str, feedback_count: int,
                        accuracy_before: float = None, accuracy_after: float = None,
                        update_details: str = None):
        """
        Log model update information.
        
        Args:
            model_version: Version identifier of the model
            update_type: Type of update ('incremental', 'retrain', 'ensemble_weight')
            feedback_count: Number of feedback samples used
            accuracy_before: Accuracy before update
            accuracy_after: Accuracy after update
            update_details: Additional details about the update
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO model_updates (
                model_version, update_type, feedback_count, accuracy_before,
                accuracy_after, update_details
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_version, update_type, feedback_count, accuracy_before,
            accuracy_after, update_details
        ))
        self.connection.commit()
    
    def close(self):
        """
        Close database connection.
        """
        if self.connection:
            self.connection.close()

class IncrementalLearner:
    """
    Implements incremental learning for various ML models.
    """
    
    def __init__(self, model_type: str = "sgd", vectorizer: TfidfVectorizer = None):
        self.model_type = model_type
        self.model = None
        self.vectorizer = vectorizer or TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.is_fitted = False
        self.update_count = 0
        self.performance_history = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the incremental learning model.
        """
        if self.model_type == "sgd":
            self.model = SGDClassifier(
                loss='log_loss',  # For probability estimates
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
        elif self.model_type == "passive_aggressive":
            self.model = PassiveAggressiveClassifier(
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == "multinomial_nb":
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def initial_fit(self, texts: List[str], labels: List[int]):
        """
        Perform initial training of the model.
        
        Args:
            texts: Training texts
            labels: Training labels
        """
        # Fit vectorizer and transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Fit model
        self.model.fit(X, labels)
        self.is_fitted = True
        
        # Record initial performance
        predictions = self.model.predict(X)
        accuracy = accuracy_score(labels, predictions)
        self.performance_history.append({
            'update_count': 0,
            'accuracy': accuracy,
            'sample_count': len(texts),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Initial model training completed. Accuracy: {accuracy:.3f}")
    
    def partial_fit(self, texts: List[str], labels: List[int], classes: List[int] = None):
        """
        Perform incremental learning with new data.
        
        Args:
            texts: New training texts
            labels: New training labels
            classes: All possible classes (for first partial_fit)
        """
        if not self.is_fitted:
            raise ValueError("Model must be initially fitted before partial_fit")
        
        # Transform texts using existing vectorizer
        X = self.vectorizer.transform(texts)
        
        # Perform partial fit
        if hasattr(self.model, 'partial_fit'):
            if classes is None:
                classes = [0, 1]  # Binary classification
            self.model.partial_fit(X, labels, classes=classes)
        else:
            # For models without partial_fit, retrain with combined data
            logger.warning(f"Model {self.model_type} doesn't support partial_fit. Consider using SGD or MultinomialNB.")
            return False
        
        self.update_count += 1
        
        # Record performance
        predictions = self.model.predict(X)
        accuracy = accuracy_score(labels, predictions)
        self.performance_history.append({
            'update_count': self.update_count,
            'accuracy': accuracy,
            'sample_count': len(texts),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Incremental update {self.update_count} completed. Accuracy on new data: {accuracy:.3f}")
        return True
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on texts.
        
        Args:
            texts: Input texts
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision function
            decision_scores = self.model.decision_function(X)
            probabilities = np.column_stack([1 - decision_scores, decision_scores])
        
        return predictions, probabilities
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """
        Get model performance history.
        
        Returns:
            List of performance records
        """
        return self.performance_history.copy()
    
    def save_model(self, filepath: str):
        """
        Save the model and vectorizer.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'update_count': self.update_count,
            'performance_history': self.performance_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model and vectorizer.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']
        self.update_count = model_data['update_count']
        self.performance_history = model_data['performance_history']
        
        logger.info(f"Model loaded from {filepath}")

class AdaptiveEnsemble:
    """
    Manages an ensemble of models with adaptive weighting based on human feedback.
    """
    
    def __init__(self, model_types: List[str] = None):
        self.model_types = model_types or ["sgd", "passive_aggressive", "multinomial_nb"]
        self.models = {}
        self.model_weights = {}
        self.model_performance = {}
        self.feedback_weights = deque(maxlen=1000)  # Recent feedback for weight adjustment
        
        # Initialize models
        for model_type in self.model_types:
            self.models[model_type] = IncrementalLearner(model_type)
            self.model_weights[model_type] = 1.0 / len(self.model_types)  # Equal initial weights
            self.model_performance[model_type] = []
    
    def initial_fit(self, texts: List[str], labels: List[int]):
        """
        Perform initial training of all models in the ensemble.
        
        Args:
            texts: Training texts
            labels: Training labels
        """
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model...")
            model.initial_fit(texts, labels)
            
            # Evaluate initial performance
            predictions, _ = model.predict(texts)
            accuracy = accuracy_score(labels, predictions)
            self.model_performance[model_type].append(accuracy)
        
        logger.info("Ensemble initial training completed")
    
    def update_with_feedback(self, texts: List[str], labels: List[int]):
        """
        Update all models with new feedback data.
        
        Args:
            texts: Feedback texts
            labels: Feedback labels
        """
        model_accuracies = {}
        
        for model_type, model in self.models.items():
            # Update model
            success = model.partial_fit(texts, labels)
            
            if success:
                # Evaluate performance on feedback data
                predictions, _ = model.predict(texts)
                accuracy = accuracy_score(labels, predictions)
                model_accuracies[model_type] = accuracy
                self.model_performance[model_type].append(accuracy)
        
        # Update ensemble weights based on recent performance
        self._update_ensemble_weights(model_accuracies)
        
        # Store feedback for weight adjustment
        self.feedback_weights.extend([(text, label) for text, label in zip(texts, labels)])
    
    def _update_ensemble_weights(self, recent_accuracies: Dict[str, float]):
        """
        Update ensemble weights based on recent performance.
        
        Args:
            recent_accuracies: Recent accuracy scores for each model
        """
        if not recent_accuracies:
            return
        
        # Calculate new weights based on recent performance
        total_accuracy = sum(recent_accuracies.values())
        
        if total_accuracy > 0:
            for model_type in self.model_types:
                if model_type in recent_accuracies:
                    # Weighted average of current weight and performance-based weight
                    performance_weight = recent_accuracies[model_type] / total_accuracy
                    current_weight = self.model_weights[model_type]
                    
                    # Smooth update (learning rate = 0.1)
                    self.model_weights[model_type] = 0.9 * current_weight + 0.1 * performance_weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_type in self.model_types:
                self.model_weights[model_type] /= total_weight
        
        logger.info(f"Updated ensemble weights: {self.model_weights}")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            texts: Input texts
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for model_type, model in self.models.items():
            if model.is_fitted:
                preds, probs = model.predict(texts)
                model_predictions[model_type] = preds
                model_probabilities[model_type] = probs
        
        if not model_predictions:
            raise ValueError("No fitted models in ensemble")
        
        # Weighted ensemble prediction
        ensemble_probabilities = np.zeros((len(texts), 2))
        total_weight = 0
        
        for model_type, probs in model_probabilities.items():
            weight = self.model_weights[model_type]
            ensemble_probabilities += weight * probs
            total_weight += weight
        
        if total_weight > 0:
            ensemble_probabilities /= total_weight
        
        # Convert probabilities to predictions
        ensemble_predictions = (ensemble_probabilities[:, 1] > 0.5).astype(int)
        
        return ensemble_predictions, ensemble_probabilities
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current ensemble weights.
        
        Returns:
            Dictionary of model weights
        """
        return self.model_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all models.
        
        Returns:
            Performance summary
        """
        summary = {
            'model_weights': self.model_weights,
            'model_performance': {},
            'ensemble_stats': {
                'total_models': len(self.models),
                'fitted_models': sum(1 for model in self.models.values() if model.is_fitted),
                'total_updates': sum(model.update_count for model in self.models.values())
            }
        }
        
        for model_type, model in self.models.items():
            if model.is_fitted:
                performance_history = model.get_performance_history()
                summary['model_performance'][model_type] = {
                    'update_count': model.update_count,
                    'latest_accuracy': performance_history[-1]['accuracy'] if performance_history else 0.0,
                    'performance_trend': performance_history
                }
        
        return summary

class HumanInLoopModerator:
    """
    Main class that orchestrates the human-in-the-loop moderation system.
    """
    
    def __init__(self, db_path: str = "feedback_storage.db", 
                 model_save_path: str = "models/",
                 update_threshold: int = 10,
                 auto_update: bool = True):
        self.feedback_storage = FeedbackStorage(db_path)
        self.ensemble = AdaptiveEnsemble()
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        self.update_threshold = update_threshold  # Minimum feedback samples before update
        self.auto_update = auto_update
        self.pending_feedback = []
        self.model_version = "1.0.0"
        
        # Threading for background updates
        self.update_lock = threading.Lock()
        self.background_thread = None
        self.stop_background = False
        
        if auto_update:
            self._start_background_updater()
    
    def initial_training(self, texts: List[str], labels: List[int]):
        """
        Perform initial training of the ensemble.
        
        Args:
            texts: Training texts
            labels: Training labels
        """
        logger.info("Starting initial training of human-in-the-loop system...")
        
        with self.update_lock:
            self.ensemble.initial_fit(texts, labels)
            
            # Save initial model
            self._save_ensemble(f"initial_{self.model_version}")
            
            # Log initial training
            self.feedback_storage.log_model_update(
                model_version=self.model_version,
                update_type="initial_training",
                feedback_count=len(texts),
                update_details=f"Initial training with {len(texts)} samples"
            )
        
        logger.info("Initial training completed")
    
    def moderate_content(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Moderate content and return prediction with explanation.
        
        Args:
            text: Text to moderate
            language: Language of the text
            
        Returns:
            Moderation result with prediction, confidence, and explanation
        """
        # Get ensemble prediction
        predictions, probabilities = self.ensemble.predict([text])
        
        prediction = predictions[0]
        confidence = probabilities[0][1] if prediction == 1 else probabilities[0][0]
        
        # Get individual model predictions for explanation
        model_predictions = {}
        for model_type, model in self.ensemble.models.items():
            if model.is_fitted:
                model_preds, model_probs = model.predict([text])
                model_predictions[model_type] = {
                    'prediction': model_preds[0],
                    'confidence': model_probs[0][1] if model_preds[0] == 1 else model_probs[0][0],
                    'weight': self.ensemble.model_weights[model_type]
                }
        
        result = {
            'text': text,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'language': language,
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'explanation': {
                'ensemble_weights': self.ensemble.get_model_weights(),
                'model_predictions': model_predictions,
                'decision_rationale': self._generate_decision_rationale(prediction, confidence, model_predictions)
            }
        }
        
        return result
    
    def submit_feedback(self, text: str, original_prediction: int, original_confidence: float,
                       human_label: int, moderator_id: str = None, 
                       feedback_type: str = "correction", explanation: str = None) -> int:
        """
        Submit human feedback for model improvement.
        
        Args:
            text: Original text
            original_prediction: Model's original prediction
            original_confidence: Model's original confidence
            human_label: Human moderator's label
            moderator_id: ID of the moderator
            feedback_type: Type of feedback
            explanation: Optional explanation
            
        Returns:
            Feedback ID
        """
        # Store feedback
        feedback_id = self.feedback_storage.store_feedback(
            text=text,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            human_label=human_label,
            moderator_id=moderator_id,
            feedback_type=feedback_type,
            explanation=explanation,
            model_version=self.model_version
        )
        
        # Add to pending feedback for batch updates
        self.pending_feedback.append({
            'text': text,
            'label': human_label,
            'feedback_id': feedback_id
        })
        
        # Trigger update if threshold reached
        if len(self.pending_feedback) >= self.update_threshold:
            if self.auto_update:
                self._trigger_model_update()
            else:
                logger.info(f"Feedback threshold reached ({len(self.pending_feedback)} samples). Manual update recommended.")
        
        return feedback_id
    
    def manual_update(self) -> Dict[str, Any]:
        """
        Manually trigger model update with pending feedback.
        
        Returns:
            Update results
        """
        return self._trigger_model_update()
    
    def _trigger_model_update(self) -> Dict[str, Any]:
        """
        Trigger model update with pending feedback.
        
        Returns:
            Update results
        """
        if not self.pending_feedback:
            return {'status': 'no_feedback', 'message': 'No pending feedback for update'}
        
        with self.update_lock:
            # Prepare feedback data
            texts = [fb['text'] for fb in self.pending_feedback]
            labels = [fb['label'] for fb in self.pending_feedback]
            
            # Get accuracy before update
            accuracy_before = self._evaluate_current_performance(texts, labels)
            
            # Update ensemble
            self.ensemble.update_with_feedback(texts, labels)
            
            # Get accuracy after update
            accuracy_after = self._evaluate_current_performance(texts, labels)
            
            # Update model version
            old_version = self.model_version
            version_parts = self.model_version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            self.model_version = '.'.join(version_parts)
            
            # Save updated model
            self._save_ensemble(f"updated_{self.model_version}")
            
            # Log update
            self.feedback_storage.log_model_update(
                model_version=self.model_version,
                update_type="incremental_feedback",
                feedback_count=len(self.pending_feedback),
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                update_details=f"Updated from {old_version} with {len(self.pending_feedback)} feedback samples"
            )
            
            # Clear pending feedback
            feedback_count = len(self.pending_feedback)
            self.pending_feedback.clear()
            
            result = {
                'status': 'success',
                'feedback_count': feedback_count,
                'model_version': self.model_version,
                'accuracy_before': accuracy_before,
                'accuracy_after': accuracy_after,
                'improvement': accuracy_after - accuracy_before if accuracy_before else 0.0
            }
            
            logger.info(f"Model updated to version {self.model_version}. Accuracy change: {result['improvement']:.3f}")
            return result
    
    def _evaluate_current_performance(self, texts: List[str], labels: List[int]) -> float:
        """
        Evaluate current ensemble performance.
        
        Args:
            texts: Test texts
            labels: Test labels
            
        Returns:
            Accuracy score
        """
        try:
            predictions, _ = self.ensemble.predict(texts)
            return accuracy_score(labels, predictions)
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return 0.0
    
    def _generate_decision_rationale(self, prediction: int, confidence: float, 
                                   model_predictions: Dict[str, Dict]) -> str:
        """
        Generate human-readable decision rationale.
        
        Args:
            prediction: Final prediction
            confidence: Final confidence
            model_predictions: Individual model predictions
            
        Returns:
            Decision rationale string
        """
        decision = "TOXIC" if prediction == 1 else "NON-TOXIC"
        
        rationale = f"Classified as {decision} with {confidence:.2f} confidence. "
        
        # Analyze model agreement
        agreements = sum(1 for mp in model_predictions.values() if mp['prediction'] == prediction)
        total_models = len(model_predictions)
        
        if agreements == total_models:
            rationale += "All models agree on this classification."
        elif agreements > total_models / 2:
            rationale += f"Majority of models ({agreements}/{total_models}) agree on this classification."
        else:
            rationale += f"Models are split on this classification ({agreements}/{total_models} agree)."
        
        # Highlight most confident model
        if model_predictions:
            most_confident = max(model_predictions.items(), key=lambda x: x[1]['confidence'])
            rationale += f" Most confident model: {most_confident[0]} ({most_confident[1]['confidence']:.2f})."
        
        return rationale
    
    def _save_ensemble(self, version_suffix: str):
        """
        Save the current ensemble.
        
        Args:
            version_suffix: Suffix for the saved model filename
        """
        for model_type, model in self.ensemble.models.items():
            if model.is_fitted:
                filepath = self.model_save_path / f"{model_type}_{version_suffix}.joblib"
                model.save_model(str(filepath))
        
        # Save ensemble metadata
        ensemble_metadata = {
            'model_version': self.model_version,
            'model_weights': self.ensemble.get_model_weights(),
            'performance_summary': self.ensemble.get_performance_summary(),
            'save_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.model_save_path / f"ensemble_metadata_{version_suffix}.json"
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
    
    def _start_background_updater(self):
        """
        Start background thread for automatic updates.
        """
        def background_update_loop():
            while not self.stop_background:
                time.sleep(300)  # Check every 5 minutes
                
                if len(self.pending_feedback) >= self.update_threshold:
                    try:
                        self._trigger_model_update()
                    except Exception as e:
                        logger.error(f"Background update failed: {e}")
        
        self.background_thread = threading.Thread(target=background_update_loop, daemon=True)
        self.background_thread.start()
        logger.info("Background updater started")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            System statistics
        """
        # Get recent feedback stats
        recent_feedback = self.feedback_storage.get_recent_feedback(hours=24)
        disagreement_cases = self.feedback_storage.get_disagreement_cases(limit=100)
        
        # Get moderator stats
        moderator_stats = self.feedback_storage.get_moderator_stats()
        
        # Get ensemble performance
        ensemble_summary = self.ensemble.get_performance_summary()
        
        stats = {
            'system_info': {
                'model_version': self.model_version,
                'pending_feedback_count': len(self.pending_feedback),
                'update_threshold': self.update_threshold,
                'auto_update_enabled': self.auto_update
            },
            'feedback_stats': {
                'recent_feedback_24h': len(recent_feedback),
                'disagreement_cases': len(disagreement_cases),
                'disagreement_rate': len(disagreement_cases) / len(recent_feedback) if recent_feedback else 0.0
            },
            'moderator_stats': moderator_stats,
            'ensemble_performance': ensemble_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def generate_system_report(self) -> str:
        """
        Generate a comprehensive system report.
        
        Returns:
            Formatted report string
        """
        stats = self.get_system_stats()
        
        report = []
        report.append("Human-in-the-Loop Moderation System Report")
        report.append("=" * 50)
        report.append("")
        
        # System Info
        system_info = stats['system_info']
        report.append("SYSTEM INFORMATION")
        report.append("-" * 20)
        report.append(f"Model Version: {system_info['model_version']}")
        report.append(f"Pending Feedback: {system_info['pending_feedback_count']}")
        report.append(f"Update Threshold: {system_info['update_threshold']}")
        report.append(f"Auto-Update: {'Enabled' if system_info['auto_update_enabled'] else 'Disabled'}")
        report.append("")
        
        # Feedback Stats
        feedback_stats = stats['feedback_stats']
        report.append("FEEDBACK STATISTICS (24h)")
        report.append("-" * 25)
        report.append(f"Recent Feedback: {feedback_stats['recent_feedback_24h']}")
        report.append(f"Disagreement Cases: {feedback_stats['disagreement_cases']}")
        report.append(f"Disagreement Rate: {feedback_stats['disagreement_rate']:.2%}")
        report.append("")
        
        # Ensemble Performance
        ensemble_perf = stats['ensemble_performance']
        report.append("ENSEMBLE PERFORMANCE")
        report.append("-" * 20)
        report.append(f"Total Models: {ensemble_perf['ensemble_stats']['total_models']}")
        report.append(f"Fitted Models: {ensemble_perf['ensemble_stats']['fitted_models']}")
        report.append(f"Total Updates: {ensemble_perf['ensemble_stats']['total_updates']}")
        report.append("")
        
        # Model Weights
        report.append("MODEL WEIGHTS:")
        for model_type, weight in ensemble_perf['model_weights'].items():
            report.append(f"  {model_type}: {weight:.3f}")
        report.append("")
        
        # Moderator Performance
        moderator_stats = stats['moderator_stats']
        if isinstance(moderator_stats, list) and moderator_stats:
            report.append("TOP MODERATORS")
            report.append("-" * 15)
            for i, moderator in enumerate(moderator_stats[:5]):
                report.append(f"{i+1}. {moderator['moderator_id']}: {moderator['total_feedback']} feedback, {moderator['agreement_rate']:.2%} agreement")
        
        return "\n".join(report)
    
    def shutdown(self):
        """
        Shutdown the system gracefully.
        """
        logger.info("Shutting down human-in-the-loop system...")
        
        # Stop background thread
        self.stop_background = True
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        # Process any remaining feedback
        if self.pending_feedback:
            logger.info(f"Processing {len(self.pending_feedback)} remaining feedback samples...")
            self._trigger_model_update()
        
        # Close database connection
        self.feedback_storage.close()
        
        logger.info("System shutdown complete")

# Example usage and testing functions
def demo_human_in_loop_system():
    """
    Demonstrate the human-in-the-loop moderation system.
    """
    print("Human-in-the-Loop ML Moderation System Demo")
    print("=" * 50)
    
    # Initialize system
    moderator = HumanInLoopModerator(
        db_path="demo_feedback.db",
        model_save_path="demo_models/",
        update_threshold=5,
        auto_update=True
    )
    
    print("\nSystem components:")
    print("- FeedbackStorage: SQLite database for storing human feedback")
    print("- IncrementalLearner: Supports SGD, Passive-Aggressive, and Multinomial NB")
    print("- AdaptiveEnsemble: Ensemble with dynamic weight adjustment")
    print("- HumanInLoopModerator: Main orchestration class")
    
    print("\nKey features:")
    print("- Incremental learning with partial_fit()")
    print("- Adaptive ensemble weighting based on performance")
    print("- Human feedback storage and analysis")
    print("- Automatic model updates when feedback threshold reached")
    print("- Moderator performance tracking")
    print("- Explainable predictions with model rationale")
    
    print("\nNote: This demo shows the system architecture.")
    print("Full functionality requires initial training data and human feedback.")
    
    # Cleanup
    moderator.shutdown()

if __name__ == "__main__":
    demo_human_in_loop_system()