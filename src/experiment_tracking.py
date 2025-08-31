"""Experiment tracking module using MLflow.

This module provides utilities for tracking experiments using MLflow.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MLFLOW_DIR = BASE_DIR / 'mlruns'

class ExperimentTracker:
    """Class for tracking experiments using MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking server (default: local file system)
        """
        self.experiment_name = experiment_name
        
        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local file system
            os.makedirs(MLFLOW_DIR, exist_ok=True)
            mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        
        # Set or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"file://{MLFLOW_DIR}/{experiment_name}"
            )
        except Exception:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        logger.info(f"Initialized experiment '{experiment_name}' with ID {self.experiment_id}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new run.
        
        Args:
            run_name: Name of the run
            
        Returns:
            MLflow active run
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: Any, model_name: str) -> None:
        """Log a model.
        
        Args:
            model: Model to log
            model_name: Name of the model
        """
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log an artifact.
        
        Args:
            local_path: Path to the artifact
        """
        mlflow.log_artifact(str(local_path))
    
    def evaluate_and_log_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                              model_name: str) -> Dict[str, float]:
        """Evaluate a model and log the results.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        # Log metrics
        self.log_metrics(metrics)
        
        # Log model
        self.log_model(model, model_name)
        
        return metrics

def get_experiment_tracker(experiment_name: str) -> ExperimentTracker:
    """Get an experiment tracker.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(experiment_name)