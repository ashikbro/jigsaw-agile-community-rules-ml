import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], file_path: Union[str, Path], **kwargs) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        **kwargs: Additional arguments to json.dump
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, **kwargs)
    
    logger.info(f"Saved data to {file_path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        average: Averaging strategy for multi-class classification
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
            else:  # Binary
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
    
    return metrics


def save_metrics(
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    file_name: str = 'metrics.json'
) -> None:
    """Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save the file
        file_name: Name of the output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / file_name
    save_json(metrics, file_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_file: Union[str, Path],
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    cmap: str = 'Blues'
) -> None:
    """Plot and save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_file: Path to save the plot
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        cmap: Color map for the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names or [str(i) for i in range(len(np.unique(y_true)))],
        yticklabels=class_names or [str(i) for i in range(len(np.unique(y_true)))]
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_file}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: Union[str, Path],
    metrics: List[str] = None
) -> None:
    """Plot training history.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save the plots
        metrics: List of metrics to plot (default: all available)
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default metrics if not provided
    if metrics is None:
        metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    # Plot each metric
    for metric in metrics:
        if metric not in history:
            logger.warning(f"Metric '{metric}' not found in history")
            continue
        
        # Create a new figure
        plt.figure(figsize=(10, 5))
        
        # Plot training metric
        plt.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            plt.plot(history[val_metric], label=f'Validation {metric}')
        
        # Customize plot
        plt.title(f'Model {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = output_dir / f'{metric}_history.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {metric} plot to {output_file}")


def setup_logging(
    log_file: Union[str, Path] = None,
    log_level: int = logging.INFO
) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to the log file (optional)
        log_level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set log level for specific loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger.info("Logging setup complete")
