"""Reddit Comment Rule Violation Classifier.

This module implements a binary classifier using transformer models (BERT/RoBERTa)
optimized for column-averaged AUC performance in the Kaggle competition.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the Reddit classifier model."""
    model_name: str = 'bert-base-uncased'
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.3
    hidden_size: int = 768
    num_classes: int = 1  # Binary classification
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

class RedditCommentDataset(Dataset):
    """Dataset class for Reddit comments."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

class RedditTransformerClassifier(nn.Module):
    """Transformer-based classifier for Reddit comment rule violation detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class RedditClassifierTrainer:
    """Trainer class for Reddit comment classifier."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_model(self):
        """Initialize model, optimizer, and scheduler."""
        self.model = RedditTransformerClassifier(self.config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_scheduler(self, num_training_steps: int):
        """Prepare learning rate scheduler."""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(logits.squeeze(), labels)
                
                total_loss += loss.item()
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits.squeeze())
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(eval_loader)
        auc_score = roc_auc_score(true_labels, predictions)
        
        return avg_loss, auc_score
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: Optional[List[str]] = None, 
              val_labels: Optional[List[int]] = None) -> Dict[str, List[float]]:
        """Train the model."""
        # Prepare model
        self.prepare_model()
        
        # Create datasets
        train_dataset = RedditCommentDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        # Validation dataset
        val_loader = None
        if val_texts and val_labels:
            val_dataset = RedditCommentDataset(
                val_texts, val_labels, self.tokenizer, self.config.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )
        
        # Prepare scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        self.prepare_scheduler(num_training_steps)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_loss, val_auc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")
        
        return history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts."""
        self.model.eval()
        
        dataset = RedditCommentDataset(
            texts, None, self.tokenizer, self.config.max_length
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits.squeeze())
                predictions.extend(probs.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path / 'tokenizer')
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and tokenizer."""
        path = Path(path)
        
        # Load model
        checkpoint = torch.load(path / 'model.pt', map_location=self.device)
        self.config = checkpoint['config']
        
        self.model = RedditTransformerClassifier(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer')
        
        logger.info(f"Model loaded from {path}")

class EnsembleClassifier:
    """Ensemble classifier combining transformer and traditional ML models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.transformer_trainer = RedditClassifierTrainer(config)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.weights = [0.6, 0.25, 0.15]  # Transformer, RF, LR
    
    def train(self, train_texts: List[str], train_labels: List[int],
              train_features: np.ndarray,
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              val_features: Optional[np.ndarray] = None):
        """Train ensemble models."""
        # Train transformer
        logger.info("Training transformer model...")
        self.transformer_trainer.train(train_texts, train_labels, val_texts, val_labels)
        
        # Train traditional ML models
        logger.info("Training Random Forest...")
        self.rf_model.fit(train_features, train_labels)
        
        logger.info("Training Logistic Regression...")
        self.lr_model.fit(train_features, train_labels)
        
        # Evaluate ensemble if validation data provided
        if val_texts and val_labels and val_features is not None:
            val_preds = self.predict(val_texts, val_features)
            val_auc = roc_auc_score(val_labels, val_preds)
            logger.info(f"Ensemble Validation AUC: {val_auc:.4f}")
    
    def predict(self, texts: List[str], features: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        # Transformer predictions
        transformer_preds = self.transformer_trainer.predict(texts)
        
        # Traditional ML predictions
        rf_preds = self.rf_model.predict_proba(features)[:, 1]
        lr_preds = self.lr_model.predict_proba(features)[:, 1]
        
        # Weighted ensemble
        ensemble_preds = (
            self.weights[0] * transformer_preds +
            self.weights[1] * rf_preds +
            self.weights[2] * lr_preds
        )
        
        return ensemble_preds

def cross_validate_model(texts: List[str], labels: List[int], 
                        features: np.ndarray, config: ModelConfig,
                        n_folds: int = 5) -> Dict[str, float]:
    """Perform cross-validation for model evaluation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        train_features = features[train_idx]
        val_features = features[val_idx]
        
        # Train ensemble
        ensemble = EnsembleClassifier(config)
        ensemble.train(train_texts, train_labels, train_features)
        
        # Predict and evaluate
        val_preds = ensemble.predict(val_texts, val_features)
        fold_auc = roc_auc_score(val_labels, val_preds)
        fold_scores.append(fold_auc)
        
        logger.info(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
    
    return {
        'mean_auc': np.mean(fold_scores),
        'std_auc': np.std(fold_scores),
        'fold_scores': fold_scores
    }

def main():
    """Example usage of the Reddit classifier."""
    # Configuration
    config = ModelConfig(
        model_name='bert-base-uncased',
        batch_size=8,  # Smaller for demo
        num_epochs=1,  # Fewer epochs for demo
        max_length=256
    )
    
    # Sample data
    texts = [
        "This is a great post, thanks for sharing!",
        "You're an idiot if you believe this nonsense.",
        "Check out my website for more info!",
        "This is completely unrelated but I love pizza.",
        "What a load of crap, this is fake news.",
    ] * 20  # Repeat for more samples
    
    labels = [0, 1, 1, 1, 1] * 20
    
    # Create dummy features
    features = np.random.random((len(texts), 10))
    
    # Train model
    trainer = RedditClassifierTrainer(config)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train
    history = trainer.train(train_texts, train_labels, val_texts, val_labels)
    
    # Make predictions
    predictions = trainer.predict(val_texts)
    
    print(f"Validation AUC: {roc_auc_score(val_labels, predictions):.4f}")

if __name__ == '__main__':
    main()