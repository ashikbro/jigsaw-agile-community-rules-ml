#!/usr/bin/env python3
"""
Multi-Rule Classification Framework for Reddit Comment Moderation

This module implements a comprehensive framework for handling multiple subreddit rules
and community-specific moderation patterns in the Kaggle competition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubredditRule:
    """Represents a specific subreddit rule with its characteristics."""
    rule_id: str
    subreddit: str
    rule_text: str
    category: str  # e.g., 'harassment', 'spam', 'off-topic', 'personal_info'
    severity: int  # 1-5 scale
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    context_dependent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'subreddit': self.subreddit,
            'rule_text': self.rule_text,
            'category': self.category,
            'severity': self.severity,
            'keywords': self.keywords,
            'patterns': self.patterns,
            'context_dependent': self.context_dependent
        }

class RuleRepository:
    """Manages collection of subreddit rules and their relationships."""
    
    def __init__(self):
        self.rules: Dict[str, SubredditRule] = {}
        self.subreddit_rules: Dict[str, List[str]] = {}
        self.category_rules: Dict[str, List[str]] = {}
        
    def add_rule(self, rule: SubredditRule) -> None:
        """Add a new rule to the repository."""
        self.rules[rule.rule_id] = rule
        
        # Update subreddit mapping
        if rule.subreddit not in self.subreddit_rules:
            self.subreddit_rules[rule.subreddit] = []
        self.subreddit_rules[rule.subreddit].append(rule.rule_id)
        
        # Update category mapping
        if rule.category not in self.category_rules:
            self.category_rules[rule.category] = []
        self.category_rules[rule.category].append(rule.rule_id)
    
    def get_rules_by_subreddit(self, subreddit: str) -> List[SubredditRule]:
        """Get all rules for a specific subreddit."""
        rule_ids = self.subreddit_rules.get(subreddit, [])
        return [self.rules[rule_id] for rule_id in rule_ids]
    
    def get_rules_by_category(self, category: str) -> List[SubredditRule]:
        """Get all rules in a specific category."""
        rule_ids = self.category_rules.get(category, [])
        return [self.rules[rule_id] for rule_id in rule_ids]
    
    def load_from_json(self, file_path: str) -> None:
        """Load rules from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        for rule_data in rules_data:
            rule = SubredditRule(**rule_data)
            self.add_rule(rule)
    
    def save_to_json(self, file_path: str) -> None:
        """Save rules to JSON file."""
        rules_data = [rule.to_dict() for rule in self.rules.values()]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)

class MultiRuleTransformer(nn.Module):
    """Transformer model for multi-rule classification."""
    
    def __init__(self, model_name: str, num_rules: int, hidden_dim: int = 768):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.num_rules = num_rules
        
        # Rule-specific heads
        self.rule_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_rules)
        ])
        
        # Shared feature extractor
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Rule interaction layer
        self.rule_interaction = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
    
    def forward(self, input_ids, attention_mask, rule_context=None):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Shared feature extraction
        shared_features = self.shared_head(pooled_output)
        
        # Rule-specific predictions
        rule_predictions = []
        for i, classifier in enumerate(self.rule_classifiers):
            pred = classifier(shared_features)
            rule_predictions.append(pred)
        
        return torch.cat(rule_predictions, dim=1)

class MultiRuleClassifier:
    """Main classifier for handling multiple subreddit rules."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.rule_repository = RuleRepository()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize with common Reddit rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize with common Reddit moderation rules."""
        default_rules = [
            SubredditRule(
                rule_id='harassment_1',
                subreddit='general',
                rule_text='No harassment, bullying, or personal attacks',
                category='harassment',
                severity=5,
                keywords=['idiot', 'stupid', 'kill yourself', 'moron', 'retard'],
                patterns=[r'you\s+are\s+(?:an?\s+)?(?:idiot|moron|stupid)', r'go\s+(?:kill\s+yourself|die)']
            ),
            SubredditRule(
                rule_id='spam_1',
                subreddit='general',
                rule_text='No spam or self-promotion',
                category='spam',
                severity=3,
                keywords=['buy now', 'click here', 'subscribe', 'follow me'],
                patterns=[r'(?:buy|purchase)\s+(?:now|today)', r'(?:subscribe|follow)\s+(?:me|my|us)']
            ),
            SubredditRule(
                rule_id='personal_info_1',
                subreddit='general',
                rule_text='No sharing of personal information',
                category='personal_info',
                severity=5,
                keywords=['phone number', 'address', 'social security', 'credit card'],
                patterns=[r'\b\d{3}-\d{3}-\d{4}\b', r'\b\d{3}\s\d{3}\s\d{4}\b']
            ),
            SubredditRule(
                rule_id='off_topic_1',
                subreddit='general',
                rule_text='Stay on topic and relevant to the subreddit',
                category='off_topic',
                severity=2,
                keywords=['off topic', 'unrelated', 'wrong subreddit'],
                context_dependent=True
            )
        ]
        
        for rule in default_rules:
            self.rule_repository.add_rule(rule)
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'comment_text') -> Dict[str, torch.Tensor]:
        """Prepare data for training/inference."""
        texts = df[text_column].tolist()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, 
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the multi-rule classifier."""
        # Initialize model
        num_rules = len(self.rule_repository.rules)
        self.model = MultiRuleTransformer(self.model_name, num_rules)
        self.model.to(self.device)
        
        # Prepare data
        train_data = self.prepare_data(train_df)
        
        # Prepare labels (assuming rule violation columns exist)
        rule_columns = [f'rule_{rule_id}' for rule_id in self.rule_repository.rules.keys()]
        train_labels = torch.tensor(train_df[rule_columns].values, dtype=torch.float32)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = len(train_data['input_ids']) // batch_size
            
            for i in range(0, len(train_data['input_ids']), batch_size):
                batch_input_ids = train_data['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = train_data['attention_mask'][i:i+batch_size].to(self.device)
                batch_labels = train_labels[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            logger.info(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
            
            # Validation
            if val_df is not None:
                val_auc = self.evaluate(val_df)
                logger.info(f'Validation AUC: {val_auc:.4f}')
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        data = self.prepare_data(df)
        
        predictions = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(data['input_ids']), batch_size):
                batch_input_ids = data['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = data['attention_mask'][i:i+batch_size].to(self.device)
                
                outputs = self.model(batch_input_ids, batch_attention_mask)
                predictions.append(outputs.cpu().numpy())
        
        return np.vstack(predictions)
    
    def evaluate(self, df: pd.DataFrame) -> float:
        """Evaluate model using column-averaged AUC."""
        predictions = self.predict(df)
        
        rule_columns = [f'rule_{rule_id}' for rule_id in self.rule_repository.rules.keys()]
        true_labels = df[rule_columns].values
        
        # Calculate AUC for each rule
        aucs = []
        for i in range(len(rule_columns)):
            if len(np.unique(true_labels[:, i])) > 1:  # Check if both classes exist
                auc = roc_auc_score(true_labels[:, i], predictions[:, i])
                aucs.append(auc)
        
        return np.mean(aucs) if aucs else 0.0
    
    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Use first rule column for stratification
        rule_columns = [f'rule_{rule_id}' for rule_id in self.rule_repository.rules.keys()]
        y_stratify = df[rule_columns[0]].values
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_stratify)):
            logger.info(f'Training fold {fold + 1}/{n_splits}')
            
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
            # Train model for this fold
            self.train(train_fold, val_fold, epochs=2)
            
            # Evaluate
            fold_score = self.evaluate(val_fold)
            cv_scores.append(fold_score)
            logger.info(f'Fold {fold + 1} AUC: {fold_score:.4f}')
        
        return {
            'mean_auc': np.mean(cv_scores),
            'std_auc': np.std(cv_scores),
            'fold_scores': cv_scores
        }
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'rules': [rule.to_dict() for rule in self.rule_repository.rules.values()]
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore rules
        self.rule_repository = RuleRepository()
        for rule_data in checkpoint['rules']:
            rule = SubredditRule(**rule_data)
            self.rule_repository.add_rule(rule)
        
        # Initialize and load model
        num_rules = len(self.rule_repository.rules)
        self.model = MultiRuleTransformer(checkpoint['model_name'], num_rules)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

def main():
    """Example usage of the multi-rule classifier."""
    # Initialize classifier
    classifier = MultiRuleClassifier()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'comment_text': [
            'You are such an idiot for posting this',
            'Great post! Thanks for sharing',
            'Buy now and get 50% off! Click here!',
            'My phone number is 555-123-4567',
            'This is completely off-topic for this subreddit'
        ],
        'rule_harassment_1': [1, 0, 0, 0, 0],
        'rule_spam_1': [0, 0, 1, 0, 0],
        'rule_personal_info_1': [0, 0, 0, 1, 0],
        'rule_off_topic_1': [0, 0, 0, 0, 1]
    })
    
    # Train the model
    logger.info("Training multi-rule classifier...")
    classifier.train(sample_data, epochs=1)  # Quick training for demo
    
    # Make predictions
    predictions = classifier.predict(sample_data)
    logger.info(f"Predictions shape: {predictions.shape}")
    
    # Evaluate
    auc_score = classifier.evaluate(sample_data)
    logger.info(f"AUC Score: {auc_score:.4f}")
    
    # Save model
    classifier.save_model('multi_rule_model.pth')
    logger.info("Model saved successfully")

if __name__ == '__main__':
    main()