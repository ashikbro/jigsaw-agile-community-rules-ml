# 🛠️ Implementation Plan: Advanced XAI Framework Upgrade

## 🎯 Overview

This document provides a detailed, actionable implementation plan to upgrade the current XAI text classification framework with state-of-the-art models, comprehensive datasets, and advanced explainability techniques.

## 📅 Phase 1: Advanced Model Integration (Weeks 1-4)

### 1.1 Transformer Models Implementation

#### Create Advanced Model Manager
```python
# File: src/advanced_models.py
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,
    DistilBertTokenizer, DistilBertModel, ElectraTokenizer, ElectraModel
)
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class AdvancedModelManager:
    """Manager for advanced transformer models."""
    
    SUPPORTED_MODELS = {
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'distilbert': 'distilbert-base-uncased',
        'electra-base': 'google/electra-base-discriminator',
        'deberta-base': 'microsoft/deberta-base',
        'scibert': 'allenai/scibert_scivocab_uncased',
        'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        'legalbert': 'nlpaueb/legal-bert-base-uncased',
        'finbert': 'ProsusAI/finbert'
    }
    
    def __init__(self, model_name: str, num_labels: int = 2):
        self.model_name = model_name
        self.model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=num_labels
        )
        
    def encode_texts(self, texts: List[str], max_length: int = 512) -> Dict:
        """Encode texts using the tokenizer."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get contextualized embeddings."""
        encoded = self.encode_texts(texts)
        with torch.no_grad():
            outputs = self.model.bert(**encoded) if hasattr(self.model, 'bert') else self.model.roberta(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
```

#### Ensemble Methods Implementation
```python
# File: src/ensemble_methods.py
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from typing import List, Dict, Any

class AdvancedEnsemble:
    """Advanced ensemble methods for text classification."""
    
    def __init__(self):
        self.base_models = {
            'lr': LogisticRegression(random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        self.ensemble_models = {}
    
    def create_voting_ensemble(self, voting_type: str = 'soft') -> VotingClassifier:
        """Create voting ensemble."""
        estimators = [(name, model) for name, model in self.base_models.items()]
        return VotingClassifier(estimators=estimators, voting=voting_type)
    
    def create_stacking_ensemble(self, meta_learner=None) -> StackingClassifier:
        """Create stacking ensemble."""
        if meta_learner is None:
            meta_learner = LogisticRegression()
        
        estimators = [(name, model) for name, model in self.base_models.items()]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5
        )
    
    def transformer_ensemble(self, model_names: List[str], texts: List[str]) -> np.ndarray:
        """Ensemble of transformer models."""
        predictions = []
        
        for model_name in model_names:
            model_manager = AdvancedModelManager(model_name)
            # Get predictions from each transformer
            embeddings = model_manager.get_embeddings(texts)
            # Add classification head logic here
            predictions.append(embeddings.numpy())
        
        # Average predictions
        return np.mean(predictions, axis=0)
```

### 1.2 Few-Shot Learning Implementation

```python
# File: src/few_shot_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot learning."""
    
    def __init__(self, encoder_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, support_set: torch.Tensor, query_set: torch.Tensor, 
                support_labels: torch.Tensor) -> torch.Tensor:
        """Forward pass for prototypical network."""
        # Encode support and query sets
        support_encoded = self.encoder(support_set)
        query_encoded = self.encoder(query_set)
        
        # Compute prototypes (class centroids)
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        for label in unique_labels:
            mask = support_labels == label
            prototype = support_encoded[mask].mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances and probabilities
        distances = torch.cdist(query_encoded, prototypes)
        probabilities = F.softmax(-distances, dim=1)
        
        return probabilities

class FewShotLearner:
    """Few-shot learning framework."""
    
    def __init__(self, encoder_dim: int):
        self.model = PrototypicalNetwork(encoder_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def train_episode(self, support_set: torch.Tensor, support_labels: torch.Tensor,
                     query_set: torch.Tensor, query_labels: torch.Tensor) -> float:
        """Train on a single episode."""
        self.optimizer.zero_grad()
        
        predictions = self.model(support_set, query_set, support_labels)
        loss = F.cross_entropy(predictions, query_labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## 📊 Phase 2: Comprehensive Dataset Integration (Weeks 5-8)

### 2.1 Academic Dataset Loader

```python
# File: src/academic_datasets.py
import datasets
from datasets import load_dataset
import pandas as pd
from typing import Tuple, Optional, Dict, List
import requests
import zipfile
import os

class AcademicDatasetLoader:
    """Loader for academic benchmark datasets."""
    
    HUGGINGFACE_DATASETS = {
        'sst2': 'glue/sst2',
        'imdb': 'imdb',
        'ag_news': 'ag_news',
        'yelp_polarity': 'yelp_polarity',
        'amazon_polarity': 'amazon_polarity',
        '20newsgroups': '20_newsgroups',
        'reuters21578': 'reuters21578',
        'banking77': 'banking77',
        'clinc150': 'clinc_oos',
        'snips': 'snips_built_in_intents'
    }
    
    def __init__(self, cache_dir: str = './data/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_huggingface_dataset(self, dataset_name: str, 
                                split: str = 'train') -> pd.DataFrame:
        """Load dataset from Hugging Face."""
        if dataset_name not in self.HUGGINGFACE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        dataset_path = self.HUGGINGFACE_DATASETS[dataset_name]
        dataset = load_dataset(dataset_path, split=split, cache_dir=self.cache_dir)
        
        return dataset.to_pandas()
    
    def load_scientific_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load scientific domain datasets."""
        datasets = {}
        
        # PubMed abstracts (example implementation)
        try:
            pubmed_data = load_dataset('pubmed_qa', 'pqa_labeled', split='train')
            datasets['pubmed'] = pubmed_data.to_pandas()
        except:
            print("PubMed dataset not available, using synthetic data")
            datasets['pubmed'] = self._generate_synthetic_scientific_data()
        
        return datasets
    
    def load_legal_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load legal domain datasets."""
        datasets = {}
        
        # European Court of Human Rights cases
        try:
            ecthr_data = load_dataset('ecthr_cases', split='train')
            datasets['ecthr'] = ecthr_data.to_pandas()
        except:
            print("ECTHR dataset not available, using synthetic data")
            datasets['ecthr'] = self._generate_synthetic_legal_data()
        
        return datasets
    
    def load_multilingual_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load multilingual datasets."""
        datasets = {}
        
        # Cross-lingual Natural Language Inference
        try:
            xnli_data = load_dataset('xnli', 'en', split='validation')
            datasets['xnli_en'] = xnli_data.to_pandas()
        except:
            print("XNLI dataset not available, using synthetic data")
            datasets['xnli_en'] = self._generate_synthetic_multilingual_data()
        
        return datasets
    
    def _generate_synthetic_scientific_data(self) -> pd.DataFrame:
        """Generate synthetic scientific text data."""
        scientific_texts = [
            "This study investigates the efficacy of novel therapeutic interventions.",
            "Our research demonstrates significant improvements in patient outcomes.",
            "The experimental results show promising applications in clinical settings.",
            "Further investigation is needed to validate these preliminary findings."
        ]
        
        return pd.DataFrame({
            'text': scientific_texts * 250,
            'label': [0, 1, 1, 0] * 250
        })
    
    def _generate_synthetic_legal_data(self) -> pd.DataFrame:
        """Generate synthetic legal text data."""
        legal_texts = [
            "The defendant's constitutional rights were violated during the proceedings.",
            "The court finds insufficient evidence to support the plaintiff's claims.",
            "This case establishes important precedent for future litigation.",
            "The ruling is consistent with established legal principles."
        ]
        
        return pd.DataFrame({
            'text': legal_texts * 250,
            'label': [1, 0, 1, 0] * 250
        })
    
    def _generate_synthetic_multilingual_data(self) -> pd.DataFrame:
        """Generate synthetic multilingual data."""
        multilingual_texts = [
            "This is a positive statement in English.",
            "Esta es una declaración positiva en español.",
            "Ceci est une déclaration positive en français.",
            "Dies ist eine positive Aussage auf Deutsch."
        ]
        
        return pd.DataFrame({
            'text': multilingual_texts * 250,
            'label': [1, 1, 1, 1] * 250
        })
```

### 2.2 Enhanced Dataset Manager

```python
# File: src/enhanced_dataset_manager.py
from src.academic_datasets import AcademicDatasetLoader
from src.dataset_integration import DatasetManager
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np

class EnhancedDatasetManager(DatasetManager):
    """Enhanced dataset manager with academic datasets."""
    
    def __init__(self):
        super().__init__()
        self.academic_loader = AcademicDatasetLoader()
        self.domain_datasets = {}
        self._load_academic_datasets()
    
    def _load_academic_datasets(self):
        """Load all academic datasets."""
        # Load Hugging Face datasets
        hf_datasets = [
            'sst2', 'yelp_polarity', 'amazon_polarity', 
            '20newsgroups', 'banking77'
        ]
        
        for dataset_name in hf_datasets:
            try:
                train_data = self.academic_loader.load_huggingface_dataset(
                    dataset_name, 'train'
                )
                test_data = self.academic_loader.load_huggingface_dataset(
                    dataset_name, 'test'
                )
                self.domain_datasets[dataset_name] = {
                    'train': train_data,
                    'test': test_data,
                    'domain': self._get_domain(dataset_name)
                }
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
        
        # Load domain-specific datasets
        scientific_datasets = self.academic_loader.load_scientific_datasets()
        legal_datasets = self.academic_loader.load_legal_datasets()
        multilingual_datasets = self.academic_loader.load_multilingual_datasets()
        
        self.domain_datasets.update({
            **{f"scientific_{k}": {'train': v, 'domain': 'scientific'} 
               for k, v in scientific_datasets.items()},
            **{f"legal_{k}": {'train': v, 'domain': 'legal'} 
               for k, v in legal_datasets.items()},
            **{f"multilingual_{k}": {'train': v, 'domain': 'multilingual'} 
               for k, v in multilingual_datasets.items()}
        })
    
    def get_datasets_by_domain(self, domain: str) -> Dict[str, pd.DataFrame]:
        """Get all datasets from a specific domain."""
        return {
            name: data for name, data in self.domain_datasets.items()
            if data.get('domain') == domain
        }
    
    def create_cross_domain_splits(self, source_domain: str, 
                                  target_domain: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create cross-domain train/test splits."""
        source_datasets = self.get_datasets_by_domain(source_domain)
        target_datasets = self.get_datasets_by_domain(target_domain)
        
        # Combine source domain data for training
        train_dfs = []
        for dataset_info in source_datasets.values():
            if 'train' in dataset_info:
                train_dfs.append(dataset_info['train'])
        
        train_data = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        
        # Combine target domain data for testing
        test_dfs = []
        for dataset_info in target_datasets.values():
            if 'train' in dataset_info:  # Use train split as test for cross-domain
                test_dfs.append(dataset_info['train'])
        
        test_data = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        return train_data, test_data
    
    def _get_domain(self, dataset_name: str) -> str:
        """Get domain for dataset."""
        domain_mapping = {
            'sst2': 'sentiment',
            'imdb': 'sentiment',
            'yelp_polarity': 'sentiment',
            'amazon_polarity': 'sentiment',
            'ag_news': 'news',
            '20newsgroups': 'news',
            'reuters21578': 'news',
            'banking77': 'intent',
            'clinc150': 'intent',
            'snips': 'intent'
        }
        return domain_mapping.get(dataset_name, 'general')
```

## 🔬 Phase 3: Advanced XAI Techniques (Weeks 9-12)

### 3.1 Transformer-Specific XAI

```python
# File: src/transformer_xai.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import (
    IntegratedGradients, GradientShap, DeepLift,
    LayerConductance, LayerIntegratedGradients
)

class TransformerXAI:
    """XAI techniques specifically for transformer models."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def get_attention_weights(self, text: str, layer: int = -1) -> np.ndarray:
        """Extract attention weights from transformer."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention_weights = outputs.attentions[layer][0]  # First batch item
        
        return attention_weights.numpy()
    
    def visualize_attention(self, text: str, layer: int = -1, head: int = 0):
        """Visualize attention weights."""
        attention_weights = self.get_attention_weights(text, layer)
        tokens = self.tokenizer.tokenize(text)
        
        # Select specific head
        attention_matrix = attention_weights[head]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_matrix[:len(tokens), :len(tokens)],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar=True
        )
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def attention_rollout(self, text: str) -> np.ndarray:
        """Compute attention rollout across all layers."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention_weights = outputs.attentions
        
        # Average attention heads and add residual connections
        rollout = torch.eye(attention_weights[0].size(-1))
        
        for layer_attention in attention_weights:
            # Average over heads
            avg_attention = layer_attention.mean(dim=1)[0]
            # Add residual connection
            avg_attention = avg_attention + torch.eye(avg_attention.size(-1))
            # Normalize
            avg_attention = avg_attention / avg_attention.sum(dim=-1, keepdim=True)
            # Multiply with previous rollout
            rollout = torch.matmul(avg_attention, rollout)
        
        return rollout.numpy()
    
    def integrated_gradients_analysis(self, text: str, target_class: int = 1) -> Dict:
        """Perform integrated gradients analysis."""
        def forward_func(input_ids):
            outputs = self.model(input_ids)
            return outputs.logits[:, target_class]
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        
        # Create baseline (all padding tokens)
        baseline = torch.zeros_like(input_ids)
        
        # Initialize Integrated Gradients
        ig = IntegratedGradients(forward_func)
        
        # Compute attributions
        attributions = ig.attribute(
            input_ids,
            baseline,
            n_steps=50,
            return_convergence_delta=True
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'tokens': tokens,
            'attributions': attributions[0][0].detach().numpy(),
            'convergence_delta': attributions[1].item()
        }
    
    def layer_wise_analysis(self, text: str) -> Dict[int, np.ndarray]:
        """Analyze representations at each layer."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        layer_outputs = {}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for i, layer_output in enumerate(hidden_states):
                # Use [CLS] token representation
                cls_representation = layer_output[0, 0, :].numpy()
                layer_outputs[i] = cls_representation
        
        return layer_outputs
    
    def probing_analysis(self, texts: List[str], labels: List[int], 
                        probe_type: str = 'linear') -> Dict:
        """Perform probing analysis on representations."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        layer_representations = {}
        layer_accuracies = {}
        
        # Extract representations for all texts
        for text in texts:
            layer_outputs = self.layer_wise_analysis(text)
            for layer, representation in layer_outputs.items():
                if layer not in layer_representations:
                    layer_representations[layer] = []
                layer_representations[layer].append(representation)
        
        # Train probes for each layer
        for layer, representations in layer_representations.items():
            X = np.array(representations)
            y = np.array(labels)
            
            # Train probe
            if probe_type == 'linear':
                probe = LogisticRegression(random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                probe = RandomForestClassifier(random_state=42)
            
            probe.fit(X, y)
            predictions = probe.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            layer_accuracies[layer] = accuracy
        
        return {
            'layer_accuracies': layer_accuracies,
            'representations': layer_representations
        }
```

### 3.2 Causal XAI Methods

```python
# File: src/causal_xai.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from scipy import stats

class CausalXAI:
    """Causal explanation methods for text classification."""
    
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
    
    def counterfactual_explanations(self, text: str, target_class: int, 
                                  num_changes: int = 3) -> List[Dict]:
        """Generate counterfactual explanations."""
        import random
        from nltk.corpus import wordnet
        
        # Tokenize text
        words = text.split()
        original_prediction = self.model.predict_proba(
            self.vectorizer.transform([text])
        )[0]
        
        counterfactuals = []
        
        for _ in range(num_changes):
            # Create a copy of the text
            modified_words = words.copy()
            
            # Randomly select words to replace
            positions_to_change = random.sample(
                range(len(words)), 
                min(num_changes, len(words))
            )
            
            for pos in positions_to_change:
                original_word = words[pos]
                
                # Find synonyms using WordNet
                synonyms = self._get_synonyms(original_word)
                if synonyms:
                    modified_words[pos] = random.choice(synonyms)
            
            modified_text = ' '.join(modified_words)
            modified_prediction = self.model.predict_proba(
                self.vectorizer.transform([modified_text])
            )[0]
            
            # Check if prediction changed to target class
            if np.argmax(modified_prediction) == target_class:
                counterfactuals.append({
                    'original_text': text,
                    'modified_text': modified_text,
                    'original_prediction': original_prediction,
                    'modified_prediction': modified_prediction,
                    'changes': [(pos, words[pos], modified_words[pos]) 
                               for pos in positions_to_change 
                               if words[pos] != modified_words[pos]]
                })
        
        return counterfactuals
    
    def causal_mediation_analysis(self, texts: List[str], labels: List[int], 
                                 mediator_features: List[str]) -> Dict:
        """Perform causal mediation analysis."""
        # Extract features
        X = self.vectorizer.transform(texts).toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Find indices of mediator features
        mediator_indices = []
        for feature in mediator_features:
            if feature in feature_names:
                mediator_indices.append(list(feature_names).index(feature))
        
        if not mediator_indices:
            return {'error': 'No mediator features found in vocabulary'}
        
        # Compute direct and indirect effects
        results = {}
        
        for i, mediator_idx in enumerate(mediator_indices):
            mediator_name = feature_names[mediator_idx]
            
            # Total effect (without controlling for mediator)
            total_model = LinearRegression()
            total_model.fit(X, labels)
            total_effect = total_model.coef_[mediator_idx]
            
            # Direct effect (controlling for mediator)
            X_without_mediator = np.delete(X, mediator_idx, axis=1)
            direct_model = LinearRegression()
            direct_model.fit(X_without_mediator, labels)
            
            # Indirect effect (through mediator)
            mediator_values = X[:, mediator_idx]
            indirect_model = LinearRegression()
            indirect_model.fit(mediator_values.reshape(-1, 1), labels)
            indirect_effect = indirect_model.coef_[0]
            
            results[mediator_name] = {
                'total_effect': total_effect,
                'indirect_effect': indirect_effect,
                'direct_effect': total_effect - indirect_effect,
                'mediation_ratio': indirect_effect / total_effect if total_effect != 0 else 0
            }
        
        return results
    
    def instrumental_variable_analysis(self, texts: List[str], labels: List[int], 
                                     instrument_features: List[str]) -> Dict:
        """Perform instrumental variable analysis."""
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr
        
        # Extract features
        X = self.vectorizer.transform(texts).toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        
        results = {}
        
        for instrument_feature in instrument_features:
            if instrument_feature not in feature_names:
                continue
            
            instrument_idx = list(feature_names).index(instrument_feature)
            instrument_values = X[:, instrument_idx]
            
            # Two-stage least squares
            # Stage 1: Regress treatment on instrument
            stage1_model = LinearRegression()
            stage1_model.fit(instrument_values.reshape(-1, 1), X)
            predicted_treatment = stage1_model.predict(instrument_values.reshape(-1, 1))
            
            # Stage 2: Regress outcome on predicted treatment
            stage2_model = LinearRegression()
            stage2_model.fit(predicted_treatment, labels)
            
            # Compute IV estimate
            iv_estimate = stage2_model.coef_.mean()
            
            # Test instrument validity
            correlation, p_value = pearsonr(instrument_values, labels)
            
            results[instrument_feature] = {
                'iv_estimate': iv_estimate,
                'instrument_correlation': correlation,
                'instrument_p_value': p_value,
                'valid_instrument': p_value < 0.05 and abs(correlation) > 0.1
            }
        
        return results
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        try:
            from nltk.corpus import wordnet
            synonyms = set()
            
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
            
            return list(synonyms)[:5]  # Return top 5 synonyms
        except:
            return []
```

## 📊 Phase 4: Advanced Evaluation Framework (Weeks 13-16)

### 4.1 XAI-Specific Metrics

```python
# File: src/xai_evaluation_metrics.py
import numpy as np
from typing import List, Dict, Tuple, Callable
from sklearn.metrics import accuracy_score, f1_score
import scipy.stats as stats

class XAIEvaluationMetrics:
    """Comprehensive evaluation metrics for XAI methods."""
    
    def __init__(self, model, explanation_method):
        self.model = model
        self.explanation_method = explanation_method
    
    def faithfulness_metrics(self, X: np.ndarray, explanations: List[np.ndarray]) -> Dict:
        """Compute faithfulness metrics for explanations."""
        results = {}
        
        # Sufficiency: How well do important features predict the outcome?
        sufficiency_scores = []
        for i, (x, explanation) in enumerate(zip(X, explanations)):
            # Select top-k most important features
            k = min(10, len(explanation))
            top_indices = np.argsort(np.abs(explanation))[-k:]
            
            # Create masked input with only important features
            masked_x = np.zeros_like(x)
            masked_x[top_indices] = x[top_indices]
            
            # Compare predictions
            original_pred = self.model.predict_proba([x])[0]
            masked_pred = self.model.predict_proba([masked_x])[0]
            
            # Compute similarity
            sufficiency = 1 - np.linalg.norm(original_pred - masked_pred)
            sufficiency_scores.append(sufficiency)
        
        results['sufficiency'] = np.mean(sufficiency_scores)
        
        # Comprehensiveness: How much does removing important features hurt?
        comprehensiveness_scores = []
        for i, (x, explanation) in enumerate(zip(X, explanations)):
            k = min(10, len(explanation))
            top_indices = np.argsort(np.abs(explanation))[-k:]
            
            # Create masked input without important features
            masked_x = x.copy()
            masked_x[top_indices] = 0
            
            original_pred = self.model.predict_proba([x])[0]
            masked_pred = self.model.predict_proba([masked_x])[0]
            
            comprehensiveness = np.linalg.norm(original_pred - masked_pred)
            comprehensiveness_scores.append(comprehensiveness)
        
        results['comprehensiveness'] = np.mean(comprehensiveness_scores)
        
        return results
    
    def stability_metrics(self, X: np.ndarray, num_perturbations: int = 10) -> Dict:
        """Compute stability metrics for explanations."""
        stability_scores = []
        
        for x in X:
            explanations = []
            
            # Generate multiple explanations with small perturbations
            for _ in range(num_perturbations):
                # Add small random noise
                noise = np.random.normal(0, 0.01, x.shape)
                perturbed_x = x + noise
                
                explanation = self.explanation_method(perturbed_x)
                explanations.append(explanation)
            
            # Compute pairwise correlations
            correlations = []
            for i in range(len(explanations)):
                for j in range(i + 1, len(explanations)):
                    corr, _ = stats.pearsonr(explanations[i], explanations[j])
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            stability_scores.append(np.mean(correlations) if correlations else 0)
        
        return {
            'stability': np.mean(stability_scores),
            'stability_std': np.std(stability_scores)
        }
    
    def monotonicity_test(self, X: np.ndarray, explanations: List[np.ndarray]) -> Dict:
        """Test monotonicity of explanations."""
        monotonicity_scores = []
        
        for x, explanation in zip(X, explanations):
            # Sort features by importance
            sorted_indices = np.argsort(np.abs(explanation))[::-1]
            
            # Incrementally add features and check prediction confidence
            confidences = []
            masked_x = np.zeros_like(x)
            
            for i, idx in enumerate(sorted_indices):
                masked_x[idx] = x[idx]
                pred_proba = self.model.predict_proba([masked_x])[0]
                confidence = np.max(pred_proba)
                confidences.append(confidence)
            
            # Check if confidences are monotonically increasing
            monotonic = all(confidences[i] <= confidences[i+1] 
                          for i in range(len(confidences)-1))
            monotonicity_scores.append(1.0 if monotonic else 0.0)
        
        return {
            'monotonicity': np.mean(monotonicity_scores)
        }
    
    def infidelity_metric(self, X: np.ndarray, explanations: List[np.ndarray], 
                         num_samples: int = 100) -> float:
        """Compute infidelity metric."""
        infidelity_scores = []
        
        for x, explanation in zip(X, explanations):
            scores = []
            
            for _ in range(num_samples):
                # Generate random perturbation
                perturbation = np.random.normal(0, 0.1, x.shape)
                perturbed_x = x + perturbation
                
                # Compute model difference
                original_pred = self.model.predict_proba([x])[0]
                perturbed_pred = self.model.predict_proba([perturbed_x])[0]
                model_diff = np.linalg.norm(original_pred - perturbed_pred)
                
                # Compute explanation difference
                explanation_diff = np.dot(explanation, perturbation)
                
                # Compute squared difference
                score = (model_diff - explanation_diff) ** 2
                scores.append(score)
            
            infidelity_scores.append(np.mean(scores))
        
        return np.mean(infidelity_scores)
```

### 4.2 Human Evaluation Framework

```python
# File: src/human_evaluation.py
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import random
from datetime import datetime

class HumanEvaluationFramework:
    """Framework for conducting human evaluation of XAI methods."""
    
    def __init__(self, output_dir: str = 'outputs/human_evaluation'):
        self.output_dir = output_dir
        self.evaluation_data = []
    
    def create_evaluation_task(self, texts: List[str], explanations: List[Dict], 
                              predictions: List[int], task_type: str = 'faithfulness') -> Dict:
        """Create human evaluation task."""
        task = {
            'task_id': f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'task_type': task_type,
            'instructions': self._get_instructions(task_type),
            'samples': []
        }
        
        # Create evaluation samples
        for i, (text, explanation, prediction) in enumerate(zip(texts, explanations, predictions)):
            sample = {
                'sample_id': i,
                'text': text,
                'prediction': prediction,
                'explanation': explanation,
                'questions': self._generate_questions(task_type)
            }
            task['samples'].append(sample)
        
        return task
    
    def _get_instructions(self, task_type: str) -> str:
        """Get instructions for different evaluation tasks."""
        instructions = {
            'faithfulness': """
            Please evaluate how well the explanation reflects the model's actual reasoning.
            Consider whether the highlighted words/features genuinely influence the prediction.
            """,
            'comprehensibility': """
            Please evaluate how easy it is to understand the explanation.
            Consider clarity, completeness, and cognitive load.
            """,
            'trust': """
            Please evaluate how much you trust the model's prediction based on the explanation.
            Consider whether the explanation increases or decreases your confidence.
            """,
            'actionability': """
            Please evaluate whether the explanation provides actionable insights.
            Consider whether you could use this explanation to improve the input.
            """
        }
        return instructions.get(task_type, "Please evaluate the explanation.")
    
    def _generate_questions(self, task_type: str) -> List[Dict]:
        """Generate evaluation questions based on task type."""
        questions = {
            'faithfulness': [
                {
                    'id': 'faithful_1',
                    'question': 'How well does this explanation reflect the model\'s reasoning?',
                    'type': 'likert',
                    'scale': [1, 2, 3, 4, 5],
                    'labels': ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
                },
                {
                    'id': 'faithful_2',
                    'question': 'Do the highlighted features seem relevant to the prediction?',
                    'type': 'likert',
                    'scale': [1, 2, 3, 4, 5],
                    'labels': ['Not at all', 'Slightly', 'Moderately', 'Very', 'Extremely']
                }
            ],
            'comprehensibility': [
                {
                    'id': 'comprehend_1',
                    'question': 'How easy is it to understand this explanation?',
                    'type': 'likert',
                    'scale': [1, 2, 3, 4, 5],
                    'labels': ['Very Difficult', 'Difficult', 'Neutral', 'Easy', 'Very Easy']
                },
                {
                    'id': 'comprehend_2',
                    'question': 'How complete is this explanation?',
                    'type': 'likert',
                    'scale': [1, 2, 3, 4, 5],
                    'labels': ['Very Incomplete', 'Incomplete', 'Adequate', 'Complete', 'Very Complete']
                }
            ]
        }
        return questions.get(task_type, [])
    
    def export_evaluation_task(self, task: Dict, format: str = 'json') -> str:
        """Export evaluation task for human annotators."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"{task['task_id']}.{format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(task, f, indent=2)
        elif format == 'csv':
            # Flatten task for CSV export
            rows = []
            for sample in task['samples']:
                for question in sample['questions']:
                    row = {
                        'task_id': task['task_id'],
                        'sample_id': sample['sample_id'],
                        'text': sample['text'],
                        'prediction': sample['prediction'],
                        'question_id': question['id'],
                        'question': question['question'],
                        'response': ''  # To be filled by annotators
                    }
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        return filepath
    
    def analyze_evaluation_results(self, results_file: str) -> Dict:
        """Analyze human evaluation results."""
        if results_file.endswith('.json'):
            with open(results_file, 'r') as f:
                results = json.load(f)
        elif results_file.endswith('.csv'):
            df = pd.read_csv(results_file)
            results = df.to_dict('records')
        
        analysis = {
            'overall_scores': {},
            'inter_annotator_agreement': {},
            'detailed_analysis': {}
        }
        
        # Compute overall scores by question type
        question_scores = {}
        for result in results:
            question_id = result.get('question_id', 'unknown')
            response = result.get('response', 0)
            
            if question_id not in question_scores:
                question_scores[question_id] = []
            question_scores[question_id].append(float(response))
        
        for question_id, scores in question_scores.items():
            analysis['overall_scores'][question_id] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'count': len(scores)
            }
        
        return analysis
```

## 🚀 Implementation Timeline

### Week 1-2: Foundation Setup
- [ ] Implement `AdvancedModelManager` with transformer support
- [ ] Create `EnhancedDatasetManager` with academic datasets
- [ ] Set up distributed computing infrastructure
- [ ] Implement basic ensemble methods

### Week 3-4: Model Integration
- [ ] Add domain-specific pre-trained models
- [ ] Implement few-shot learning framework
- [ ] Create model comparison utilities
- [ ] Add multilingual model support

### Week 5-6: Dataset Expansion
- [ ] Integrate Hugging Face datasets
- [ ] Add scientific domain datasets
- [ ] Implement cross-domain evaluation
- [ ] Create synthetic data generators

### Week 7-8: Advanced Datasets
- [ ] Add legal and financial datasets
- [ ] Implement multilingual datasets
- [ ] Create challenging evaluation sets
- [ ] Add temporal and adversarial datasets

### Week 9-10: XAI Enhancement
- [ ] Implement transformer-specific XAI
- [ ] Add attention visualization
- [ ] Create integrated gradients analysis
- [ ] Implement probing tasks

### Week 11-12: Causal XAI
- [ ] Add counterfactual explanations
- [ ] Implement causal mediation analysis
- [ ] Create instrumental variable methods
- [ ] Add causal graph analysis

### Week 13-14: Evaluation Framework
- [ ] Implement XAI-specific metrics
- [ ] Create faithfulness evaluation
- [ ] Add stability and robustness tests
- [ ] Implement human evaluation framework

### Week 15-16: Integration & Testing
- [ ] Integrate all components
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Documentation and examples

## 📊 Success Metrics

### Technical Metrics
- **Model Performance**: Achieve SOTA results on 5+ benchmark datasets
- **XAI Quality**: Score >0.8 on faithfulness and stability metrics
- **Scalability**: Process 10K+ samples efficiently
- **Coverage**: Support 20+ datasets and 10+ model types

### Academic Impact
- **Publications**: Submit to 2+ top-tier conferences
- **Reproducibility**: 100% reproducible experiments
- **Community Adoption**: 100+ GitHub stars, 10+ citations
- **Code Quality**: 90%+ test coverage, comprehensive documentation

This implementation plan provides a concrete roadmap to transform your XAI framework into a publication-worthy academic research platform.