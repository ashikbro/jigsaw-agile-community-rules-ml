import os
import joblib
import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
import json

class ModelCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get_key(self, text: str, model_name: str) -> str:
        """Generate a unique cache key for the input text and model."""
        key_str = f"{model_name}:{text}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a cached prediction if it exists and is not expired."""
        key = self.get_key(text, model_name)
        if key not in self.cache:
            return None
            
        cached = self.cache[key]
        if datetime.now() - cached['timestamp'] > self.ttl:
            del self.cache[key]
            return None
            
        return cached['prediction']
    
    def set(self, text: str, model_name: str, prediction: Dict[str, Any]) -> None:
        """Cache a prediction."""
        if len(self.cache) >= self.max_size:
            # Remove the oldest item if cache is full
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        key = self.get_key(text, model_name)
        self.cache[key] = {
            'prediction': prediction,
            'timestamp': datetime.now()
        }
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.cache = ModelCache()
        
        # Load models on initialization
        self._discover_models()
        
        # Set default model to real dataset model if available
        if 'logistic_regression_real' in self.models:
            self._set_default_model('logistic_regression_real')
        elif 'simple_model' in self.models:
            self._set_default_model('simple_model')
    
    def _set_default_model(self, model_name: str) -> None:
        """Set the default model for predictions."""
        if model_name in self.models:
            # Create aliases for the default model
            self.models['simple_model'] = self.models[model_name]
            self.vectorizers['simple_model'] = self.vectorizers[model_name]
            self.model_metadata['simple_model'] = {
                **self.model_metadata.get(model_name, {}),
                'name': 'simple_model',
                'description': f'Default model (alias for {model_name})'
            }
            print(f"Set default model to: {model_name}")
    
    def _discover_models(self) -> None:
        """Discover and load all available models from the models directory."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return
        
        # First, load real dataset models (priority)
        real_models = {
            'logistic_regression_real': 'logistic_regression_real.pkl',
            'random_forest_real': 'random_forest_real.pkl'
        }
        
        for model_name, model_file in real_models.items():
            model_path = os.path.join(self.models_dir, model_file)
            vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer_real.pkl')
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.vectorizers[model_name] = joblib.load(vectorizer_path)
                    
                    self.model_metadata[model_name] = {
                        'name': model_name,
                        'loaded_at': datetime.now().isoformat(),
                        'description': f'Real dataset trained {model_name.replace("_real", "")} model',
                        'type': 'real_dataset',
                        'accuracy': '93.67%' if 'logistic' in model_name else '91.75%'
                    }
                    
                    print(f"Loaded real dataset model: {model_name}")
                except Exception as e:
                    print(f"Error loading real dataset model {model_name}: {str(e)}")
        
        # Then load legacy .joblib models
        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('.joblib'):
                model_name = model_file.replace('.joblib', '')
                if model_name.endswith('_vectorizer') or model_name == 'vectorizer':
                    continue
                    
                # Skip if we already loaded a real dataset version
                if f"{model_name}_real" in self.models:
                    continue
                    
                # Look for corresponding vectorizer with multiple naming patterns
                # Try model-specific vectorizer first, then fallback to generic
                vectorizer_patterns = [
                    f"{model_name}_vectorizer.joblib",
                    f"{model_name.replace('_model', '')}_vectorizer.joblib",
                    f"{model_name.replace('_multilingual', '')}_vectorizer.joblib",
                    "vectorizer.joblib"
                ]
                
                vectorizer_path = None
                for pattern in vectorizer_patterns:
                    potential_path = os.path.join(self.models_dir, pattern)
                    if os.path.exists(potential_path):
                        vectorizer_path = potential_path
                        print(f"Found vectorizer for {model_name}: {pattern}")
                        break
                
                metadata_file = f"{model_name}_metadata.json"
                
                if vectorizer_path:
                    try:
                        # Load the model and vectorizer
                        model_path = os.path.join(self.models_dir, model_file)
                        
                        self.models[model_name] = joblib.load(model_path)
                        self.vectorizers[model_name] = joblib.load(vectorizer_path)
                        
                        # Load metadata if available
                        metadata_path = os.path.join(self.models_dir, metadata_file)
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                self.model_metadata[model_name] = json.load(f)
                        else:
                            self.model_metadata[model_name] = {
                                'name': model_name,
                                'loaded_at': datetime.now().isoformat(),
                                'description': 'No description available'
                            }
                            
                        print(f"Loaded model: {model_name}")
                        
                    except Exception as e:
                        print(f"Error loading model {model_name}: {str(e)}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their metadata."""
        return [
            {
                'name': name,
                'loaded': name in self.models,
                'metadata': self.model_metadata.get(name, {})
            }
            for name in self.model_metadata.keys()
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_name not in self.models:
            return {
                'error': f"Model '{model_name}' not found",
                'exists': False
            }
            
        return {
            'name': model_name,
            'loaded': True,
            'metadata': self.model_metadata.get(model_name, {}),
            'cache_stats': {
                'size': len(self.cache.cache),
                'max_size': self.cache.max_size,
                'ttl_seconds': self.cache.ttl.total_seconds()
            }
        }
    
    def predict(self, text: str, model_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        Returns a dictionary with the prediction results.
        """
        # Check if model exists
        if model_name not in self.models:
            return {
                'error': f"Model '{model_name}' not found",
                'status': 'error'
            }
        
        # Try to get from cache first
        cache_key = self.cache.get_key(text, model_name)
        if use_cache and (cached_pred := self.cache.get(text, model_name)):
            return {
                **cached_pred,
                'cached': True,
                'status': 'success'
            }
        
        try:
            # Make prediction
            vectorizer = self.vectorizers[model_name]
            model = self.models[model_name]
            
            # Preprocess text before vectorization
            cleaned_text = self._preprocess_text(text)
            
            X = vectorizer.transform([cleaned_text])
            prediction = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0])
            
            result = {
                'prediction': 'positive' if prediction == 1 else 'negative',
                'confidence': float(confidence),
                'model_used': model_name,
                'cached': False,
                'status': 'success'
            }
            
            # Cache the result
            self.cache.set(text, model_name, result)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear the prediction cache."""
        self.cache.clear()
        return {
            'status': 'success',
            'message': 'Cache cleared',
            'cache_size': 0
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the prediction cache."""
        return {
            'size': len(self.cache.cache),
            'max_size': self.cache.max_size,
            'ttl_seconds': self.cache.ttl.total_seconds(),
            'status': 'success'
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before vectorization (same as training preprocessing)."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?\S+|www\.\S+', ' [URL] ', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', ' [USER] ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

# Create a global instance for easy access
model_manager = ModelManager()
