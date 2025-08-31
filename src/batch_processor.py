#!/usr/bin/env python3
"""
Batch Processing Utility for Large Datasets
Handles memory-efficient processing of large datasets with parallel capabilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Iterator, Callable, Optional, Any
import gc
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from functools import partial
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MemoryEfficientBatchProcessor:
    """
    Memory-efficient batch processor for large datasets
    """
    
    def __init__(self, 
                 batch_size: Optional[int] = None,
                 memory_threshold: float = 0.85,
                 n_jobs: Optional[int] = None,
                 cache_dir: str = 'data/cache'):
        
        self.memory_threshold = memory_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-calculate optimal batch size and workers
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Limit to prevent memory issues
        
        logger.info(f"BatchProcessor initialized: batch_size={self.batch_size}, n_jobs={self.n_jobs}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Conservative batch sizes based on available memory
        if available_gb > 32:
            return 5000
        elif available_gb > 16:
            return 3000
        elif available_gb > 8:
            return 1500
        elif available_gb > 4:
            return 800
        else:
            return 400
    
    def _monitor_memory(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'warning': memory.percent > (self.memory_threshold * 100)
        }
    
    def _cleanup_memory(self):
        """Force garbage collection"""
        gc.collect()
    
    def batch_iterator(self, data: List[Any], batch_size: Optional[int] = None) -> Iterator[List[Any]]:
        """Create batches from data"""
        batch_size = batch_size or self.batch_size
        
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def process_text_batches(self, 
                           texts: List[str], 
                           processor_func: Callable,
                           **kwargs) -> List[Any]:
        """
        Process text data in batches with memory management
        """
        results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        
        for batch_idx, batch in enumerate(self.batch_iterator(texts)):
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Monitor memory before processing
            memory_info = self._monitor_memory()
            if memory_info['warning']:
                logger.warning(f"High memory usage: {memory_info['used_percent']:.1f}%")
                self._cleanup_memory()
            
            # Process batch
            batch_results = processor_func(batch, **kwargs)
            results.extend(batch_results)
            
            # Cleanup after each batch
            if batch_idx % 5 == 0:  # Cleanup every 5 batches
                self._cleanup_memory()
        
        return results
    
    def parallel_feature_extraction(self, 
                                  texts: List[str], 
                                  vectorizer: TfidfVectorizer,
                                  fit_transform: bool = True) -> np.ndarray:
        """
        Extract features in parallel batches
        """
        logger.info(f"Extracting features for {len(texts)} texts")
        
        if len(texts) <= self.batch_size:
            # Small dataset, process directly
            if fit_transform:
                return vectorizer.fit_transform(texts)
            else:
                return vectorizer.transform(texts)
        
        # Large dataset, process in batches
        if fit_transform:
            # Fit on a sample first
            sample_size = min(self.batch_size, len(texts))
            sample_texts = texts[:sample_size]
            vectorizer.fit(sample_texts)
            logger.info(f"Vectorizer fitted on {sample_size} samples")
        
        # Transform in batches
        feature_batches = []
        
        for batch_idx, batch in enumerate(self.batch_iterator(texts)):
            logger.info(f"Transforming batch {batch_idx + 1}")
            
            batch_features = vectorizer.transform(batch)
            
            # Save batch to disk if memory is tight
            memory_info = self._monitor_memory()
            if memory_info['warning']:
                batch_path = self.cache_dir / f"features_batch_{batch_idx}.joblib"
                joblib.dump(batch_features, batch_path)
                feature_batches.append(batch_path)
                logger.info(f"Batch saved to disk: {batch_path}")
            else:
                feature_batches.append(batch_features)
            
            self._cleanup_memory()
        
        # Combine batches
        logger.info("Combining feature batches")
        return self._combine_feature_batches(feature_batches)
    
    def _combine_feature_batches(self, batches: List) -> np.ndarray:
        """Combine feature batches efficiently"""
        from scipy.sparse import vstack
        
        # Load batches from disk if needed
        loaded_batches = []
        for batch in batches:
            if isinstance(batch, Path):
                loaded_batch = joblib.load(batch)
                loaded_batches.append(loaded_batch)
                # Clean up temporary file
                batch.unlink()
            else:
                loaded_batches.append(batch)
        
        # Combine using sparse matrix operations
        combined = vstack(loaded_batches)
        
        # Cleanup
        del loaded_batches
        self._cleanup_memory()
        
        return combined
    
    def stratified_batch_split(self, 
                             X: List[str], 
                             y: np.ndarray, 
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """
        Perform stratified split with memory efficiency for large datasets
        """
        logger.info(f"Performing stratified split on {len(X)} samples")
        
        if len(X) <= 10000:  # Small dataset, use standard split
            return train_test_split(X, y, test_size=test_size, 
                                  random_state=random_state, stratify=y)
        
        # Large dataset, use batch-wise stratified sampling
        unique_labels = np.unique(y)
        train_indices = []
        test_indices = []
        
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            
            # Split indices for this label
            label_train, label_test = train_test_split(
                label_indices, test_size=test_size, random_state=random_state
            )
            
            train_indices.extend(label_train)
            test_indices.extend(label_test)
        
        # Shuffle indices
        np.random.seed(random_state)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Extract data
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        logger.info(f"Split completed: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def batch_model_training(self, 
                           X_train: List[str], 
                           y_train: np.ndarray,
                           model_class,
                           model_params: Dict,
                           incremental: bool = False) -> Any:
        """
        Train model in batches for large datasets
        """
        logger.info(f"Training model on {len(X_train)} samples")
        
        if not incremental or len(X_train) <= self.batch_size:
            # Standard training for small datasets or non-incremental models
            model = model_class(**model_params)
            
            # Extract features
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            
            X_features = self.parallel_feature_extraction(X_train, vectorizer, fit_transform=True)
            model.fit(X_features, y_train)
            
            return model, vectorizer
        
        # Incremental training for large datasets
        logger.info("Using incremental training for large dataset")
        
        # Initialize model
        model = model_class(**model_params)
        
        # Fit vectorizer on sample
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        sample_size = min(self.batch_size, len(X_train))
        vectorizer.fit(X_train[:sample_size])
        
        # Train in batches
        for batch_idx, (X_batch, y_batch) in enumerate(zip(
            self.batch_iterator(X_train), 
            self.batch_iterator(y_train.tolist()))):
            
            logger.info(f"Training batch {batch_idx + 1}")
            
            # Transform batch
            X_batch_features = vectorizer.transform(X_batch)
            y_batch = np.array(y_batch)
            
            # Incremental fit
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_batch_features, y_batch)
            else:
                # For non-incremental models, retrain on accumulated data
                if batch_idx == 0:
                    model.fit(X_batch_features, y_batch)
                else:
                    logger.warning("Model doesn't support incremental learning")
            
            self._cleanup_memory()
        
        return model, vectorizer
    
    def save_large_dataset(self, 
                          data: pd.DataFrame, 
                          filepath: str,
                          format: str = 'parquet',
                          compression: str = 'snappy') -> str:
        """
        Save large dataset efficiently
        """
        filepath = Path(filepath)
        
        logger.info(f"Saving dataset with {len(data)} rows to {filepath}")
        
        if format == 'parquet':
            data.to_parquet(filepath, compression=compression, index=False)
        elif format == 'csv':
            # Save in chunks for very large datasets
            if len(data) > 100000:
                chunk_size = 50000
                base_path = filepath.with_suffix('')
                
                for i, chunk in enumerate(self.batch_iterator(data.to_dict('records'), chunk_size)):
                    chunk_df = pd.DataFrame(chunk)
                    chunk_path = f"{base_path}_chunk_{i}.csv"
                    chunk_df.to_csv(chunk_path, index=False)
                    logger.info(f"Saved chunk {i + 1} to {chunk_path}")
            else:
                data.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved successfully to {filepath}")
        return str(filepath)
    
    def load_large_dataset(self, 
                          filepath: str,
                          format: str = 'parquet',
                          chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Load large dataset efficiently
        """
        filepath = Path(filepath)
        
        logger.info(f"Loading dataset from {filepath}")
        
        if format == 'parquet':
            return pd.read_parquet(filepath)
        elif format == 'csv':
            if chunksize:
                # Load in chunks
                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=chunksize):
                    chunks.append(chunk)
                    
                    # Memory check
                    memory_info = self._monitor_memory()
                    if memory_info['warning']:
                        logger.warning("High memory usage during loading")
                        self._cleanup_memory()
                
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        """
        logger.info("Optimizing DataFrame memory usage")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize string columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('string')
        
        # Convert boolean columns
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype('int8')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        return df

class ParallelDataProcessor:
    """
    Parallel data processing utilities
    """
    
    def __init__(self, n_jobs: Optional[int] = None):
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)
        logger.info(f"ParallelDataProcessor initialized with {self.n_jobs} workers")
    
    def parallel_text_preprocessing(self, 
                                  texts: List[str], 
                                  preprocess_func: Callable) -> List[str]:
        """
        Preprocess texts in parallel
        """
        logger.info(f"Preprocessing {len(texts)} texts in parallel")
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            processed_texts = list(executor.map(preprocess_func, texts))
        
        return processed_texts
    
    def parallel_feature_computation(self, 
                                   data: List[Any], 
                                   feature_func: Callable,
                                   **kwargs) -> List[Any]:
        """
        Compute features in parallel
        """
        logger.info(f"Computing features for {len(data)} items in parallel")
        
        # Use partial to pass additional arguments
        func_with_kwargs = partial(feature_func, **kwargs)
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            features = list(executor.map(func_with_kwargs, data))
        
        return features

def create_optimized_batch_processor(dataset_size: str = 'medium') -> MemoryEfficientBatchProcessor:
    """
    Factory function to create optimized batch processor based on dataset size
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if dataset_size == 'small':
        batch_size = 500
        memory_threshold = 0.8
    elif dataset_size == 'medium':
        batch_size = 1000 if available_gb > 8 else 500
        memory_threshold = 0.85
    elif dataset_size == 'large':
        if available_gb > 16:
            batch_size = 2000
        elif available_gb > 8:
            batch_size = 1000
        else:
            batch_size = 500
        memory_threshold = 0.9
    else:
        batch_size = 500
        memory_threshold = 0.8
    
    return MemoryEfficientBatchProcessor(
        batch_size=batch_size,
        memory_threshold=memory_threshold
    )

if __name__ == "__main__":
    # Example usage
    processor = create_optimized_batch_processor('large')
    
    # Example data
    sample_texts = [f"Sample text {i}" for i in range(1000)]
    sample_labels = np.random.randint(0, 2, 1000)
    
    # Test batch processing
    X_train, X_test, y_train, y_test = processor.stratified_batch_split(
        sample_texts, sample_labels
    )
    
    print(f"Split completed: train={len(X_train)}, test={len(X_test)}")
    print("Batch processor test completed successfully!")