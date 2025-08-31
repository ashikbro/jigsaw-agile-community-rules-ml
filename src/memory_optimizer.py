#!/usr/bin/env python3
"""
Memory Optimization and Monitoring Utilities
Advanced memory management for large-scale ML training
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import logging
import time
from contextlib import contextmanager
import threading
from pathlib import Path
import joblib
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Advanced memory monitoring and alerting system
    """
    
    def __init__(self, 
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.9,
                 monitoring_interval: float = 1.0):
        
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread = None
        self.memory_history = []
        self.alerts = []
        
        logger.info(f"MemoryMonitor initialized: warning={warning_threshold*100:.1f}%, critical={critical_threshold*100:.1f}%")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'timestamp': time.time(),
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'used_percent': memory.percent,
            'free_gb': memory.free / (1024**3),
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
            'warning_level': self._get_warning_level(memory.percent / 100)
        }
    
    def _get_warning_level(self, usage_ratio: float) -> str:
        """Determine warning level based on memory usage"""
        if usage_ratio >= self.critical_threshold:
            return 'CRITICAL'
        elif usage_ratio >= self.warning_threshold:
            return 'WARNING'
        else:
            return 'NORMAL'
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                self.memory_history.append(memory_info)
                
                # Keep only last 1000 entries
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]
                
                # Check for alerts
                warning_level = memory_info['warning_level']
                if warning_level in ['WARNING', 'CRITICAL']:
                    alert = {
                        'timestamp': memory_info['timestamp'],
                        'level': warning_level,
                        'usage_percent': memory_info['used_percent'],
                        'available_gb': memory_info['available_gb']
                    }
                    self.alerts.append(alert)
                    
                    logger.warning(f"Memory {warning_level}: {memory_info['used_percent']:.1f}% used, {memory_info['available_gb']:.1f}GB available")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.memory_history:
            return self.get_memory_info()
        
        recent_usage = [entry['used_percent'] for entry in self.memory_history[-100:]]
        
        return {
            'current': self.get_memory_info(),
            'avg_usage_percent': np.mean(recent_usage),
            'max_usage_percent': np.max(recent_usage),
            'min_usage_percent': np.min(recent_usage),
            'total_alerts': len(self.alerts),
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 300])  # Last 5 minutes
        }
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for monitoring memory during operations"""
        start_info = self.get_memory_info()
        start_time = time.time()
        
        logger.info(f"Starting {operation_name} - Memory: {start_info['used_percent']:.1f}% used")
        
        try:
            yield self
        finally:
            end_info = self.get_memory_info()
            end_time = time.time()
            
            memory_change = end_info['used_gb'] - start_info['used_gb']
            duration = end_time - start_time
            
            logger.info(f"Completed {operation_name} in {duration:.1f}s - Memory change: {memory_change:+.2f}GB, Final usage: {end_info['used_percent']:.1f}%")

class MemoryOptimizer:
    """
    Advanced memory optimization utilities
    """
    
    def __init__(self, aggressive_cleanup: bool = False):
        self.aggressive_cleanup = aggressive_cleanup
        self.cleanup_history = []
        
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """Perform memory cleanup"""
        before_info = psutil.virtual_memory()
        
        # Standard cleanup
        gc.collect()
        
        if self.aggressive_cleanup or force:
            # More aggressive cleanup
            for _ in range(3):
                gc.collect()
            
            # Clear numpy cache if available
            try:
                np.core._multiarray_umath._reload_guard = []
            except:
                pass
        
        after_info = psutil.virtual_memory()
        
        cleanup_result = {
            'timestamp': time.time(),
            'before_used_gb': before_info.used / (1024**3),
            'after_used_gb': after_info.used / (1024**3),
            'freed_gb': (before_info.used - after_info.used) / (1024**3),
            'before_percent': before_info.percent,
            'after_percent': after_info.percent
        }
        
        self.cleanup_history.append(cleanup_result)
        
        if cleanup_result['freed_gb'] > 0.1:  # Log if freed more than 100MB
            logger.info(f"Memory cleanup freed {cleanup_result['freed_gb']:.2f}GB ({cleanup_result['before_percent']:.1f}% -> {cleanup_result['after_percent']:.1f}%)")
        
        return cleanup_result
    
    def optimize_pandas_dataframe(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage"""
        if not inplace:
            df = df.copy()
        
        initial_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            
            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('string')
        
        final_memory = df.memory_usage(deep=True).sum() / (1024**2)
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"DataFrame memory optimized: {initial_memory:.1f}MB -> {final_memory:.1f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    def optimize_numpy_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimize numpy arrays memory usage"""
        optimized = {}
        total_saved = 0
        
        for name, arr in arrays.items():
            original_size = arr.nbytes / (1024**2)
            
            if arr.dtype == np.float64:
                # Try to downcast to float32
                arr_f32 = arr.astype(np.float32)
                if np.allclose(arr, arr_f32, rtol=1e-6):
                    optimized[name] = arr_f32
                    saved = original_size - arr_f32.nbytes / (1024**2)
                    total_saved += saved
                    logger.debug(f"Array '{name}' downcasted to float32, saved {saved:.1f}MB")
                else:
                    optimized[name] = arr
            elif arr.dtype == np.int64:
                # Try to downcast integers
                if arr.min() >= 0:
                    if arr.max() < 255:
                        optimized[name] = arr.astype(np.uint8)
                    elif arr.max() < 65535:
                        optimized[name] = arr.astype(np.uint16)
                    elif arr.max() < 4294967295:
                        optimized[name] = arr.astype(np.uint32)
                    else:
                        optimized[name] = arr
                else:
                    if arr.min() > -128 and arr.max() < 127:
                        optimized[name] = arr.astype(np.int8)
                    elif arr.min() > -32768 and arr.max() < 32767:
                        optimized[name] = arr.astype(np.int16)
                    elif arr.min() > -2147483648 and arr.max() < 2147483647:
                        optimized[name] = arr.astype(np.int32)
                    else:
                        optimized[name] = arr
                
                if optimized[name].dtype != arr.dtype:
                    saved = original_size - optimized[name].nbytes / (1024**2)
                    total_saved += saved
                    logger.debug(f"Array '{name}' downcasted from {arr.dtype} to {optimized[name].dtype}, saved {saved:.1f}MB")
            else:
                optimized[name] = arr
        
        if total_saved > 0.1:
            logger.info(f"Numpy arrays optimized, total saved: {total_saved:.1f}MB")
        
        return optimized
    
    def create_memory_efficient_cache(self, 
                                    cache_dir: str = 'data/cache',
                                    max_cache_size_gb: float = 2.0) -> 'MemoryEfficientCache':
        """Create a memory-efficient cache system"""
        return MemoryEfficientCache(cache_dir, max_cache_size_gb)

class MemoryEfficientCache:
    """
    Memory-efficient caching system with automatic cleanup
    """
    
    def __init__(self, cache_dir: str, max_cache_size_gb: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_gb * (1024**3)  # Convert to bytes
        self.cache_index = {}
        
        # Load existing cache index
        self._load_cache_index()
        
        logger.info(f"MemoryEfficientCache initialized: {cache_dir}, max_size={max_cache_size_gb}GB")
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / 'cache_index.json'
        if index_file.exists():
            import json
            with open(index_file, 'r') as f:
                self.cache_index = json.load(f)
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / 'cache_index.json'
        import json
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for cache_file in self.cache_dir.glob('*.joblib'):
            total_size += cache_file.stat().st_size
        return total_size
    
    def _cleanup_cache(self):
        """Remove old cache files to stay under size limit"""
        current_size = self._get_cache_size()
        
        if current_size <= self.max_cache_size:
            return
        
        logger.info(f"Cache size ({current_size / (1024**3):.2f}GB) exceeds limit, cleaning up...")
        
        # Sort cache files by access time (oldest first)
        cache_files = []
        for key, info in self.cache_index.items():
            cache_file = self.cache_dir / f"{key}.joblib"
            if cache_file.exists():
                cache_files.append((cache_file, info.get('last_access', 0)))
        
        cache_files.sort(key=lambda x: x[1])  # Sort by access time
        
        # Remove oldest files until under limit
        for cache_file, _ in cache_files:
            if self._get_cache_size() <= self.max_cache_size * 0.8:  # Leave some buffer
                break
            
            cache_file.unlink()
            key = cache_file.stem
            if key in self.cache_index:
                del self.cache_index[key]
            
            logger.debug(f"Removed cache file: {cache_file}")
        
        self._save_cache_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        cache_file = self.cache_dir / f"{key}.joblib"
        
        if not cache_file.exists():
            return None
        
        try:
            data = joblib.load(cache_file)
            
            # Update access time
            self.cache_index[key] = {
                'last_access': time.time(),
                'size': cache_file.stat().st_size
            }
            self._save_cache_index()
            
            return data
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            return None
    
    def set(self, key: str, data: Any):
        """Set item in cache"""
        cache_file = self.cache_dir / f"{key}.joblib"
        
        try:
            joblib.dump(data, cache_file, compress=3)
            
            self.cache_index[key] = {
                'last_access': time.time(),
                'size': cache_file.stat().st_size
            }
            
            self._save_cache_index()
            self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error saving to cache file {cache_file}: {e}")
    
    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob('*.joblib'):
            cache_file.unlink()
        
        self.cache_index.clear()
        self._save_cache_index()
        
        logger.info("Cache cleared")

def memory_profile(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of functions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        with monitor.memory_context(func.__name__):
            result = func(*args, **kwargs)
        
        return result
    
    return wrapper

def auto_cleanup(threshold: float = 0.85) -> Callable:
    """
    Decorator to automatically cleanup memory when threshold is exceeded
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            memory = psutil.virtual_memory()
            if memory.percent / 100 > threshold:
                optimizer = MemoryOptimizer(aggressive_cleanup=True)
                optimizer.cleanup_memory(force=True)
                logger.info(f"Auto-cleanup triggered before {func.__name__}")
            
            result = func(*args, **kwargs)
            
            # Check memory after execution
            memory = psutil.virtual_memory()
            if memory.percent / 100 > threshold:
                optimizer = MemoryOptimizer(aggressive_cleanup=True)
                optimizer.cleanup_memory(force=True)
                logger.info(f"Auto-cleanup triggered after {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

class MemoryAwareDataLoader:
    """
    Data loader that adapts batch size based on available memory
    """
    
    def __init__(self, 
                 data: List[Any],
                 initial_batch_size: int = 1000,
                 memory_threshold: float = 0.8):
        
        self.data = data
        self.initial_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold
        self.current_batch_size = initial_batch_size
        self.position = 0
        
        logger.info(f"MemoryAwareDataLoader initialized: {len(data)} items, batch_size={initial_batch_size}")
    
    def _adjust_batch_size(self):
        """Adjust batch size based on current memory usage"""
        memory = psutil.virtual_memory()
        usage_ratio = memory.percent / 100
        
        if usage_ratio > self.memory_threshold:
            # Reduce batch size
            new_batch_size = max(100, int(self.current_batch_size * 0.7))
            if new_batch_size != self.current_batch_size:
                logger.info(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size} (memory: {usage_ratio*100:.1f}%)")
                self.current_batch_size = new_batch_size
        elif usage_ratio < self.memory_threshold * 0.6:
            # Increase batch size
            new_batch_size = min(self.initial_batch_size * 2, int(self.current_batch_size * 1.3))
            if new_batch_size != self.current_batch_size:
                logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size} (memory: {usage_ratio*100:.1f}%)")
                self.current_batch_size = new_batch_size
    
    def __iter__(self):
        self.position = 0
        return self
    
    def __next__(self):
        if self.position >= len(self.data):
            raise StopIteration
        
        self._adjust_batch_size()
        
        end_pos = min(self.position + self.current_batch_size, len(self.data))
        batch = self.data[self.position:end_pos]
        self.position = end_pos
        
        return batch
    
    def __len__(self):
        return (len(self.data) + self.current_batch_size - 1) // self.current_batch_size

if __name__ == "__main__":
    # Example usage
    monitor = MemoryMonitor()
    optimizer = MemoryOptimizer()
    
    # Test memory monitoring
    with monitor.memory_context("test_operation"):
        # Simulate some memory usage
        data = np.random.random((1000, 1000))
        time.sleep(1)
    
    # Test memory cleanup
    cleanup_result = optimizer.cleanup_memory()
    print(f"Memory cleanup result: {cleanup_result}")
    
    # Test memory-aware data loader
    sample_data = list(range(10000))
    loader = MemoryAwareDataLoader(sample_data, initial_batch_size=1000)
    
    batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count > 3:  # Just test a few batches
            break
    
    print(f"Processed {batch_count} batches with adaptive sizing")
    print("Memory optimization utilities test completed!")