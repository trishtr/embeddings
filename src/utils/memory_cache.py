import threading
from typing import Dict, Any, Optional
from collections import OrderedDict
import time
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class InMemoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.timestamps[key] > self.ttl:
                    self._remove(key)
                    return None
                
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self._remove(key)
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Remove oldest if exceeds max size
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove(oldest_key)
    
    def _remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        return len(self.cache)
    
    def keys(self) -> list:
        return list(self.cache.keys())
    
    def exists(self, key: str) -> bool:
        return key in self.cache and not self._is_expired(key)
    
    def _is_expired(self, key: str) -> bool:
        return time.time() - self.timestamps[key] > self.ttl

class AdvancedInMemoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.access_count = {}
        self.lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Check expiration
                if self._is_expired(key):
                    self._remove(key)
                    self.stats['misses'] += 1
                    return None
                
                # Update access count
                self.access_count[key] += 1
                self.stats['hits'] += 1
                return self.cache[key]
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_least_used()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_count[key] = 0
            self.stats['sets'] += 1
    
    def _evict_least_used(self):
        """Evict least frequently used item"""
        if not self.cache:
            return
        
        # Find least used item
        least_used_key = min(self.access_count.keys(), 
                           key=lambda k: self.access_count[k])
        self._remove(least_used_key)
        self.stats['evictions'] += 1
    
    def _is_expired(self, key: str) -> bool:
        return time.time() - self.timestamps[key] > self.ttl
    
    def _remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            del self.access_count[key]
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        total_size = 0
        for key, value in self.cache.items():
            try:
                total_size += len(pickle.dumps(key)) + len(pickle.dumps(value))
            except:
                total_size += 100  # Rough estimate
        return total_size / (1024 * 1024)  # Convert to MB
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_count.clear()
    
    def keys(self) -> list:
        return list(self.cache.keys())
    
    def exists(self, key: str) -> bool:
        return key in self.cache and not self._is_expired(key)

class SchemaMappingWithInternalMemory:
    def __init__(self):
        # Internal memory caches
        self.embedding_cache = AdvancedInMemoryCache(max_size=10000, ttl=7200)  # 2 hours
        self.mapping_cache = AdvancedInMemoryCache(max_size=5000, ttl=3600)     # 1 hour
        self.schema_cache = AdvancedInMemoryCache(max_size=1000, ttl=1800)      # 30 minutes
        
        # Learning state
        self.field_patterns = {}
        self.confidence_history = {}
        
        # Statistics
        self.mapping_stats = {
            'total_mappings': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def get_embedding(self, field_name: str, model) -> np.ndarray:
        """Get embedding with internal memory cache"""
        cache_key = f"embedding:{field_name}"
        
        # Try memory cache first
        cached_embedding = self.embedding_cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Compute embedding
        embedding = model.encode(field_name)
        
        # Store in memory cache
        self.embedding_cache.set(cache_key, embedding)
        
        return embedding
    
    def map_schema(self, source_schema: dict, target_schema: dict, embedding_handler) -> dict:
        """Map schemas using internal memory"""
        # Generate cache key
        cache_key = self._generate_schema_key(source_schema, target_schema)
        
        # Try memory cache first
        cached_mapping = self.mapping_cache.get(cache_key)
        if cached_mapping is not None:
            self.mapping_stats['cache_hits'] += 1
            return cached_mapping
        
        # Compute mapping
        mapping = embedding_handler.find_field_mappings(source_schema, target_schema)
        
        # Convert to dictionary format
        mapping_dict = {}
        for source_field, target_field, confidence in mapping:
            mapping_dict[source_field] = {
                'target_field': target_field,
                'confidence': confidence
            }
        
        # Store in memory cache
        self.mapping_cache.set(cache_key, mapping_dict)
        self.mapping_stats['cache_misses'] += 1
        self.mapping_stats['total_mappings'] += 1
        
        # Update learning state
        self._update_learning_state(mapping_dict)
        
        return mapping_dict
    
    def _generate_schema_key(self, source_schema: dict, target_schema: dict) -> str:
        """Generate unique key for schema combination"""
        source_str = str(sorted(source_schema.items()))
        target_str = str(sorted(target_schema.items()))
        return f"mapping:{hash(source_str + target_str)}"
    
    def _update_learning_state(self, mapping: dict):
        """Update internal learning patterns"""
        for source_field, mapping_info in mapping.items():
            target_field = mapping_info['target_field']
            confidence = mapping_info['confidence']
            
            # Update field patterns
            if source_field not in self.field_patterns:
                self.field_patterns[source_field] = {}
            self.field_patterns[source_field][target_field] = confidence
            
            # Update confidence history
            if source_field not in self.confidence_history:
                self.confidence_history[source_field] = []
            self.confidence_history[source_field].append(confidence)
    
    def get_cache_stats(self) -> dict:
        """Get comprehensive cache statistics"""
        return {
            'embedding_cache': self.embedding_cache.get_stats(),
            'mapping_cache': self.mapping_cache.get_stats(),
            'schema_cache': self.schema_cache.get_stats(),
            'mapping_stats': self.mapping_stats,
            'learning_state': {
                'field_patterns_count': len(self.field_patterns),
                'confidence_history_count': len(self.confidence_history)
            }
        }
    
    def clear_all_caches(self):
        """Clear all internal caches"""
        self.embedding_cache.clear()
        self.mapping_cache.clear()
        self.schema_cache.clear()
        logger.info("All internal caches cleared")

if __name__ == "__main__":
    # Test in-memory cache
    import numpy as np
    
    # Test basic cache
    cache = InMemoryCache(max_size=5, ttl=60)
    cache.set("test1", "value1")
    cache.set("test2", "value2")
    
    print("Basic cache test:")
    print(f"Get test1: {cache.get('test1')}")
    print(f"Cache size: {cache.size()}")
    
    # Test advanced cache
    advanced_cache = AdvancedInMemoryCache(max_size=5, ttl=60)
    advanced_cache.set("test1", "value1")
    advanced_cache.set("test2", "value2")
    advanced_cache.get("test1")  # Access to increase count
    
    print("\nAdvanced cache test:")
    print(f"Get test1: {advanced_cache.get('test1')}")
    print(f"Stats: {advanced_cache.get_stats()}")
    
    # Test schema mapping cache
    schema_mapper = SchemaMappingWithInternalMemory()
    print(f"\nSchema mapping cache stats: {schema_mapper.get_cache_stats()}") 