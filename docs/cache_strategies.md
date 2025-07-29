# Cache Strategies in Schema Mapping

This document explains the caching strategies implemented in the schema mapping system.

## Overview

The system implements a multi-layer caching strategy:

- Memory Cache (fastest, temporary)
- Redis Cache (shared, distributed)
- Blob Storage (persistent, scalable)

## Cache Layers

### 1. Memory Cache (In-Process)

```python
from src.utils.memory_cache import AdvancedInMemoryCache

cache = AdvancedInMemoryCache(max_size=10000, ttl=7200)
```

**Features:**

- Fastest access (microseconds)
- LRU eviction policy (least recently used - determine which items to remove when the cache reaches its capacity limit)
- Thread-safe operations
- Memory usage monitoring
- Cache statistics

**Best for:**

- Frequently accessed embeddings
- Recent mapping results
- Session-specific data
- Small to medium datasets (<1GB)

### 2. Redis Cache (Distributed)

```python
from src.utils.cloud_cache import AzureRedisCacheHandler

redis_cache = AzureRedisCacheHandler(redis_connection_string)
```

**Features:**

- Shared across instances
- Fast access (milliseconds)
- Automatic serialization
- TTL support
- Cluster support

**Best for:**

- Shared embeddings
- Cross-instance data
- Temporary persistence
- Medium datasets (1-10GB)

### 3. Blob Storage (Persistent)

```python
from src.utils.cloud_cache import AzureBlobCacheHandler

blob_cache = AzureBlobCacheHandler(storage_account_url, container_name)
```

**Features:**

- Permanent storage
- Large capacity
- Cost-effective
- Backup support
- Version control

**Best for:**

- Long-term storage
- Large datasets (>10GB)
- Backup and recovery
- Historical mappings

## Cache Strategy Decision Matrix

| Data Type  | Size         | Access Pattern | Recommended Cache |
| ---------- | ------------ | -------------- | ----------------- |
| Embeddings | Small (<1MB) | Frequent       | Memory Cache      |
| Embeddings | Medium       | Shared         | Redis Cache       |
| Embeddings | Large        | Infrequent     | Blob Storage      |
| Mappings   | Small        | Session        | Memory Cache      |
| Mappings   | Any          | Shared         | Redis Cache       |
| History    | Any          | Archive        | Blob Storage      |

## Hybrid Cache Implementation

```python
class HybridCacheHandler:
    def __init__(self,
                 redis_url: str = None,
                 blob_storage_url: str = None,
                 local_cache_dir: str = None):
        # Initialize cache layers
        self.memory_cache = AdvancedInMemoryCache()
        self.redis_cache = AzureRedisCacheHandler(redis_url) if redis_url else None
        self.blob_cache = AzureBlobCacheHandler(blob_storage_url) if blob_storage_url else None

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        # 1. Try memory cache (fastest)
        if result := self.memory_cache.get(key):
            return result

        # 2. Try Redis cache (shared)
        if self.redis_cache:
            if result := self.redis_cache.get_embedding(key):
                self.memory_cache.set(key, result)  # Cache for future
                return result

        # 3. Try blob storage (persistent)
        if self.blob_cache:
            if result := self.blob_cache.get_embedding(key):
                # Cache in faster layers
                self.memory_cache.set(key, result)
                if self.redis_cache:
                    self.redis_cache.cache_embedding(key, result)
                return result

        return None
```

## Cache Performance Optimization

### 1. Memory Cache Optimization

```python
@lru_cache(maxsize=1000)
def _preprocess_field_name(self, field_name: str) -> str:
    """Cache frequently used field name preprocessing"""
    return field_name.replace('_', ' ').lower()
```

### 2. Batch Operations

```python
def cache_embeddings_batch(self, embeddings: Dict[str, np.ndarray]):
    """Batch cache operations for better performance"""
    # Memory cache
    self.memory_cache.set_many(embeddings)

    # Redis cache (pipeline)
    if self.redis_cache:
        with self.redis_cache.pipeline() as pipe:
            for key, embedding in embeddings.items():
                pipe.setex(key, 3600, pickle.dumps(embedding))
            pipe.execute()
```

### 3. Cache Warming

```python
def warm_cache(self, schema: Dict[str, str]):
    """Pre-cache frequently used embeddings"""
    embeddings = {}
    for field_name in schema:
        embedding = self.generate_embedding(field_name)
        embeddings[field_name] = embedding

    self.cache_embeddings_batch(embeddings)
```

## Cloud Deployment Considerations

### Azure Deployment

```python
# App Service configuration
cache_handler = HybridCacheHandler(
    redis_url=os.getenv('AZURE_REDIS_CONNECTION_STRING'),
    blob_storage_url=os.getenv('AZURE_BLOB_STORAGE_URL'),
    local_cache_dir='/tmp/cache'
)
```

### AWS Deployment

```python
# Lambda configuration
cache_handler = HybridCacheHandler(
    redis_url=os.getenv('ELASTICACHE_ENDPOINT'),
    blob_storage_url=os.getenv('S3_BUCKET_URL'),
    local_cache_dir='/tmp'
)
```

## Cache Monitoring

### 1. Cache Statistics

```python
def get_cache_stats(self) -> Dict[str, Any]:
    return {
        "memory_cache": {
            "hit_rate": self.memory_cache.get_hit_rate(),
            "size": self.memory_cache.size(),
            "evictions": self.memory_cache.get_eviction_count()
        },
        "redis_cache": {
            "hit_rate": self.redis_cache.get_hit_rate(),
            "memory_usage": self.redis_cache.get_memory_usage()
        },
        "blob_cache": {
            "size": self.blob_cache.get_total_size(),
            "file_count": self.blob_cache.get_file_count()
        }
    }
```

### 2. Performance Metrics

```python
def monitor_cache_performance(self):
    return {
        "average_latency_ms": {
            "memory": self.memory_cache.get_average_latency(),
            "redis": self.redis_cache.get_average_latency(),
            "blob": self.blob_cache.get_average_latency()
        },
        "throughput": {
            "memory": self.memory_cache.get_operations_per_second(),
            "redis": self.redis_cache.get_operations_per_second(),
            "blob": self.blob_cache.get_operations_per_second()
        }
    }
```

## Best Practices

1. **Cache Hierarchy**

   - Use fastest cache first
   - Cascade through layers
   - Cache frequently accessed data in memory

2. **Cache Invalidation**

   - Use TTL for temporary data
   - Implement versioning
   - Clear cache on schema changes

3. **Performance Optimization**

   - Batch operations
   - Pre-warm cache
   - Monitor usage patterns

4. **Resource Management**

   - Set appropriate size limits
   - Implement eviction policies
   - Monitor memory usage

5. **Error Handling**
   - Graceful degradation
   - Fallback mechanisms
   - Error logging and monitoring
