import redis
import pickle
import logging
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Azure imports
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    from azure.cosmos import CosmosClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# AWS imports
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# GCP imports
try:
    from google.cloud import storage
    from google.cloud import redis_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class AzureRedisCacheHandler:
    def __init__(self, redis_connection_string: str):
        """Initialize Azure Cache for Redis"""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure libraries not available. Install with: pip install azure-storage-blob azure-identity azure-cosmos")
        
        self.redis_client = redis.from_url(
            redis_connection_string,
            decode_responses=False,  # Keep binary for embeddings
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    def cache_embedding(self, key: str, embedding: np.ndarray, ttl: int = 3600):
        """Cache embedding with TTL (Time To Live)"""
        try:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(embedding)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Failed to retrieve embedding: {e}")
        return None
    
    def cache_mapping(self, key: str, mapping: dict, ttl: int = 3600):
        """Cache schema mapping"""
        try:
            self.redis_client.setex(
                key,
                ttl,
                pickle.dumps(mapping)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache mapping: {e}")
            return False
    
    def get_mapping(self, key: str) -> Optional[dict]:
        """Retrieve cached mapping"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Failed to retrieve mapping: {e}")
        return None

class AzureBlobCacheHandler:
    def __init__(self, storage_account_url: str, container_name: str):
        """Initialize Azure Blob Storage cache"""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure libraries not available")
        
        # Use managed identity for authentication
        credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(
            account_url=storage_account_url,
            credential=credential
        )
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(container_name)
    
    def cache_embedding(self, key: str, embedding: np.ndarray):
        """Store embedding in Azure Blob Storage"""
        try:
            blob_name = f"embeddings/{key}.pkl"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload embedding data
            blob_client.upload_blob(
                pickle.dumps(embedding),
                overwrite=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding in blob: {e}")
            return False
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from Azure Blob Storage"""
        try:
            blob_name = f"embeddings/{key}.pkl"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download embedding data
            blob_data = blob_client.download_blob()
            return pickle.loads(blob_data.readall())
        except Exception as e:
            logger.error(f"Failed to retrieve embedding from blob: {e}")
        return None
    
    def cache_mapping_history(self, mapping_history: list):
        """Cache mapping history in blob storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"mappings/mapping_history_{timestamp}.json"
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload mapping history
            blob_client.upload_blob(
                json.dumps(mapping_history, default=str),
                overwrite=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cache mapping history: {e}")
            return False

class AzureCosmosCacheHandler:
    def __init__(self, cosmos_endpoint: str, database_name: str, container_name: str):
        """Initialize Azure Cosmos DB cache"""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure libraries not available")
        
        credential = DefaultAzureCredential()
        self.client = CosmosClient(cosmos_endpoint, credential)
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
    
    def cache_mapping_metadata(self, mapping_id: str, metadata: dict):
        """Cache schema mapping metadata"""
        try:
            document = {
                'id': mapping_id,
                'type': 'schema_mapping',
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.container.upsert_item(document)
            return True
        except Exception as e:
            logger.error(f"Failed to cache metadata: {e}")
            return False
    
    def get_mapping_metadata(self, mapping_id: str) -> Optional[dict]:
        """Retrieve mapping metadata"""
        try:
            response = self.container.read_item(
                item=mapping_id,
                partition_key=mapping_id
            )
            return response.get('metadata')
        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
        return None

class AWSCloudCacheHandler:
    def __init__(self, redis_endpoint: str = None, s3_bucket: str = None):
        """Initialize AWS cloud cache handlers"""
        if not AWS_AVAILABLE:
            raise ImportError("AWS libraries not available. Install with: pip install boto3")
        
        # Redis cache (ElastiCache)
        if redis_endpoint:
            self.redis_client = redis.Redis(
                host=redis_endpoint,
                port=6379,
                decode_responses=False
            )
        else:
            self.redis_client = None
        
        # S3 cache
        if s3_bucket:
            self.s3_client = boto3.client('s3')
            self.s3_bucket = s3_bucket
        else:
            self.s3_client = None
    
    def cache_embedding(self, key: str, embedding: np.ndarray, ttl: int = 3600):
        """Cache embedding using Redis and/or S3"""
        success = True
        
        # Cache in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(embedding))
            except Exception as e:
                logger.error(f"Failed to cache in Redis: {e}")
                success = False
        
        # Cache in S3 if available
        if self.s3_client:
            try:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"embeddings/{key}.pkl",
                    Body=pickle.dumps(embedding)
                )
            except Exception as e:
                logger.error(f"Failed to cache in S3: {e}")
                success = False
        
        return success
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return pickle.loads(cached_data)
            except Exception as e:
                logger.error(f"Failed to retrieve from Redis: {e}")
        
        # Try S3
        if self.s3_client:
            try:
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key=f"embeddings/{key}.pkl"
                )
                return pickle.loads(response['Body'].read())
            except Exception as e:
                logger.error(f"Failed to retrieve from S3: {e}")
        
        return None

class GCPCloudCacheHandler:
    def __init__(self, redis_endpoint: str = None, storage_bucket: str = None):
        """Initialize GCP cloud cache handlers"""
        if not GCP_AVAILABLE:
            raise ImportError("GCP libraries not available. Install with: pip install google-cloud-storage google-cloud-redis")
        
        # Redis cache (Cloud Memorystore)
        if redis_endpoint:
            self.redis_client = redis.Redis(
                host=redis_endpoint,
                port=6379
            )
        else:
            self.redis_client = None
        
        # Cloud Storage cache
        if storage_bucket:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(storage_bucket)
        else:
            self.storage_client = None
            self.bucket = None
    
    def cache_embedding(self, key: str, embedding: np.ndarray, ttl: int = 3600):
        """Cache embedding using Redis and/or Cloud Storage"""
        success = True
        
        # Cache in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(embedding))
            except Exception as e:
                logger.error(f"Failed to cache in Redis: {e}")
                success = False
        
        # Cache in Cloud Storage if available
        if self.bucket:
            try:
                blob = self.bucket.blob(f"embeddings/{key}.pkl")
                blob.upload_from_string(pickle.dumps(embedding))
            except Exception as e:
                logger.error(f"Failed to cache in Cloud Storage: {e}")
                success = False
        
        return success
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return pickle.loads(cached_data)
            except Exception as e:
                logger.error(f"Failed to retrieve from Redis: {e}")
        
        # Try Cloud Storage
        if self.bucket:
            try:
                blob = self.bucket.blob(f"embeddings/{key}.pkl")
                blob_data = blob.download_as_bytes()
                return pickle.loads(blob_data)
            except Exception as e:
                logger.error(f"Failed to retrieve from Cloud Storage: {e}")
        
        return None

class HybridCloudCacheHandler:
    def __init__(self, 
                 redis_url: str = None,
                 blob_storage_url: str = None,
                 cosmos_endpoint: str = None,
                 local_cache_dir: str = None):
        
        # Internal memory cache (fastest)
        self.memory_cache = {}
        
        # Redis cache (fast, shared)
        self.redis_handler = None
        if redis_url:
            try:
                self.redis_handler = AzureRedisCacheHandler(redis_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
        
        # Blob storage (persistent, scalable)
        self.blob_handler = None
        if blob_storage_url:
            try:
                self.blob_handler = AzureBlobCacheHandler(blob_storage_url, 'embedding-cache')
            except Exception as e:
                logger.warning(f"Failed to initialize Blob Storage: {e}")
        
        # Cosmos DB (metadata and relationships)
        self.cosmos_handler = None
        if cosmos_endpoint:
            try:
                self.cosmos_handler = AzureCosmosCacheHandler(cosmos_endpoint, 'schema-mappings', 'mappings')
            except Exception as e:
                logger.warning(f"Failed to initialize Cosmos DB: {e}")
        
        # Local disk cache (instance-specific)
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
    
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Multi-layer cache retrieval"""
        # 1. Memory cache (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 2. Redis cache (fast, shared)
        if self.redis_handler:
            embedding = self.redis_handler.get_embedding(key)
            if embedding is not None:
                self.memory_cache[key] = embedding
                return embedding
        
        # 3. Local disk cache (fast, instance-specific)
        if self.local_cache_dir:
            cache_file = self.local_cache_dir / f"{key}.pkl"
            if cache_file.exists():
                embedding = pickle.load(cache_file)
                self.memory_cache[key] = embedding
                return embedding
        
        # 4. Blob storage (persistent)
        if self.blob_handler:
            embedding = self.blob_handler.get_embedding(key)
            if embedding is not None:
                # Store in faster caches
                self.memory_cache[key] = embedding
                if self.redis_handler:
                    self.redis_handler.cache_embedding(key, embedding)
                return embedding
        
        return None
    
    def set_embedding(self, key: str, embedding: np.ndarray):
        """Multi-layer cache storage"""
        # Store in all available caches
        self.memory_cache[key] = embedding
        
        if self.redis_handler:
            self.redis_handler.cache_embedding(key, embedding)
        
        if self.local_cache_dir:
            cache_file = self.local_cache_dir / f"{key}.pkl"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(embedding, cache_file)
        
        if self.blob_handler:
            self.blob_handler.cache_embedding(key, embedding)
    
    def get_mapping(self, key: str) -> Optional[dict]:
        """Get mapping from cache"""
        if self.redis_handler:
            return self.redis_handler.get_mapping(key)
        return None
    
    def set_mapping(self, key: str, mapping: dict):
        """Set mapping in cache"""
        if self.redis_handler:
            self.redis_handler.cache_mapping(key, mapping)
        
        if self.cosmos_handler:
            self.cosmos_handler.cache_mapping_metadata(key, mapping)

if __name__ == "__main__":
    # Test cloud cache handlers
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test Azure handlers (if credentials available)
    if AZURE_AVAILABLE and os.getenv('AZURE_REDIS_CONNECTION_STRING'):
        print("Testing Azure Redis Cache...")
        azure_redis = AzureRedisCacheHandler(os.getenv('AZURE_REDIS_CONNECTION_STRING'))
        
        # Test embedding cache
        test_embedding = np.random.rand(384)
        azure_redis.cache_embedding("test_key", test_embedding)
        retrieved = azure_redis.get_embedding("test_key")
        print(f"Azure Redis test: {retrieved is not None}")
    
    # Test AWS handlers (if credentials available)
    if AWS_AVAILABLE and os.getenv('AWS_REDIS_ENDPOINT'):
        print("Testing AWS Cloud Cache...")
        aws_cache = AWSCloudCacheHandler(
            redis_endpoint=os.getenv('AWS_REDIS_ENDPOINT'),
            s3_bucket=os.getenv('AWS_S3_BUCKET')
        )
        
        test_embedding = np.random.rand(384)
        aws_cache.cache_embedding("test_key", test_embedding)
        retrieved = aws_cache.get_embedding("test_key")
        print(f"AWS cache test: {retrieved is not None}")
    
    print("Cloud cache handlers initialized successfully!") 