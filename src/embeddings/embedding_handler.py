from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = 'data/embedding_cache',
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 k: int = 3):  # k for k-NN
        """Initialize with a sentence transformer model and performance settings"""
        self.model = SentenceTransformer(model_name)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.k = k  # Number of nearest neighbors to find
        
        # GPU optimization
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Using GPU for embeddings")
        else:
            logger.info("Using CPU for embeddings")
            
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize k-NN model
        self.knn_model = None
        self.target_fields = None
        self.target_embeddings = None
    
    @lru_cache(maxsize=1000)
    def _preprocess_field_name(self, field_name: str) -> str:
        """Convert field name to natural language description with caching"""
        return field_name.replace('_', ' ').lower()
    
    def _generate_field_context(self, field_name: str, field_type: str) -> str:
        """Generate context string for field embedding"""
        field_desc = self._preprocess_field_name(field_name)
        return f"database field {field_desc} of type {field_type}"
    
    def _get_cache_path(self, schema_id: str) -> Path:
        """Get cache file path for schema embeddings"""
        return self.cache_dir / f"embeddings_{schema_id}.joblib"
    
    def _compute_schema_hash(self, schema: Dict[str, str]) -> str:
        """Compute a unique identifier for schema"""
        schema_str = str(sorted(schema.items()))
        return joblib.hash(schema_str)
    
    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches for better performance"""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def generate_schema_embeddings(self, schema: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all fields in a schema with caching"""
        if self.cache_dir:
            schema_hash = self._compute_schema_hash(schema)
            cache_path = self._get_cache_path(schema_hash)
            
            # Try to load from cache
            if cache_path.exists():
                try:
                    return joblib.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
        
        # Generate contexts in parallel
        with ThreadPoolExecutor() as executor:
            contexts = list(executor.map(
                lambda x: self._generate_field_context(x[0], x[1]),
                schema.items()
            ))
        
        # Generate embeddings in batches
        all_embeddings = self._batch_encode(contexts)
        
        # Create embeddings dictionary
        embeddings = {
            field_name: embedding
            for (field_name, _), embedding in zip(schema.items(), all_embeddings)
        }
        
        # Cache results
        if self.cache_dir:
            try:
                joblib.dump(embeddings, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
        
        return embeddings
    
    def find_field_mappings(self, 
                           source_schema: Dict[str, str],
                           target_schema: Dict[str, str],
                           threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find matching fields between source and target schemas"""
        source_embeddings = self.generate_schema_embeddings(source_schema)
        target_embeddings = self.generate_schema_embeddings(target_schema)
        
        # Convert embeddings to matrices for batch computation
        source_matrix = np.stack(list(source_embeddings.values()))
        target_matrix = np.stack(list(target_embeddings.values()))
        
        # Compute similarity matrix in one go
        similarity_matrix = cosine_similarity(source_matrix, target_matrix)
        
        mappings = []
        source_fields = list(source_embeddings.keys())
        target_fields = list(target_embeddings.keys())
        
        for i, source_field in enumerate(source_fields):
            best_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i][best_idx]
            
            if best_score >= threshold:
                mappings.append((
                    source_field,
                    target_fields[best_idx],
                    float(best_score)
                ))
        
        return sorted(mappings, key=lambda x: x[2], reverse=True)
    
    def explain_mapping(self, source_field: str, target_field: str, score: float) -> str:
        """Generate explanation for why fields were mapped"""
        source_desc = self._preprocess_field_name(source_field)
        target_desc = self._preprocess_field_name(target_field)
        confidence = "high" if score > 0.8 else "moderate" if score > 0.7 else "low"
        
        return (f"Mapped '{source_desc}' to '{target_desc}' with {confidence} confidence "
                f"(similarity score: {score:.2f})")
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("embeddings_*.joblib"):
                cache_file.unlink()
            logger.info("Cleared embedding cache")

    def find_similar_fields_knn(self, 
                               source_field: str,
                               target_schema: Dict[str, str],
                               k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Find k most similar fields in target schema using k-NN"""
        k = k or self.k
        
        # Generate embedding for source field
        source_context = self._generate_field_context(source_field, "VARCHAR")  # Default type
        source_embedding = self.model.encode(source_context).reshape(1, -1)
        
        # Generate or get target embeddings
        if self.target_embeddings is None or self.target_fields is None:
            self._prepare_target_embeddings(target_schema)
        
        # Find k nearest neighbors
        distances, indices = self.knn_model.kneighbors(source_embedding)
        
        # Convert to list of (field, similarity) tuples
        similar_fields = [
            (self.target_fields[idx], 1 - dist)  # Convert distance to similarity
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return similar_fields
    
    def _prepare_target_embeddings(self, target_schema: Dict[str, str]):
        """Prepare target embeddings and initialize k-NN model"""
        # Generate embeddings for all target fields
        self.target_fields = list(target_schema.keys())
        target_contexts = [
            self._generate_field_context(field, field_type)
            for field, field_type in target_schema.items()
        ]
        
        # Generate embeddings in batches
        self.target_embeddings = self._batch_encode(target_contexts)
        
        # Initialize and fit k-NN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.k,
            metric='cosine'
        )
        self.knn_model.fit(self.target_embeddings)
    
    def find_field_patterns_knn(self, schema):
        """Find patterns in field names using k-NN clustering"""
        field_names = list(schema.keys())
        if not field_names:
            return {}
        
        # Generate embeddings for all fields
        embeddings = self._batch_encode(field_names)
        
        # Use k-means clustering to group similar fields
        from sklearn.cluster import KMeans
        n_clusters = min(len(field_names), 3)  # Use at most 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group fields by cluster
        patterns = {}
        for i in range(n_clusters):
            cluster_fields = [field_names[j] for j in range(len(field_names)) if labels[j] == i]
            if cluster_fields:
                # Name the pattern based on common characteristics
                pattern_name = self._get_pattern_name(cluster_fields)
                patterns[pattern_name] = cluster_fields
        
        return patterns
    
    def _get_pattern_name(self, fields):
        """Generate a descriptive name for a pattern based on field characteristics"""
        # Check for common substrings
        common_parts = set()
        for field in fields:
            parts = field.lower().replace('_', ' ').split()
            common_parts.update(parts)
        
        # Look for identifying characteristics
        if all('id' in field.lower() for field in fields):
            return "identifier_fields"
        elif all('date' in field.lower() or 'time' in field.lower() for field in fields):
            return "temporal_fields"
        elif all('name' in field.lower() for field in fields):
            return "name_fields"
        elif all('status' in field.lower() or 'type' in field.lower() for field in fields):
            return "category_fields"
        else:
            # Use most common word as pattern name
            most_common = max(common_parts, key=lambda x: sum(x in f.lower() for f in fields))
            return f"{most_common}_related_fields"
    
    def find_hierarchical_matches_knn(self,
                                    source_field: str,
                                    target_schema: Dict[str, str],
                                    k: int = 2) -> Dict[str, List[Tuple[str, float]]]:
        """Find hierarchical matches using k-NN"""
        # Split hierarchical field
        parts = source_field.split('.')
        results = {}
        
        if len(parts) > 1:
            # Find matches for parent
            parent = parts[0]
            parent_matches = self.find_similar_fields_knn(parent, target_schema, k)
            results['parent'] = parent_matches
            
            # Find matches for child within parent contexts
            child = parts[1]
            child_matches = []
            
            for parent_field, parent_score in parent_matches:
                # Look for child fields in target schema that match the parent context
                child_candidates = [
                    field for field in target_schema.keys()
                    if field.startswith(f"{parent_field}.")
                ]
                
                if child_candidates:
                    child_schema = {
                        field: target_schema[field]
                        for field in child_candidates
                    }
                    child_matches.extend(
                        self.find_similar_fields_knn(child, child_schema, k)
                    )
            
            results['child'] = sorted(child_matches, key=lambda x: x[1], reverse=True)[:k]
        else:
            # Single level field
            results['direct'] = self.find_similar_fields_knn(source_field, target_schema, k)
        
        return results
    
    def find_context_aware_matches_knn(self,
                                     source_field: str,
                                     target_schema: Dict[str, str],
                                     context: str,
                                     k: int = 3) -> List[Tuple[str, float]]:
        """Find matches considering context using k-NN"""
        # Create context-enriched field description
        context_field = f"{context} {source_field}"
        source_context = self._generate_field_context(context_field, "VARCHAR")
        source_embedding = self.model.encode(source_context).reshape(1, -1)
        
        # Generate context-aware target embeddings
        target_fields = list(target_schema.keys())
        target_contexts = [
            self._generate_field_context(f"{context} {field}", field_type)
            for field, field_type in target_schema.items()
        ]
        target_embeddings = self._batch_encode(target_contexts)
        
        # Initialize and fit k-NN model
        context_knn = NearestNeighbors(
            n_neighbors=k,
            metric='cosine'
        )
        context_knn.fit(target_embeddings)
        
        # Find nearest neighbors
        distances, indices = context_knn.kneighbors(source_embedding)
        
        # Convert to list of (field, similarity) tuples
        similar_fields = [
            (target_fields[idx], 1 - dist)
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return similar_fields

if __name__ == "__main__":
    # Test embedding handler with performance optimizations
    from ..data.mock_data_generator import HealthcareDataGenerator
    
    generator = HealthcareDataGenerator()
    source_schema = generator.generate_source1_schema()
    target_schema = generator.generate_target_schema()
    
    handler = EmbeddingHandler(
        cache_dir='data/embedding_cache',
        batch_size=32,
        use_gpu=True
    )
    
    # Warm up cache
    logger.info("Warming up embedding cache...")
    mappings = handler.find_field_mappings(source_schema, target_schema)
    
    # Test cached performance
    logger.info("Testing cached performance...")
    mappings = handler.find_field_mappings(source_schema, target_schema)
    
    print("\nSample Field Mappings:")
    for source, target, score in mappings[:3]:
        print(handler.explain_mapping(source, target, score)) 

    # Test k-NN functionality
    handler = EmbeddingHandler(use_gpu=False)
    
    # Example target schema
    target_schema = {
        "healthcare_provider_id": "VARCHAR(50)",
        "doctor_id": "VARCHAR(50)",
        "provider_number": "VARCHAR(50)",
        "patient_id": "VARCHAR(50)",
        "appointment_id": "VARCHAR(50)"
    }
    
    # Test similar fields
    similar = handler.find_similar_fields_knn("provider_id", target_schema)
    print("\nSimilar fields for 'provider_id':")
    for field, score in similar:
        print(f"- {field}: {score:.3f}")
    
    # Test pattern discovery
    patterns = handler.find_field_patterns_knn(target_schema)
    print("\nDiscovered patterns:")
    for pattern, fields in patterns.items():
        print(f"\n{pattern}:")
        for field in fields:
            print(f"- {field}")
    
    # Test hierarchical matching
    hierarchical = handler.find_hierarchical_matches_knn(
        "provider_details.name",
        {
            "doctor_information.full_name": "VARCHAR(100)",
            "physician_data.name": "VARCHAR(100)",
            "provider_details.id": "VARCHAR(50)"
        }
    )
    print("\nHierarchical matches:")
    for level, matches in hierarchical.items():
        print(f"\n{level}:")
        for field, score in matches:
            print(f"- {field}: {score:.3f}")
    
    # Test context-aware matching
    context_matches = handler.find_context_aware_matches_knn(
        "id",
        target_schema,
        "Medical Provider"
    )
    print("\nContext-aware matches for 'id' in Medical Provider context:")
    for field, score in context_matches:
        print(f"- {field}: {score:.3f}") 