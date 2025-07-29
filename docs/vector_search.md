# Vector Search in Schema Mapping

This document explains the vector search approaches used in the schema mapping system.

## Overview

The system uses vector search for:

- Finding similar fields
- Pattern discovery
- Semantic matching
- Context-aware mapping

## Vector Search Approaches

### 1. k-NN Search

```python
from sklearn.neighbors import NearestNeighbors

class VectorSearchHandler:
    def __init__(self, k: int = 3):
        self.knn_model = NearestNeighbors(
            n_neighbors=k,
            metric='cosine'
        )
```

**Features:**

- k-nearest neighbor search
- Cosine similarity metric
- Batch processing
- GPU acceleration support

### 2. Semantic Search

```python
def semantic_search(self, query: str, corpus: List[str], top_k: int = 5):
    # Generate query embedding
    query_embedding = self.model.encode(query)

    # Generate corpus embeddings
    corpus_embeddings = self.model.encode(corpus)

    # Calculate similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        corpus_embeddings
    )

    # Get top-k matches
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    return [(corpus[i], similarities[0][i]) for i in top_k_indices]
```

### 3. Pattern-Based Search

```python
def find_field_patterns(self, schema: Dict[str, str], k: int = 4):
    # Generate embeddings
    fields = list(schema.keys())
    embeddings = self.generate_embeddings(fields)

    # Initialize k-NN
    pattern_knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    pattern_knn.fit(embeddings)

    # Find patterns
    patterns = {}
    for i, field in enumerate(fields):
        distances, indices = pattern_knn.kneighbors(embeddings[i:i+1])
        similar_fields = [fields[idx] for idx in indices[0]]
        patterns[field] = similar_fields

    return patterns
```

## Search Optimization Techniques

### 1. Batch Processing

```python
def batch_search(self, queries: List[str], corpus: List[str], batch_size: int = 32):
    results = []

    # Process in batches
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_embeddings = self.model.encode(batch)
        batch_results = self.knn_model.kneighbors(batch_embeddings)
        results.extend(batch_results)

    return results
```

### 2. GPU Acceleration

```python
def initialize_gpu_search(self):
    if torch.cuda.is_available():
        self.model = self.model.to('cuda')
        self.use_gpu = True

    return self.use_gpu
```

### 3. Approximate Search

```python
from annoy import AnnoyIndex

class ApproximateVectorSearch:
    def __init__(self, vector_dim: int, n_trees: int = 10):
        self.index = AnnoyIndex(vector_dim, 'angular')
        self.n_trees = n_trees

    def build_index(self, vectors: List[np.ndarray]):
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        self.index.build(self.n_trees)
```

## Search Strategies

### 1. Exact Search (Small Datasets)

```python
def exact_search(self, query_vector: np.ndarray, corpus_vectors: np.ndarray):
    similarities = cosine_similarity(query_vector.reshape(1, -1), corpus_vectors)
    return similarities[0]
```

### 2. Approximate Search (Large Datasets)

```python
def approximate_search(self, query_vector: np.ndarray, n_neighbors: int = 10):
    return self.annoy_index.get_nns_by_vector(
        query_vector,
        n_neighbors,
        include_distances=True
    )
```

### 3. Hybrid Search

```python
def hybrid_search(self, query_vector: np.ndarray, corpus_vectors: np.ndarray):
    if len(corpus_vectors) < 1000:
        return self.exact_search(query_vector, corpus_vectors)
    else:
        return self.approximate_search(query_vector)
```

## Context-Aware Search

### 1. Context Enrichment

```python
def enrich_with_context(self, field: str, context: str) -> str:
    return f"{context} {field}"
```

### 2. Context-Aware Similarity

```python
def context_aware_similarity(self,
                           source_field: str,
                           target_field: str,
                           context: str) -> float:
    source_context = self.enrich_with_context(source_field, context)
    target_context = self.enrich_with_context(target_field, context)

    source_embedding = self.model.encode(source_context)
    target_embedding = self.model.encode(target_context)

    return cosine_similarity(
        source_embedding.reshape(1, -1),
        target_embedding.reshape(1, -1)
    )[0][0]
```

## Performance Considerations

### 1. Index Building

```python
def build_search_index(self, vectors: List[np.ndarray]):
    """Build search index for vectors"""
    if len(vectors) > 10000:
        # Use approximate search for large datasets
        self._build_approximate_index(vectors)
    else:
        # Use exact search for small datasets
        self._build_exact_index(vectors)
```

### 2. Memory Management

```python
def manage_vector_memory(self, vectors: List[np.ndarray]):
    """Manage memory for vector operations"""
    total_size = sum(v.nbytes for v in vectors)

    if total_size > 1e9:  # 1GB
        # Use disk-based storage
        return self._handle_large_vectors(vectors)
    else:
        # Keep in memory
        return vectors
```

### 3. Search Optimization

```python
def optimize_search(self, query: str, corpus: List[str]):
    """Optimize search based on corpus size"""
    if len(corpus) < 100:
        return self._simple_search(query, corpus)
    elif len(corpus) < 10000:
        return self._batch_search(query, corpus)
    else:
        return self._approximate_search(query, corpus)
```

## Best Practices

1. **Vector Preparation**

   - Normalize vectors
   - Use appropriate dimensions
   - Consider data type (float32/float16)

2. **Search Strategy Selection**

   - Use exact search for small datasets
   - Use approximate search for large datasets
   - Consider memory constraints

3. **Performance Optimization**

   - Implement batch processing
   - Use GPU when available
   - Optimize index parameters

4. **Context Handling**

   - Enrich vectors with context
   - Consider domain-specific information
   - Balance context weight

5. **Resource Management**
   - Monitor memory usage
   - Implement cleanup strategies
   - Use appropriate storage backends
