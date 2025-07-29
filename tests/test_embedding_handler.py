import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.embeddings.embedding_handler import EmbeddingHandler

class TestEmbeddingHandler:
    @pytest.fixture
    def mock_model(self):
        mock = Mock()
        # Return a small embedding vector for testing
        mock.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        return mock
    
    @pytest.fixture
    def embedding_handler(self, mock_model):
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            handler = EmbeddingHandler(use_gpu=False)
            # Mock the k-NN model
            handler.knn_model = Mock()
            handler.knn_model.kneighbors.return_value = (
                np.array([[0.1, 0.2]]),  # distances
                np.array([[0, 1]])       # indices
            )
            return handler
    
    def test_initialization(self, embedding_handler):
        """Test embedding handler initialization"""
        assert embedding_handler is not None
        assert embedding_handler.batch_size == 32
        assert embedding_handler.k == 3
    
    def test_find_similar_fields_knn(self, embedding_handler):
        """Test k-NN search for similar fields"""
        target_schema = {
            "healthcare_provider_id": "VARCHAR(50)",
            "doctor_id": "VARCHAR(50)",
            "provider_number": "VARCHAR(50)"
        }
        
        # Mock target embeddings
        embedding_handler.target_fields = list(target_schema.keys())
        embedding_handler.target_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        similar_fields = embedding_handler.find_similar_fields_knn("provider_id", target_schema)
        
        assert len(similar_fields) == 2  # Based on mock kneighbors return value
        assert all(isinstance(score, float) for _, score in similar_fields)
        assert all(0 <= score <= 1 for _, score in similar_fields)
    
    def test_find_field_patterns_knn(self, embedding_handler):
        """Test pattern discovery using k-NN"""
        schema = {
            "provider_id": "VARCHAR(50)",
            "doctor_id": "VARCHAR(50)",
            "patient_id": "VARCHAR(50)",
            "appointment_date": "DATE"
        }
        
        # Mock the batch_encode method
        with patch.object(embedding_handler, '_batch_encode') as mock_encode:
            mock_encode.return_value = np.array([
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.7, 0.8, 0.9]
            ])
            
            # Mock clustering behavior
            with patch('sklearn.cluster.KMeans') as mock_kmeans:
                mock_cluster = Mock()
                mock_cluster.labels_ = np.array([0, 0, 0, 1])  # Group ID fields together
                mock_cluster.fit_predict.return_value = np.array([0, 0, 0, 1])
                mock_kmeans.return_value = mock_cluster
                
                patterns = embedding_handler.find_field_patterns_knn(schema)
                
                assert isinstance(patterns, dict)
                assert len(patterns) > 0
                
                # Check if ID fields are grouped together
                id_pattern = None
                for pattern_name, fields in patterns.items():
                    if all('id' in field.lower() for field in fields):
                        id_pattern = pattern_name
                        break
                
                assert id_pattern is not None
                assert len(patterns[id_pattern]) == 3  # All ID fields
    
    def test_find_hierarchical_matches_knn(self, embedding_handler):
        """Test hierarchical field matching"""
        target_schema = {
            "doctor_information.full_name": "VARCHAR(100)",
            "physician_data.name": "VARCHAR(100)",
            "provider_details.id": "VARCHAR(50)"
        }
        
        hierarchical = embedding_handler.find_hierarchical_matches_knn(
            "provider_details.name",
            target_schema
        )
        
        assert isinstance(hierarchical, dict)
        assert 'parent' in hierarchical or 'direct' in hierarchical
        
        if 'parent' in hierarchical:
            assert isinstance(hierarchical['parent'], list)
            assert len(hierarchical['parent']) > 0
    
    def test_find_context_aware_matches_knn(self, embedding_handler):
        """Test context-aware field matching"""
        target_schema = {
            "healthcare_provider_id": "VARCHAR(50)",
            "doctor_id": "VARCHAR(50)",
            "patient_id": "VARCHAR(50)"
        }
        
        matches = embedding_handler.find_context_aware_matches_knn(
            "id",
            target_schema,
            "Medical Provider"
        )
        
        assert isinstance(matches, list)
        assert len(matches) > 0
        assert all(isinstance(score, float) for _, score in matches)
        assert all(0 <= score <= 1 for _, score in matches)
    
    def test_batch_encode(self, embedding_handler):
        """Test batch encoding of texts"""
        texts = ["field1", "field2", "field3"]
        
        # Mock the encode method to return consistent size embeddings
        with patch.object(embedding_handler.model, 'encode') as mock_encode:
            mock_encode.return_value = np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ])
            
            embeddings = embedding_handler._batch_encode(texts)
            
            assert isinstance(embeddings, np.ndarray)
            assert len(embeddings) == len(texts)
            assert embeddings.shape[1] == 3  # Mock embedding dimension
    
    def test_prepare_target_embeddings(self, embedding_handler):
        """Test preparation of target embeddings"""
        target_schema = {
            "field1": "VARCHAR(50)",
            "field2": "VARCHAR(50)"
        }
        
        embedding_handler._prepare_target_embeddings(target_schema)
        
        assert embedding_handler.target_fields is not None
        assert len(embedding_handler.target_fields) == len(target_schema)
        assert embedding_handler.target_embeddings is not None
        assert len(embedding_handler.target_embeddings) == len(target_schema)
    
    @pytest.mark.parametrize("source_field,target_field,expected_range", [
        ("provider_id", "healthcare_provider_id", (0.7, 1.0)),
        ("random_field", "unrelated_field", (0.0, 0.7))
    ])
    def test_similarity_ranges(self, embedding_handler, source_field, target_field, expected_range):
        """Test similarity score ranges for different field pairs"""
        target_schema = {target_field: "VARCHAR(50)"}
        
        # Mock target embeddings
        embedding_handler.target_fields = [target_field]
        embedding_handler.target_embeddings = np.array([[0.1, 0.2, 0.3]])
        
        # Mock k-NN behavior for different cases
        if source_field == "provider_id":
            embedding_handler.knn_model.kneighbors.return_value = (
                np.array([[0.2]]),  # High similarity (low distance)
                np.array([[0]])
            )
        else:
            embedding_handler.knn_model.kneighbors.return_value = (
                np.array([[0.8]]),  # Low similarity (high distance)
                np.array([[0]])
            )
        
        similar_fields = embedding_handler.find_similar_fields_knn(source_field, target_schema)
        
        if similar_fields:
            score = similar_fields[0][1]
            assert expected_range[0] <= score <= expected_range[1] 