import pytest
from unittest.mock import Mock, patch
import numpy as np
from src.schema_mapping.schema_mapper import SchemaMapper

class TestSchemaMapper:
    @pytest.fixture
    def mock_embedding_handler(self):
        mock = Mock()
        mock.find_similar_fields_knn.return_value = [
            ("healthcare_provider_id", 0.95),
            ("doctor_id", 0.85)
        ]
        mock.find_field_patterns_knn.return_value = {
            "pattern_1": ["provider_id", "doctor_id"]
        }
        mock.find_hierarchical_matches_knn.return_value = {
            "parent": [("doctor_information", 0.9)],
            "direct": [("full_name", 0.85)]
        }
        mock.find_context_aware_matches_knn.return_value = [
            ("provider_id", 0.9),
            ("provider_name", 0.85)
        ]
        return mock
    
    @pytest.fixture
    def schema_mapper(self, mock_embedding_handler):
        return SchemaMapper(mock_embedding_handler)
    
    def test_find_field_mappings(self, schema_mapper):
        """Test finding field mappings"""
        source_schema = {
            "provider_id": "VARCHAR(50)",
            "doctor_name": "VARCHAR(100)"
        }
        target_schema = {
            "healthcare_provider_id": "VARCHAR(50)",
            "provider_name": "VARCHAR(100)"
        }
        
        mappings = schema_mapper.find_field_mappings(source_schema, target_schema)
        
        assert isinstance(mappings, list)
        assert len(mappings) > 0
        assert all(len(m) == 3 for m in mappings)  # (source, target, score)
        assert all(isinstance(score, float) for _, _, score in mappings)
    
    def test_find_field_patterns(self, schema_mapper):
        """Test finding field patterns"""
        schema = {
            "provider_id": "VARCHAR(50)",
            "doctor_id": "VARCHAR(50)",
            "patient_name": "VARCHAR(100)"
        }
        
        patterns = schema_mapper.find_field_patterns(schema)
        
        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        assert all(isinstance(fields, list) for fields in patterns.values())
    
    def test_find_hierarchical_mappings(self, schema_mapper):
        """Test finding hierarchical mappings"""
        source_schema = {
            "provider_details.name": "VARCHAR(100)",
            "provider_details.contact": "VARCHAR(100)"
        }
        target_schema = {
            "doctor_information.full_name": "VARCHAR(100)",
            "doctor_information.contact_info": "VARCHAR(100)"
        }
        
        mappings = schema_mapper.find_hierarchical_mappings(source_schema, target_schema)
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        for field, matches in mappings.items():
            assert '.' in field  # Hierarchical field
            assert isinstance(matches, dict)
            if 'parent' in matches:
                assert isinstance(matches['parent'], list)
                assert all(isinstance(score, float) for _, score in matches['parent'])
            if 'direct' in matches:
                assert isinstance(matches['direct'], list)
                assert all(isinstance(score, float) for _, score in matches['direct'])
    
    def test_find_context_aware_mappings(self, schema_mapper):
        """Test finding context-aware mappings"""
        source_schema = {
            "id": "VARCHAR(50)",
            "name": "VARCHAR(100)"
        }
        target_schema = {
            "provider_id": "VARCHAR(50)",
            "provider_name": "VARCHAR(100)"
        }
        context = "Medical Provider"
        
        mappings = schema_mapper.find_context_aware_mappings(
            source_schema,
            target_schema,
            context
        )
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        for field, matches in mappings.items():
            assert isinstance(matches, list)
            assert all(isinstance(score, float) for _, score in matches)
            assert all(0 <= score <= 1 for _, score in matches)
    
    def test_transform_data(self, schema_mapper):
        """Test data transformation"""
        source_data = [
            {"provider_id": "P123", "name": "John"},
            {"provider_id": "P456", "name": "Jane"}
        ]
        
        mapping = {
            "provider_id": {
                "target_field": "healthcare_provider_id",
                "confidence": 0.95
            },
            "name": {
                "target_field": "provider_name",
                "confidence": 0.90
            }
        }
        
        transformed = schema_mapper.transform_data(source_data, mapping)
        
        assert isinstance(transformed, list)
        assert len(transformed) == len(source_data)
        assert all("healthcare_provider_id" in record for record in transformed)
        assert all("provider_name" in record for record in transformed)
    
    def test_mapping_validation(self, schema_mapper):
        """Test mapping validation"""
        mapping = {
            "provider_id": {
                "target_field": "healthcare_provider_id",
                "confidence": 0.95
            },
            "name": {
                "target_field": "provider_name",
                "confidence": 0.60
            }
        }
        
        # Test with default threshold
        is_valid = schema_mapper.validate_mapping(mapping)
        assert isinstance(is_valid, bool)
        
        # Test with custom threshold
        is_valid_strict = schema_mapper.validate_mapping(mapping, threshold=0.9)
        assert isinstance(is_valid_strict, bool)
        assert not is_valid_strict  # Should fail due to low confidence mapping
    
    def test_export_import_mapping(self, schema_mapper, tmp_path):
        """Test mapping export and import"""
        mapping_history = [
            {
                "source_schema": {"field1": "VARCHAR"},
                "target_schema": {"field2": "VARCHAR"},
                "mappings": {"field1": {"target_field": "field2", "confidence": 0.9}},
                "threshold": 0.7
            }
        ]
        schema_mapper.mapping_history = mapping_history
        
        # Export mapping
        export_path = tmp_path / "mapping_history.json"
        schema_mapper.export_mapping(str(export_path))
        
        # Import mapping
        schema_mapper.mapping_history = []  # Clear history
        schema_mapper.import_mapping(str(export_path))
        
        assert len(schema_mapper.mapping_history) == len(mapping_history)
        assert schema_mapper.mapping_history[0]["mappings"] == mapping_history[0]["mappings"] 