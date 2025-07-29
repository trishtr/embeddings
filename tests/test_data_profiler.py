import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.utils.data_profiler import DataProfiler, EnhancedDataProfiler, PostMappingProfiler
from src.db.db_handler import MultiSourceDBHandler
from sqlalchemy import create_engine

class TestDataProfiler:
    @pytest.fixture
    def mock_db_handler(self):
        mock = Mock()
        mock.connections = {
            'source1': Mock(),
            'target': Mock()
        }
        # Mock the engine and inspector
        mock_engine = create_engine('sqlite:///:memory:')
        mock.connections['source1'].engine = mock_engine
        return mock
    
    @pytest.fixture
    def sample_data(self):
        return [
            {"value": 1, "name": "John"},
            {"value": 2, "name": "Jane"},
            {"value": None, "name": None}
        ]
    
    def test_get_column_stats(self, mock_db_handler, sample_data):
        """Test column statistics calculation"""
        profiler = DataProfiler(mock_db_handler)
        
        # Test numeric column
        value_stats = profiler._get_column_stats(sample_data, "value")
        assert value_stats["count"] == 3
        assert value_stats["null_count"] == 1
        assert value_stats["unique_count"] == 2
        
        # Test string column
        name_stats = profiler._get_column_stats(sample_data, "name")
        assert name_stats["count"] == 3
        assert name_stats["null_count"] == 1
        assert name_stats["unique_count"] == 2
    
    def test_profile_database(self, mock_db_handler):
        """Test database profiling"""
        profiler = DataProfiler(mock_db_handler)
        
        # Mock database response
        mock_db_handler.connections['source1'].execute_query.return_value = [
            {"id": 1, "name": "Test"}
        ]
        
        # Mock SQLite dialect
        mock_dialect = Mock()
        mock_dialect.get_columns.return_value = [
            {"name": "id", "type": "INTEGER", "nullable": True, "default": None},
            {"name": "name", "type": "VARCHAR", "nullable": True, "default": None}
        ]
        
        # Mock connection
        mock_connection = Mock()
        mock_connection.dialect = mock_dialect
        
        # Mock inspector
        with patch('sqlalchemy.inspect') as mock_inspect:
            mock_inspector = Mock()
            mock_inspector.get_table_names.return_value = ['test_table']
            mock_inspector.get_columns.return_value = [
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"}
            ]
            mock_inspector.has_table.return_value = True
            mock_inspector.bind = mock_connection
            mock_inspect.return_value = mock_inspector
            
            profile = profiler.profile_database('source1', 'test_table')
            
            assert profile["table_name"] == "test_table"
            assert "schema" in profile
            assert "statistics" in profile
            assert "sample_data" in profile

class TestEnhancedDataProfiler:
    @pytest.fixture
    def mock_db_handler(self):
        return Mock()
    
    @pytest.fixture
    def mock_embedding_handler(self):
        return Mock()
    
    @pytest.fixture
    def enhanced_profiler(self, mock_db_handler, mock_embedding_handler):
        return EnhancedDataProfiler(mock_db_handler, mock_embedding_handler)
    
    def test_analyze_naming_conventions(self, enhanced_profiler):
        """Test naming convention analysis"""
        schema = {
            "user_id": "INTEGER",
            "userName": "VARCHAR",
            "UserType": "VARCHAR",
            "EMAIL": "VARCHAR"
        }
        
        conventions = enhanced_profiler._analyze_naming_conventions(schema)
        
        # Each field should be counted in exactly one convention category
        total_fields = len(schema)
        total_categorized = sum(conventions.values())
        assert total_fields == total_categorized
        
        # Check specific conventions
        assert conventions["snake_case"] == 1  # user_id
        assert conventions["camel_case"] == 1  # userName
        assert conventions["pascal_case"] == 1  # UserType
        assert conventions["uppercase"] == 1    # EMAIL
    
    def test_categorize_fields(self, enhanced_profiler):
        """Test field categorization"""
        schema = {
            "user_id": "INTEGER",
            "email_address": "VARCHAR",
            "phone_number": "VARCHAR",
            "created_date": "DATE"
        }
        
        categories = enhanced_profiler._categorize_fields(schema)
        
        assert "user_id" in categories["identifiers"]
        assert "email_address" in categories["contact_info"]
        assert "phone_number" in categories["contact_info"]
        assert "created_date" in categories["dates"]
    
    def test_calculate_field_similarity(self, enhanced_profiler):
        """Test field similarity calculation"""
        source_stats = {"type": "VARCHAR(100)"}
        target_stats = {"type": "TEXT"}
        
        similarity = enhanced_profiler._calculate_field_similarity(
            "user_name",
            "userName",
            source_stats,
            target_stats
        )
        
        assert 0 <= similarity <= 1

class TestPostMappingProfiler:
    @pytest.fixture
    def mock_db_handler(self):
        return Mock()
    
    @pytest.fixture
    def mock_schema_mapper(self):
        return Mock()
    
    @pytest.fixture
    def post_profiler(self, mock_db_handler, mock_schema_mapper):
        return PostMappingProfiler(mock_db_handler, mock_schema_mapper)
    
    def test_create_mapping_aware_comparison(self, post_profiler):
        """Test mapping-aware comparison creation"""
        source_profile = {
            "statistics": {
                "columns": {
                    "user_name": {"type": "VARCHAR", "count": 100},
                    "email": {"type": "VARCHAR", "count": 100}
                }
            }
        }
        
        target_profile = {
            "statistics": {
                "columns": {
                    "userName": {"type": "TEXT", "count": 100},
                    "email_address": {"type": "VARCHAR", "count": 100}
                }
            }
        }
        
        mapping = {
            "user_name": {
                "target_field": "userName",
                "confidence": 0.8
            },
            "email": {
                "target_field": "email_address",
                "confidence": 0.9
            }
        }
        
        comparison = post_profiler.create_mapping_aware_comparison(
            source_profile, target_profile, mapping
        )
        
        assert "mapped_fields" in comparison
        assert "unmapped_source_fields" in comparison
        assert "unmapped_target_fields" in comparison
        assert "data_compatibility" in comparison
    
    def test_calculate_compatibility(self, post_profiler):
        """Test compatibility calculation"""
        source_stats = {"type": "VARCHAR(100)"}
        target_stats = {"type": "TEXT"}
        
        compatibility = post_profiler._calculate_compatibility(source_stats, target_stats)
        
        assert 0 <= compatibility <= 1
    
    def test_are_types_compatible(self, post_profiler):
        """Test type compatibility checking"""
        # Test compatible types
        assert post_profiler._are_types_compatible("VARCHAR(100)", "TEXT")
        assert post_profiler._are_types_compatible("INTEGER", "INT")
        
        # Test incompatible types
        assert not post_profiler._are_types_compatible("VARCHAR(100)", "INTEGER")
        assert not post_profiler._are_types_compatible("DATE", "VARCHAR(100)")
    
    def test_are_ranges_compatible(self, post_profiler):
        """Test value range compatibility"""
        source_stats = {
            "min": 1,
            "max": 10
        }
        target_stats = {
            "min": 0,
            "max": 20
        }
        
        assert post_profiler._are_ranges_compatible(source_stats, target_stats)
        
        # Test incompatible ranges
        source_stats["max"] = 30
        assert not post_profiler._are_ranges_compatible(source_stats, target_stats)
    
    def test_are_distributions_compatible(self, post_profiler):
        """Test distribution compatibility"""
        source_stats = {
            "null_count": 10,
            "count": 100
        }
        target_stats = {
            "null_count": 12,
            "count": 100
        }
        
        assert post_profiler._are_distributions_compatible(source_stats, target_stats)

if __name__ == "__main__":
    pytest.main([__file__]) 