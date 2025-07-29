#!/usr/bin/env python3
"""
Example usage of the enhanced schema mapping system with data profiling and caching.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.mock_data_generator import HealthcareDataGenerator
from embeddings.embedding_handler import EmbeddingHandler
from schema_mapping.schema_mapper import SchemaMapper
from db.db_handler import MultiSourceDBHandler
from utils.data_profiler import EnhancedDataProfiler, PostMappingProfiler
from utils.memory_cache import SchemaMappingWithInternalMemory
from utils.cloud_cache import HybridCloudCacheHandler
import yaml
import json

def main():
    """Demonstrate the enhanced schema mapping system"""
    
    print("🚀 Enhanced Schema Mapping System Demo")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    
    # Mock database connections for demo
    connections = {
        "source1": "sqlite:///data/source1.db",
        "source2": "sqlite:///data/source2.db",
        "target": "sqlite:///data/target.db"
    }
    
    db_handler = MultiSourceDBHandler(connections)
    generator = HealthcareDataGenerator()
    
    # Initialize embedding handler with optimizations
    embedding_handler = EmbeddingHandler(
        model_name='all-MiniLM-L6-v2',
        cache_dir='data/embedding_cache',
        batch_size=32,
        use_gpu=False  # Set to False for demo
    )
    
    # Initialize schema mapper
    schema_mapper = SchemaMapper(embedding_handler)
    
    # Initialize data profiler
    profiler = EnhancedDataProfiler(db_handler, embedding_handler)
    
    # Initialize cache handlers
    internal_cache = SchemaMappingWithInternalMemory()
    
    print("✅ Components initialized successfully")
    
    # 2. Generate mock data
    print("\n2. Generating mock data...")
    
    source1_schema = generator.generate_source1_schema()
    source1_data = generator.generate_source1_data(5)
    
    source2_schema = generator.generate_source2_schema()
    source2_data = generator.generate_source2_data(5)
    
    target_schema = generator.generate_target_schema()
    
    print(f"✅ Generated {len(source1_data)} records for source1")
    print(f"✅ Generated {len(source2_data)} records for source2")
    
    # 3. Create and populate databases
    print("\n3. Creating and populating databases...")
    
    # Create tables
    db_handler.connections['source1'].create_table("providers", source1_schema)
    db_handler.connections['source2'].create_table("physicians", source2_schema)
    db_handler.connections['target'].create_table("healthcare_providers", target_schema)
    
    # Insert data
    db_handler.connections['source1'].insert_data("providers", source1_data)
    db_handler.connections['source2'].insert_data("physicians", source2_data)
    
    print("✅ Databases created and populated")
    
    # 4. Data profiling
    print("\n4. Performing data profiling...")
    
    # Profile source1
    source1_profile = profiler.profile_source_independently('source1', 'providers')
    print(f"✅ Source1 profile: {source1_profile['schema']['column_count']} columns")
    
    # Profile source2
    source2_profile = profiler.profile_source_independently('source2', 'physicians')
    print(f"✅ Source2 profile: {source2_profile['schema']['column_count']} columns")
    
    # Profile target
    target_profile = profiler.profile_target_independently('target', 'healthcare_providers')
    print(f"✅ Target profile: {target_profile['schema']['column_count']} columns")
    
    # 5. Pre-mapping analysis
    print("\n5. Pre-mapping analysis...")
    
    pre_mapping_analysis = profiler.analyze_potential_mappings(source1_profile, target_profile)
    
    print("Potential mappings for source1:")
    for source_field, mapping_info in pre_mapping_analysis['potential_mappings'].items():
        best_match = mapping_info['best_match']
        if best_match:
            print(f"  {source_field} -> {best_match[0]} (confidence: {best_match[1]:.2f})")
    
    # 6. Mapping readiness assessment
    print("\n6. Mapping readiness assessment...")
    
    mapping_readiness = profiler.assess_mapping_readiness(source1_profile, target_profile)
    
    print(f"Mapping readiness score: {mapping_readiness['overall_score']:.2f}")
    if mapping_readiness['issues']:
        print("Issues found:")
        for issue in mapping_readiness['issues']:
            print(f"  - {issue}")
    
    if mapping_readiness['recommendations']:
        print("Recommendations:")
        for rec in mapping_readiness['recommendations']:
            print(f"  - {rec}")
    
    # 7. Schema mapping with caching
    print("\n7. Performing schema mapping with caching...")
    
    # Map source1 using internal cache
    source1_mapping = internal_cache.map_schema(source1_schema, target_schema, embedding_handler)
    
    print("Source1 mappings:")
    for source_field, mapping_info in source1_mapping.items():
        print(f"  {source_field} -> {mapping_info['target_field']} (confidence: {mapping_info['confidence']:.2f})")
    
    # Map source2 using internal cache
    source2_mapping = internal_cache.map_schema(source2_schema, target_schema, embedding_handler)
    
    print("Source2 mappings:")
    for source_field, mapping_info in source2_mapping.items():
        print(f"  {source_field} -> {mapping_info['target_field']} (confidence: {mapping_info['confidence']:.2f})")
    
    # 8. Data transformation
    print("\n8. Transforming data...")
    
    transformed_data1 = schema_mapper.transform_data(source1_data, source1_mapping)
    transformed_data2 = schema_mapper.transform_data(source2_data, source2_mapping)
    
    print(f"✅ Transformed {len(transformed_data1)} records from source1")
    print(f"✅ Transformed {len(transformed_data2)} records from source2")
    
    # 9. Post-mapping comparison
    print("\n9. Post-mapping comparison...")
    
    post_mapping_profiler = PostMappingProfiler(db_handler, schema_mapper)
    
    source1_comparison = post_mapping_profiler.create_mapping_aware_comparison(
        source1_profile, target_profile, source1_mapping
    )
    
    print(f"✅ Mapped fields: {len(source1_comparison['mapped_fields'])}")
    print(f"✅ Unmapped source fields: {len(source1_comparison['unmapped_source_fields'])}")
    print(f"✅ Unmapped target fields: {len(source1_comparison['unmapped_target_fields'])}")
    
    # 10. Cache statistics
    print("\n10. Cache statistics...")
    
    cache_stats = internal_cache.get_cache_stats()
    
    print("Embedding cache:")
    print(f"  Hit rate: {cache_stats['embedding_cache']['hit_rate']:.2f}")
    print(f"  Current size: {cache_stats['embedding_cache']['current_size']}")
    
    print("Mapping cache:")
    print(f"  Hit rate: {cache_stats['mapping_cache']['hit_rate']:.2f}")
    print(f"  Current size: {cache_stats['mapping_cache']['current_size']}")
    
    # 11. Export results
    print("\n11. Exporting results...")
    
    # Create comprehensive report
    report = {
        "source1": {
            "profile": source1_profile,
            "mapping": source1_mapping,
            "comparison": source1_comparison
        },
        "source2": {
            "profile": source2_profile,
            "mapping": source2_mapping
        },
        "target": {
            "profile": target_profile
        },
        "pre_mapping_analysis": pre_mapping_analysis,
        "mapping_readiness": mapping_readiness,
        "cache_statistics": cache_stats,
        "transformed_data_sample": {
            "source1": transformed_data1[0] if transformed_data1 else None,
            "source2": transformed_data2[0] if transformed_data2 else None
        }
    }
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    # Export report
    with open("data/comprehensive_mapping_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("✅ Comprehensive report exported to data/comprehensive_mapping_report.json")
    
    # 12. Summary
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nSummary:")
    print(f"  • Profiled {source1_profile['schema']['column_count'] + source2_profile['schema']['column_count']} source columns")
    print(f"  • Mapped {len(source1_mapping) + len(source2_mapping)} fields")
    print(f"  • Transformed {len(transformed_data1) + len(transformed_data2)} records")
    print(f"  • Cache hit rate: {cache_stats['embedding_cache']['hit_rate']:.2f}")
    print(f"  • Mapping readiness: {mapping_readiness['overall_score']:.2f}")
    
    print("\nFiles generated:")
    print("  • data/comprehensive_mapping_report.json")
    print("  • data/source1.db")
    print("  • data/source2.db")
    print("  • data/target.db")

if __name__ == "__main__":
    main() 