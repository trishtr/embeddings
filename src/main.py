import yaml
import os
from typing import Dict, Any
import logging
from pathlib import Path
import json
from datetime import datetime

from data.mock_data_generator import HealthcareDataGenerator
from embeddings.embedding_handler import EmbeddingHandler
from schema_mapping.schema_mapper import SchemaMapper
from db.db_handler import MultiSourceDBHandler
from utils.data_profiler import EnhancedDataProfiler, PostMappingProfiler
from utils.memory_cache import SchemaMappingWithInternalMemory
from utils.cloud_cache import HybridCloudCacheHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(config: Dict[str, Any]):
    """Create necessary directories"""
    Path("data").mkdir(exist_ok=True)
    Path(config['embedding']['cache_dir']).parent.mkdir(exist_ok=True)
    Path("data/profiles").mkdir(exist_ok=True)
    Path("data/cache").mkdir(exist_ok=True)

def export_mapping_report(mapping_data: Dict[str, Any], 
                        profile_data: Dict[str, Any],
                        output_path: str):
    """Export comprehensive mapping and profiling report"""
    report = {
        "mapping_summary": mapping_data,
        "data_profiles": profile_data,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Exported mapping report to {output_path}")

def initialize_cache_handlers(config: Dict[str, Any]):
    """Initialize cache handlers based on configuration"""
    cache_handlers = {}
    
    # Initialize cloud cache if configured
    if 'cloud_cache' in config:
        cloud_config = config['cloud_cache']
        try:
            cache_handlers['cloud'] = HybridCloudCacheHandler(
                redis_url=cloud_config.get('redis_url'),
                blob_storage_url=cloud_config.get('blob_storage_url'),
                cosmos_endpoint=cloud_config.get('cosmos_endpoint'),
                local_cache_dir=cloud_config.get('local_cache_dir', 'data/cache')
            )
            logger.info("Cloud cache handlers initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize cloud cache: {e}")
    
    # Initialize internal memory cache
    cache_handlers['internal'] = SchemaMappingWithInternalMemory()
    logger.info("Internal memory cache initialized")
    
    return cache_handlers

def demonstrate_knn_features(schema_mapper, source_profile, target_profile):
    """Demonstrate k-NN features for schema mapping"""
    
    print("\n🔍 Demonstrating k-NN Features")
    print("=" * 60)
    
    source_schema = source_profile["schema"]["columns"]
    target_schema = target_profile["schema"]["columns"]
    
    # 1. Basic k-NN Field Matching
    print("\n1. Basic k-NN Field Matching")
    print("-" * 40)
    
    mappings = schema_mapper.find_field_mappings(source_schema, target_schema)
    for source_field, target_field, score in mappings[:5]:  # Show top 5
        print(f"✓ {source_field} → {target_field} (similarity: {score:.3f})")
    
    # 2. Pattern Discovery
    print("\n2. Pattern Discovery")
    print("-" * 40)
    
    patterns = schema_mapper.find_field_patterns(source_schema)
    for pattern_name, fields in patterns.items():
        print(f"\nPattern: {pattern_name}")
        for field in fields:
            print(f"  - {field}")
    
    # 3. Hierarchical Mapping
    print("\n3. Hierarchical Mapping")
    print("-" * 40)
    
    # Add some hierarchical fields for demonstration
    hierarchical_source = {
        "provider_details.name": "VARCHAR(100)",
        "provider_details.contact": "VARCHAR(100)",
        **source_schema
    }
    
    hierarchical_target = {
        "doctor_information.full_name": "VARCHAR(100)",
        "doctor_information.contact_info": "VARCHAR(100)",
        **target_schema
    }
    
    hierarchical_mappings = schema_mapper.find_hierarchical_mappings(
        hierarchical_source,
        hierarchical_target
    )
    
    for source_field, matches in hierarchical_mappings.items():
        print(f"\nField: {source_field}")
        for level, level_matches in matches.items():
            print(f"  {level}:")
            for field, score in level_matches:
                print(f"    → {field} (similarity: {score:.3f})")
    
    # 4. Context-Aware Mapping
    print("\n4. Context-Aware Mapping")
    print("-" * 40)
    
    contexts = ["Medical Provider", "Patient", "Administrative"]
    for context in contexts:
        print(f"\nContext: {context}")
        context_mappings = schema_mapper.find_context_aware_mappings(
            source_schema,
            target_schema,
            context
        )
        
        # Show a few examples
        for source_field, matches in list(context_mappings.items())[:2]:
            print(f"\n  Field: {source_field}")
            for field, score in matches[:2]:  # Show top 2 matches
                print(f"    → {field} (similarity: {score:.3f})")
    
    return {
        "basic_mappings": mappings,
        "patterns": patterns,
        "hierarchical_mappings": hierarchical_mappings,
        "context_mappings": context_mappings
    }

def main():
    # Load configuration
    config = load_config('config/db_config.yaml')
    setup_directories(config)
    
    # Initialize cache handlers
    cache_handlers = initialize_cache_handlers(config)
    
    # Initialize components with optimizations
    generator = HealthcareDataGenerator()
    embedding_handler = EmbeddingHandler(
        model_name=config['embedding']['model_name'],
        cache_dir=config['embedding']['cache_dir'],
        batch_size=32,
        use_gpu=True
    )
    schema_mapper = SchemaMapper(embedding_handler)
    
    # Initialize database connections
    db_handler = MultiSourceDBHandler({
        name: db_config['connection_string']
        for name, db_config in config['databases'].items()
    })
    
    # Initialize data profiler
    profiler = EnhancedDataProfiler(db_handler, embedding_handler)
    
    # Process source1
    logger.info("Processing Source 1...")
    source1_schema = generator.generate_source1_schema()
    source1_data = generator.generate_source1_data(10)
    
    # Create and populate source1 table
    db_handler.connections['source1'].create_table(
        config['databases']['source1']['table_name'],
        source1_schema
    )
    db_handler.connections['source1'].insert_data(
        config['databases']['source1']['table_name'],
        source1_data
    )
    
    # Profile source1
    source1_profile = profiler.profile_source_independently(
        'source1',
        config['databases']['source1']['table_name']
    )
    profiler.export_profile(source1_profile)
    
    # Process source2
    logger.info("Processing Source 2...")
    source2_schema = generator.generate_source2_schema()
    source2_data = generator.generate_source2_data(10)
    
    # Create and populate source2 table
    db_handler.connections['source2'].create_table(
        config['databases']['source2']['table_name'],
        source2_schema
    )
    db_handler.connections['source2'].insert_data(
        config['databases']['source2']['table_name'],
        source2_data
    )
    
    # Profile source2
    source2_profile = profiler.profile_source_independently(
        'source2',
        config['databases']['source2']['table_name']
    )
    profiler.export_profile(source2_profile)
    
    # Define target schema
    target_schema = generator.generate_target_schema()
    
    # Profile target
    target_profile = profiler.profile_target_independently(
        'target',
        config['databases']['target']['table_name']
    )
    profiler.export_profile(target_profile)
    
    # Pre-mapping analysis
    logger.info("Performing pre-mapping analysis...")
    pre_mapping_analysis = profiler.analyze_potential_mappings(source1_profile, target_profile)
    
    # Check mapping readiness
    mapping_readiness = profiler.assess_mapping_readiness(source1_profile, target_profile)
    
    if mapping_readiness["overall_score"] > 0.7:
        logger.info("Data is ready for mapping")
        
        # Map schemas using internal memory cache
        logger.info("Mapping Source 1 schema...")
        source1_mapping = cache_handlers['internal'].map_schema(
            source1_schema, 
            target_schema, 
            embedding_handler
        )
        
        logger.info("Mapping Source 2 schema...")
        source2_mapping = cache_handlers['internal'].map_schema(
            source2_schema, 
            target_schema, 
            embedding_handler
        )
        
        # Transform and merge data
        transformed_data1 = schema_mapper.transform_data(source1_data, source1_mapping)
        transformed_data2 = schema_mapper.transform_data(source2_data, source2_mapping)
        merged_data = transformed_data1 + transformed_data2
        
        # Write to target database
        logger.info("Writing to target database...")
        db_handler.write_target_data(
            'target',
            config['databases']['target']['table_name'],
            target_schema,
            merged_data
        )
        
        # Create post-mapping comparison
        post_mapping_profiler = PostMappingProfiler(db_handler, schema_mapper)
        source1_comparison = post_mapping_profiler.create_mapping_aware_comparison(
            source1_profile, target_profile, source1_mapping
        )
        source2_comparison = post_mapping_profiler.create_mapping_aware_comparison(
            source2_profile, target_profile, source2_mapping
        )
        
        # Export comprehensive report
        mapping_report = {
            "source1": {
                "mapping": source1_mapping,
                "profile_comparison": source1_comparison
            },
            "source2": {
                "mapping": source2_mapping,
                "profile_comparison": source2_comparison
            }
        }
        
        profiles_report = {
            "source1": source1_profile,
            "source2": source2_profile,
            "target": target_profile,
            "pre_mapping_analysis": pre_mapping_analysis
        }
        
        export_mapping_report(
            mapping_report,
            profiles_report,
            "data/profiles/complete_mapping_report.json"
        )
        
        # Export mapping history if configured
        if config['mapping']['export_mappings']:
            schema_mapper.export_mapping(config['mapping']['mapping_history_file'])
            logger.info(f"Exported mapping history to {config['mapping']['mapping_history_file']}")
        
        # Cache statistics
        cache_stats = cache_handlers['internal'].get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        logger.info("Schema mapping and profiling completed successfully!")
        
        # Print sample mappings and profile insights
        print("\nSample mappings from Source 1:")
        for source_field, info in source1_mapping.items():
            print(f"\n{info['explanation']}")
        
        print("\nSample mappings from Source 2:")
        for source_field, info in source2_mapping.items():
            print(f"\n{info['explanation']}")
        
        print("\nProfile Comparison Highlights:")
        print("\nSource 1 vs Target:")
        print(f"Schema differences: {len(source1_comparison['schema_differences']['type_mismatches'])} type mismatches")
        print("\nSource 2 vs Target:")
        print(f"Schema differences: {len(source2_comparison['schema_differences']['type_mismatches'])} type mismatches")
        
        print(f"\nMapping Readiness Score: {mapping_readiness['overall_score']:.2f}")
        if mapping_readiness['issues']:
            print(f"Issues: {mapping_readiness['issues']}")
        if mapping_readiness['recommendations']:
            print(f"Recommendations: {mapping_readiness['recommendations']}")
        
        # Demonstrate k-NN features
        knn_results = demonstrate_knn_features(schema_mapper, source1_profile, target_profile)
        
        # Add k-NN results to the comprehensive report
        mapping_report["knn_analysis"] = {
            "basic_mappings": [
                {
                    "source_field": source,
                    "target_field": target,
                    "similarity": score
                }
                for source, target, score in knn_results["basic_mappings"]
            ],
            "discovered_patterns": knn_results["patterns"],
            "hierarchical_mappings": knn_results["hierarchical_mappings"]
        }
        
    else:
        logger.warning("Data needs improvement before mapping")
        logger.warning(f"Issues: {mapping_readiness['issues']}")
        logger.warning(f"Recommendations: {mapping_readiness['recommendations']}")
        
        # Export pre-mapping analysis for review
        pre_mapping_report = {
            "source1_profile": source1_profile,
            "source2_profile": source2_profile,
            "target_profile": target_profile,
            "pre_mapping_analysis": pre_mapping_analysis,
            "mapping_readiness": mapping_readiness
        }
        
        profiler.export_profile(
            pre_mapping_report,
            filename="pre_mapping_analysis_report.json"
        )

if __name__ == "__main__":
    main() 