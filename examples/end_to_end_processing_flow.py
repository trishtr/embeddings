#!/usr/bin/env python3
"""
End-to-End Processing Flow for Schema Mapping System

This script demonstrates the complete workflow for schema mapping from multiple SQL sources
to a single target SQL database, specifically designed for healthcare medical provider data.

The flow includes:
1. Data Generation and Setup
2. Pre-Mapping Data Profiling
3. Context Creation with Business Rules
4. Embedding with Context
5. Schema Mapping with k-NN
6. Data Transformation
7. Post-Mapping Analysis
8. Quality Assessment and Reporting

Author: Schema Mapping System
Date: 2024
"""

import yaml
import os
import json
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
import time

# Import all our custom modules
from src.data.mock_data_generator import HealthcareDataGenerator
from src.embeddings.embedding_handler import EmbeddingHandler
from src.schema_mapping.schema_mapper import SchemaMapper
from src.schema_mapping.healthcare_mapper import HealthcareMapper
from src.db.db_handler import MultiSourceDBHandler
from src.utils.data_profiler import EnhancedDataProfiler, PostMappingProfiler
from src.utils.memory_cache import SchemaMappingWithInternalMemory
from src.utils.cloud_cache import HybridCloudCacheHandler
from src.utils.context_reporter import ContextReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/end_to_end_flow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EndToEndProcessingFlow:
    """
    Complete end-to-end processing flow for schema mapping system.
    """
    
    def __init__(self, config_path: str = 'config/db_config.yaml'):
        """Initialize the end-to-end processing flow."""
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.initialize_components()
        self.start_time = time.time()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_directories(self):
        """Create necessary directories for the workflow."""
        directories = [
            "data",
            "data/profiles", 
            "data/cache",
            "data/mappings",
            "data/reports",
            "logs",
            "exports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        # Initialize cache handlers
        self.cache_handlers = self._initialize_cache_handlers()
        
        # Initialize data generator
        self.generator = HealthcareDataGenerator()
        
        # Initialize embedding handler with optimizations
        self.embedding_handler = EmbeddingHandler(
            model_name=self.config['embedding']['model_name'],
            cache_dir=self.config['embedding']['cache_dir'],
            batch_size=self.config['performance']['batch_size'],
            use_gpu=self.config['performance']['use_gpu']
        )
        
        # Initialize database handler
        self.db_handler = MultiSourceDBHandler({
            name: db_config['connection_string']
            for name, db_config in self.config['databases'].items()
        })
        
        # Initialize schema mappers
        self.basic_mapper = SchemaMapper(self.embedding_handler)
        self.healthcare_mapper = HealthcareMapper(self.embedding_handler)
        
        # Initialize data profiler
        self.profiler = EnhancedDataProfiler(self.db_handler, self.embedding_handler)
        
        # Initialize context reporter
        self.context_reporter = ContextReporter()
        
        logger.info("All components initialized successfully")
    
    def _initialize_cache_handlers(self) -> Dict[str, Any]:
        """Initialize cache handlers based on configuration."""
        cache_handlers = {}
        
        # Initialize cloud cache if configured
        if 'cloud_cache' in self.config:
            cloud_config = self.config['cloud_cache']
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
    
    def step_1_data_generation_and_setup(self) -> Dict[str, Any]:
        """Step 1: Generate mock data and set up database tables."""
        logger.info("=" * 60)
        logger.info("STEP 1: DATA GENERATION AND SETUP")
        logger.info("=" * 60)
        
        setup_data = {}
        
        # Generate source schemas and data
        for source_name in ['source1', 'source2']:
            logger.info(f"Setting up {source_name}...")
            
            # Generate schema and data
            if source_name == 'source1':
                schema = self.generator.generate_source1_schema()
                data = self.generator.generate_source1_data(50)  # 50 records
            else:
                schema = self.generator.generate_source2_schema()
                data = self.generator.generate_source2_data(50)  # 50 records
            
            # Create and populate table
            table_name = self.config['databases'][source_name]['table_name']
            self.db_handler.connections[source_name].create_table(table_name, schema)
            self.db_handler.connections[source_name].insert_data(table_name, data)
            
            setup_data[source_name] = {
                'schema': schema,
                'data': data,
                'table_name': table_name
            }
            
            logger.info(f"{source_name}: Created table '{table_name}' with {len(data)} records")
        
        # Set up target schema
        logger.info("Setting up target schema...")
        target_schema = self.generator.generate_target_schema()
        target_table_name = self.config['databases']['target']['table_name']
        
        # Create target table (empty for now)
        self.db_handler.connections['target'].create_table(target_table_name, target_schema)
        
        setup_data['target'] = {
            'schema': target_schema,
            'table_name': target_table_name
        }
        
        logger.info(f"Target: Created table '{target_table_name}'")
        
        return setup_data
    
    def step_2_pre_mapping_data_profiling(self, setup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Perform comprehensive pre-mapping data profiling."""
        logger.info("=" * 60)
        logger.info("STEP 2: PRE-MAPPING DATA PROFILING")
        logger.info("=" * 60)
        
        profiling_results = {}
        
        # Profile each source independently
        for source_name in ['source1', 'source2']:
            logger.info(f"Profiling {source_name}...")
            
            profile = self.profiler.profile_source_independently(
                source_name,
                setup_data[source_name]['table_name']
            )
            
            # Export profile
            profile_filename = f"data/profiles/{source_name}_pre_mapping_profile.json"
            self.profiler.export_profile(profile, filename=profile_filename)
            
            profiling_results[source_name] = profile
            logger.info(f"{source_name} profiling completed - exported to {profile_filename}")
        
        # Profile target
        logger.info("Profiling target...")
        target_profile = self.profiler.profile_target_independently(
            'target',
            setup_data['target']['table_name']
        )
        
        target_profile_filename = "data/profiles/target_pre_mapping_profile.json"
        self.profiler.export_profile(target_profile, filename=target_profile_filename)
        
        profiling_results['target'] = target_profile
        logger.info(f"Target profiling completed - exported to {target_profile_filename}")
        
        # Analyze potential mappings
        logger.info("Analyzing potential mappings...")
        potential_mappings = self.profiler.analyze_potential_mappings(
            profiling_results['source1'], 
            profiling_results['target']
        )
        
        profiling_results['potential_mappings'] = potential_mappings
        
        # Assess mapping readiness
        readiness = self.profiler.assess_mapping_readiness(
            profiling_results['source1'], 
            profiling_results['target']
        )
        
        profiling_results['mapping_readiness'] = readiness
        
        logger.info(f"Mapping readiness score: {readiness['overall_score']:.2f}")
        if readiness['issues']:
            logger.warning(f"Mapping issues detected: {readiness['issues']}")
        
        return profiling_results
    
    def step_3_context_creation_with_business_rules(self, profiling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Create context with healthcare business rules and domain knowledge."""
        logger.info("=" * 60)
        logger.info("STEP 3: CONTEXT CREATION WITH BUSINESS RULES")
        logger.info("=" * 60)
        
        context_data = {}
        
        # Load healthcare rules
        logger.info("Loading healthcare business rules...")
        healthcare_rules = self.healthcare_mapper._load_rules()
        context_data['healthcare_rules'] = healthcare_rules
        
        # Identify PHI fields
        logger.info("Identifying PHI fields...")
        phi_fields = {}
        for source_name in ['source1', 'source2']:
            phi_fields[source_name] = self.healthcare_mapper._get_phi_fields(
                profiling_results[source_name]['schema']['columns']
            )
        
        context_data['phi_fields'] = phi_fields
        
        # Get domain contexts
        logger.info("Determining domain contexts...")
        domain_contexts = {}
        for source_name in ['source1', 'source2']:
            source_contexts = {}
            for column in profiling_results[source_name]['schema']['columns']:
                context = self.healthcare_mapper._get_domain_context(column['name'])
                source_contexts[column['name']] = context
            domain_contexts[source_name] = source_contexts
        
        context_data['domain_contexts'] = domain_contexts
        
        # Create enhanced field descriptions
        logger.info("Creating enhanced field descriptions...")
        enhanced_descriptions = {}
        for source_name in ['source1', 'source2']:
            enhanced = {}
            for column in profiling_results[source_name]['schema']['columns']:
                field_name = column['name']
                base_description = column.get('description', '')
                domain_context = domain_contexts[source_name][field_name]
                phi_info = "PHI" if field_name in phi_fields[source_name] else "Non-PHI"
                
                enhanced_description = f"{base_description} | Domain: {domain_context} | Classification: {phi_info}"
                enhanced[field_name] = enhanced_description
            
            enhanced_descriptions[source_name] = enhanced
        
        context_data['enhanced_descriptions'] = enhanced_descriptions
        
        # Export context data
        context_filename = "data/reports/context_creation_report.json"
        with open(context_filename, 'w') as f:
            json.dump(context_data, f, indent=2, default=str)
        
        logger.info(f"Context creation completed - exported to {context_filename}")
        
        return context_data
    
    def step_4_embedding_with_context(self, profiling_results: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Generate embeddings with enhanced context."""
        logger.info("=" * 60)
        logger.info("STEP 4: EMBEDDING WITH CONTEXT")
        logger.info("=" * 60)
        
        embedding_results = {}
        
        # Generate embeddings for each source with enhanced context
        for source_name in ['source1', 'source2']:
            logger.info(f"Generating embeddings for {source_name} with context...")
            
            # Get enhanced descriptions
            enhanced_descriptions = context_data['enhanced_descriptions'][source_name]
            
            # Create context-enhanced field list
            context_enhanced_fields = []
            for column in profiling_results[source_name]['schema']['columns']:
                field_name = column['name']
                enhanced_desc = enhanced_descriptions[field_name]
                context_enhanced_fields.append({
                    'name': field_name,
                    'type': column['type'],
                    'description': enhanced_desc,
                    'phi': field_name in context_data['phi_fields'][source_name]
                })
            
            # Generate embeddings
            embeddings = self.embedding_handler.generate_schema_embeddings(context_enhanced_fields)
            
            embedding_results[source_name] = {
                'fields': context_enhanced_fields,
                'embeddings': embeddings
            }
            
            logger.info(f"{source_name}: Generated embeddings for {len(context_enhanced_fields)} fields")
        
        # Generate target embeddings
        logger.info("Generating target embeddings...")
        target_fields = [
            {
                'name': column['name'],
                'type': column['type'],
                'description': column.get('description', '')
            }
            for column in profiling_results['target']['schema']['columns']
        ]
        
        target_embeddings = self.embedding_handler.generate_schema_embeddings(target_fields)
        
        embedding_results['target'] = {
            'fields': target_fields,
            'embeddings': target_embeddings
        }
        
        logger.info(f"Target: Generated embeddings for {len(target_fields)} fields")
        
        # Export embedding results
        embedding_filename = "data/reports/embedding_results.json"
        with open(embedding_filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in embedding_results.items():
                if 'embeddings' in value:
                    serializable_results[key] = {
                        'fields': value['fields'],
                        'embeddings_shape': value['embeddings'].shape if hasattr(value['embeddings'], 'shape') else 'numpy_array'
                    }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Embedding results exported to {embedding_filename}")
        
        return embedding_results
    
    def step_5_schema_mapping_with_knn(self, setup_data: Dict[str, Any], embedding_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Perform schema mapping using k-NN and advanced features."""
        logger.info("=" * 60)
        logger.info("STEP 5: SCHEMA MAPPING WITH K-NN")
        logger.info("=" * 60)
        
        mapping_results = {}
        
        # Map each source to target
        for source_name in ['source1', 'source2']:
            logger.info(f"Mapping {source_name} to target...")
            
            source_schema = setup_data[source_name]['schema']
            target_schema = setup_data['target']['schema']
            
            # Use healthcare mapper for domain-aware mapping
            mappings = self.healthcare_mapper.find_field_mappings(
                source_schema['columns'], 
                target_schema['columns']
            )
            
            # Find field patterns using k-NN
            patterns = self.healthcare_mapper.find_field_patterns(source_schema['columns'])
            
            # Find hierarchical mappings
            hierarchical_mappings = self.healthcare_mapper.find_hierarchical_mappings(
                source_schema['columns'], 
                target_schema['columns']
            )
            
            # Find context-aware mappings
            context_aware_mappings = self.healthcare_mapper.find_context_aware_mappings(
                source_schema['columns'], 
                target_schema['columns'],
                context="healthcare provider data"
            )
            
            mapping_results[source_name] = {
                'basic_mappings': mappings,
                'patterns': patterns,
                'hierarchical_mappings': hierarchical_mappings,
                'context_aware_mappings': context_aware_mappings
            }
            
            logger.info(f"{source_name}: Found {len(mappings)} basic mappings, {len(patterns)} patterns")
        
        # Export mapping results
        mapping_filename = "data/mappings/schema_mapping_results.json"
        with open(mapping_filename, 'w') as f:
            json.dump(mapping_results, f, indent=2, default=str)
        
        logger.info(f"Schema mapping completed - exported to {mapping_filename}")
        
        return mapping_results
    
    def step_6_data_transformation(self, setup_data: Dict[str, Any], mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Transform source data to target schema format."""
        logger.info("=" * 60)
        logger.info("STEP 6: DATA TRANSFORMATION")
        logger.info("=" * 60)
        
        transformation_results = {}
        
        # Transform each source data
        for source_name in ['source1', 'source2']:
            logger.info(f"Transforming {source_name} data...")
            
            source_data = setup_data[source_name]['data']
            mappings = mapping_results[source_name]['basic_mappings']
            
            # Transform data using healthcare mapper
            transformed_data = self.healthcare_mapper.transform_data(source_data, mappings)
            
            transformation_results[source_name] = {
                'original_count': len(source_data),
                'transformed_count': len(transformed_data),
                'transformed_data': transformed_data
            }
            
            logger.info(f"{source_name}: Transformed {len(source_data)} records to {len(transformed_data)} records")
        
        # Merge transformed data
        logger.info("Merging transformed data...")
        merged_data = []
        for source_name in ['source1', 'source2']:
            merged_data.extend(transformation_results[source_name]['transformed_data'])
        
        transformation_results['merged'] = {
            'total_records': len(merged_data),
            'merged_data': merged_data
        }
        
        logger.info(f"Merged data: {len(merged_data)} total records")
        
        # Write to target database
        logger.info("Writing merged data to target database...")
        target_table_name = setup_data['target']['table_name']
        self.db_handler.write_target_data(
            'target',
            target_table_name,
            setup_data['target']['schema'],
            merged_data
        )
        
        logger.info(f"Data written to target table '{target_table_name}'")
        
        return transformation_results
    
    def step_7_post_mapping_analysis(self, profiling_results: Dict[str, Any], mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Perform post-mapping analysis and comparison."""
        logger.info("=" * 60)
        logger.info("STEP 7: POST-MAPPING ANALYSIS")
        logger.info("=" * 60)
        
        post_mapping_results = {}
        
        # Create post-mapping profiler
        post_profiler = PostMappingProfiler(self.db_handler, self.healthcare_mapper)
        
        # Analyze each source mapping
        for source_name in ['source1', 'source2']:
            logger.info(f"Analyzing post-mapping results for {source_name}...")
            
            source_profile = profiling_results[source_name]
            target_profile = profiling_results['target']
            mappings = mapping_results[source_name]['basic_mappings']
            
            # Create mapping-aware comparison
            comparison = post_profiler.create_mapping_aware_comparison(
                source_profile, target_profile, mappings
            )
            
            post_mapping_results[source_name] = comparison
            
            logger.info(f"{source_name}: Post-mapping analysis completed")
        
        # Export post-mapping results
        post_mapping_filename = "data/reports/post_mapping_analysis.json"
        with open(post_mapping_filename, 'w') as f:
            json.dump(post_mapping_results, f, indent=2, default=str)
        
        logger.info(f"Post-mapping analysis completed - exported to {post_mapping_filename}")
        
        return post_mapping_results
    
    def step_8_quality_assessment_and_reporting(self, profiling_results: Dict[str, Any], 
                                              mapping_results: Dict[str, Any], 
                                              transformation_results: Dict[str, Any],
                                              post_mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Assess quality and generate comprehensive reports."""
        logger.info("=" * 60)
        logger.info("STEP 8: QUALITY ASSESSMENT AND REPORTING")
        logger.info("=" * 60)
        
        # Generate context-enhanced report
        logger.info("Generating context-enhanced report...")
        context_report = self.context_reporter.generate_context_report(
            profiling_results,
            mapping_results,
            transformation_results,
            post_mapping_results
        )
        
        # Calculate quality metrics
        quality_metrics = {
            'mapping_coverage': len(mapping_results['source1']['basic_mappings']) / len(profiling_results['source1']['schema']['columns']),
            'data_preservation': transformation_results['merged']['total_records'] / (transformation_results['source1']['original_count'] + transformation_results['source2']['original_count']),
            'schema_compatibility': post_mapping_results['source1']['compatibility_score'],
            'processing_time': time.time() - self.start_time
        }
        
        # Generate comprehensive final report
        final_report = {
            'execution_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_processing_time': quality_metrics['processing_time'],
                'steps_completed': 8
            },
            'quality_metrics': quality_metrics,
            'context_report': context_report,
            'mapping_summary': {
                'source1_mappings': len(mapping_results['source1']['basic_mappings']),
                'source2_mappings': len(mapping_results['source2']['basic_mappings']),
                'total_patterns_discovered': len(mapping_results['source1']['patterns']) + len(mapping_results['source2']['patterns'])
            },
            'data_summary': {
                'source1_records': transformation_results['source1']['original_count'],
                'source2_records': transformation_results['source2']['original_count'],
                'merged_records': transformation_results['merged']['total_records']
            }
        }
        
        # Export final report
        final_report_filename = "data/reports/final_comprehensive_report.json"
        with open(final_report_filename, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final comprehensive report exported to {final_report_filename}")
        
        # Print summary
        self._print_execution_summary(final_report)
        
        return final_report
    
    def _print_execution_summary(self, final_report: Dict[str, Any]):
        """Print a summary of the execution results."""
        print("\n" + "=" * 80)
        print("END-TO-END PROCESSING FLOW - EXECUTION SUMMARY")
        print("=" * 80)
        
        summary = final_report['execution_summary']
        metrics = final_report['quality_metrics']
        mapping_summary = final_report['mapping_summary']
        data_summary = final_report['data_summary']
        
        print(f"\n⏱️  Execution Time: {metrics['processing_time']:.2f} seconds")
        print(f"📅 Start Time: {summary['start_time']}")
        print(f"📅 End Time: {summary['end_time']}")
        
        print(f"\n📊 Data Processing Summary:")
        print(f"   • Source 1 Records: {data_summary['source1_records']}")
        print(f"   • Source 2 Records: {data_summary['source2_records']}")
        print(f"   • Merged Records: {data_summary['merged_records']}")
        
        print(f"\n🔗 Mapping Summary:")
        print(f"   • Source 1 Mappings: {mapping_summary['source1_mappings']}")
        print(f"   • Source 2 Mappings: {mapping_summary['source2_mappings']}")
        print(f"   • Patterns Discovered: {mapping_summary['total_patterns_discovered']}")
        
        print(f"\n📈 Quality Metrics:")
        print(f"   • Mapping Coverage: {metrics['mapping_coverage']:.2%}")
        print(f"   • Data Preservation: {metrics['data_preservation']:.2%}")
        print(f"   • Schema Compatibility: {metrics['schema_compatibility']:.2f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   • Profiles: data/profiles/")
        print(f"   • Mappings: data/mappings/")
        print(f"   • Reports: data/reports/")
        print(f"   • Logs: logs/end_to_end_flow.log")
        
        print("\n" + "=" * 80)
        print("END-TO-END PROCESSING FLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    
    def run_complete_flow(self) -> Dict[str, Any]:
        """Run the complete end-to-end processing flow."""
        logger.info("Starting End-to-End Processing Flow")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Generation and Setup
            setup_data = self.step_1_data_generation_and_setup()
            
            # Step 2: Pre-Mapping Data Profiling
            profiling_results = self.step_2_pre_mapping_data_profiling(setup_data)
            
            # Step 3: Context Creation with Business Rules
            context_data = self.step_3_context_creation_with_business_rules(profiling_results)
            
            # Step 4: Embedding with Context
            embedding_results = self.step_4_embedding_with_context(profiling_results, context_data)
            
            # Step 5: Schema Mapping with k-NN
            mapping_results = self.step_5_schema_mapping_with_knn(setup_data, embedding_results)
            
            # Step 6: Data Transformation
            transformation_results = self.step_6_data_transformation(setup_data, mapping_results)
            
            # Step 7: Post-Mapping Analysis
            post_mapping_results = self.step_7_post_mapping_analysis(profiling_results, mapping_results)
            
            # Step 8: Quality Assessment and Reporting
            final_report = self.step_8_quality_assessment_and_reporting(
                profiling_results, mapping_results, transformation_results, post_mapping_results
            )
            
            logger.info("End-to-End Processing Flow completed successfully!")
            return final_report
            
        except Exception as e:
            logger.error(f"Error in end-to-end flow: {e}")
            raise

def main():
    """Main function to run the end-to-end processing flow."""
    try:
        # Initialize the flow
        flow = EndToEndProcessingFlow()
        
        # Run the complete flow
        final_report = flow.run_complete_flow()
        
        print("\n🎉 End-to-End Processing Flow completed successfully!")
        print("Check the generated files in the data/ directory for detailed results.")
        
    except Exception as e:
        logger.error(f"Failed to run end-to-end flow: {e}")
        raise

if __name__ == "__main__":
    main() 