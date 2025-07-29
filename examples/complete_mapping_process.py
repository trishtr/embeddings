#!/usr/bin/env python3

import json
from pathlib import Path
from src.embeddings.embedding_handler import EmbeddingHandler
from src.schema_mapping.healthcare_mapper import HealthcareSchemaMapper
from src.utils.data_profiler import EnhancedDataProfiler, PostMappingProfiler
from src.utils.context_reporter import ContextReporter

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def main():
    # Initialize components
    embedding_handler = EmbeddingHandler(use_gpu=False)
    healthcare_mapper = HealthcareSchemaMapper(
        embedding_handler,
        rules_file='config/healthcare_rules.yaml'
    )
    
    # Step 1: Define Source and Target Schemas
    print_section("Step 1: Schema Definition")
    source_schema = {
        "provider_npi": "VARCHAR(10)",
        "doctor_name": "VARCHAR(100)",
        "specialty_code": "VARCHAR(20)",
        "license_number": "VARCHAR(15)",
        "practice_address": "VARCHAR(200)"
    }

    target_schema = {
        "healthcare_provider_id": "VARCHAR(10)",
        "provider_full_name": "VARCHAR(100)",
        "provider_taxonomy": "VARCHAR(20)",
        "state_license": "VARCHAR(15)",
        "facility_location": "VARCHAR(200)"
    }

    print("Source Schema Fields:", list(source_schema.keys()))
    print("Target Schema Fields:", list(target_schema.keys()))

    # Step 2: Pre-Mapping Data Profiling
    print_section("Step 2: Pre-Mapping Data Profiling")
    data_profiler = EnhancedDataProfiler(None, embedding_handler)  # None for db_handler as we're using schema only
    
    # Profile source schema
    source_profile = data_profiler.profile_source_independently(source_schema)
    print("\nSource Schema Profile:")
    print(json.dumps(source_profile, indent=2))
    
    # Profile target schema
    target_profile = data_profiler.profile_target_independently(target_schema)
    print("\nTarget Schema Profile:")
    print(json.dumps(target_profile, indent=2))
    
    # Analyze potential mappings
    potential_mappings = data_profiler.analyze_potential_mappings(
        source_profile,
        target_profile
    )
    print("\nPotential Mapping Analysis:")
    print(json.dumps(potential_mappings, indent=2))

    # Step 3: Create Context with Business Rules and Domain
    print_section("Step 3: Context Creation")
    
    # Load healthcare rules
    healthcare_rules = healthcare_mapper.rules
    
    # Analyze domain context for each field
    source_contexts = {}
    for field in source_schema:
        context = healthcare_mapper._get_domain_context(field)
        rules = healthcare_mapper._get_business_rules(field, context)
        source_contexts[field] = {
            "domain": context,
            "business_rules": rules,
            "is_phi": field in healthcare_mapper.phi_fields
        }
    
    print("\nSource Field Contexts:")
    print(json.dumps(source_contexts, indent=2))
    
    # Get applicable business rules
    print("\nApplicable Business Rules:")
    for context_type, rules in healthcare_rules.get('mapping_rules', {}).items():
        print(f"\n{context_type}:")
        for rule in rules:
            print(f"- {rule.get('rule', '')}")

    # Step 4: Embedding with Context
    print_section("Step 4: Contextual Embedding")
    
    # Generate embeddings with context
    for field in source_schema:
        context = source_contexts[field]['domain']
        print(f"\nProcessing {field} with {context} context:")
        
        # Get context-specific rules
        context_rules = healthcare_rules.get('context_enhancement', {}).get(f'{context}_context', [])
        print("Context Enhancement Rules:")
        for rule in context_rules:
            if field in rule.get('fields', []):
                print(f"- {rule.get('rule', '')}")

    # Step 5: Apply Mapping
    print_section("Step 5: Mapping Application")
    
    # Generate mappings with context
    mappings = healthcare_mapper.find_field_mappings(source_schema, target_schema)
    
    print("\nGenerated Mappings:")
    for source, target, confidence in mappings:
        print(f"{source:15} -> {target:20} (confidence: {confidence:.2f})")
        
        # Show applied context and rules
        source_context = source_contexts[source]
        print(f"Domain: {source_context['domain']}")
        if source_context['business_rules']:
            print("Applied Rules:")
            for rule in source_context['business_rules']:
                print(f"- {rule}")

    # Step 6: Post-Mapping Data Profiling
    print_section("Step 6: Post-Mapping Analysis")
    
    # Convert mappings to dictionary format
    mapping_dict = {
        field: {
            "target_field": target,
            "confidence": score
        } for field, target, score in mappings
    }
    
    # Create post-mapping profiler
    post_profiler = PostMappingProfiler(None, healthcare_mapper)  # None for db_handler as we're using schema only
    
    # Generate post-mapping comparison
    post_mapping_analysis = post_profiler.create_mapping_aware_comparison(
        source_profile,
        target_profile,
        mapping_dict
    )
    
    print("\nPost-Mapping Analysis:")
    print(json.dumps(post_mapping_analysis, indent=2))

    # Generate final report
    print_section("Final Report")
    context_reporter = ContextReporter(healthcare_mapper)
    final_report = context_reporter.generate_context_report(
        source_schema,
        target_schema,
        mappings,
        post_mapping_analysis
    )

    # Save results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'complete_mapping_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nComplete analysis saved to: {report_path}")
    
    # Show summary metrics
    print("\nSummary Metrics:")
    metrics = final_report['quality_metrics']
    print(f"Overall Score: {metrics['overall_score']:.2f}")
    print("\nMetrics Breakdown:")
    for metric, details in metrics['metrics'].items():
        print(f"- {metric}: {details['score']:.2f}")
        print(f"  {details['details']}")
    
    print("\nKey Recommendations:")
    for rec in metrics['recommendations']:
        print(f"- {rec['field']}: {rec['suggestion']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    main() 