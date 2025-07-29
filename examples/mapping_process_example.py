#!/usr/bin/env python3

import json
from pathlib import Path
from src.embeddings.embedding_handler import EmbeddingHandler
from src.schema_mapping.healthcare_mapper import HealthcareSchemaMapper
from src.utils.context_reporter import ContextReporter

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def main():
    # Setup
    print_section("Setup")
    embedding_handler = EmbeddingHandler(use_gpu=False)
    healthcare_mapper = HealthcareSchemaMapper(
        embedding_handler,
        rules_file='config/healthcare_rules.yaml'
    )
    
    # Sample schemas
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

    # Step 1: Basic Embedding-based Mapping
    print_section("Step 1: Basic Mapping")
    mappings = healthcare_mapper.find_field_mappings(source_schema, target_schema)
    print("\nInitial mappings from embedding similarity:")
    for source, target, confidence in mappings:
        print(f"{source:15} -> {target:20} (confidence: {confidence:.2f})")

    # Step 2: Apply Healthcare Rules
    print_section("Step 2: Healthcare Rules")
    mapping_dict = {
        field: {
            "target_field": target,
            "confidence": score,
            "source_type": source_schema[field],
            "target_type": target_schema[target]
        } for field, target, score in mappings
    }
    
    # Check domain rules
    for source_field, mapping_info in mapping_dict.items():
        target_field = mapping_info['target_field']
        context = healthcare_mapper._get_domain_context(source_field)
        print(f"\nAnalyzing {source_field} -> {target_field}:")
        print(f"Domain Context: {context}")
        
        # Check field type rules
        is_valid = healthcare_mapper._validate_field_type(
            source_field,
            "sample_value",  # Would be actual value in real scenario
            mapping_info['source_type']
        )
        print(f"Field Type Valid: {is_valid}")
        
        # Get business rules
        rules = healthcare_mapper.rules.get('mapping_rules', {}).get(f'{context}_rules', [])
        relevant_rules = [
            rule['rule'] for rule in rules 
            if source_field in rule.get('fields', []) or target_field in rule.get('fields', [])
        ]
        if relevant_rules:
            print("Applicable Business Rules:")
            for rule in relevant_rules:
                print(f"- {rule}")

    # Step 3: Validate Mappings
    print_section("Step 3: Validation")
    validation_results = healthcare_mapper.validate_mapping(mapping_dict)
    print("\nValidation Results:")
    print(json.dumps(validation_results, indent=2))

    # Step 4: Generate Context Report
    print_section("Step 4: Context Report")
    context_reporter = ContextReporter(healthcare_mapper)
    context_report = context_reporter.generate_context_report(
        source_schema,
        target_schema,
        mappings,
        validation_results
    )

    # Save and show summary
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / 'detailed_mapping_process.json'
    
    with open(report_path, 'w') as f:
        json.dump(context_report, f, indent=2)
    
    print(f"\nFull report saved to: {report_path}")
    
    # Show key metrics
    print("\nKey Metrics from Context Report:")
    metrics = context_report['quality_metrics']
    print(f"Overall Score: {metrics['overall_score']:.2f}")
    print("\nMetrics Breakdown:")
    for metric, details in metrics['metrics'].items():
        print(f"- {metric}: {details['score']:.2f}")
        print(f"  {details['details']}")

if __name__ == "__main__":
    main() 