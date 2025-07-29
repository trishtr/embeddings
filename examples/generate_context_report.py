#!/usr/bin/env python3

import json
from pathlib import Path
from src.embeddings.embedding_handler import EmbeddingHandler
from src.schema_mapping.healthcare_mapper import HealthcareSchemaMapper
from src.utils.context_reporter import ContextReporter

def main():
    # Step 1: Initialize components
    print("Initializing components...")
    embedding_handler = EmbeddingHandler(use_gpu=False)
    healthcare_mapper = HealthcareSchemaMapper(
        embedding_handler,
        rules_file='config/healthcare_rules.yaml'
    )
    context_reporter = ContextReporter(healthcare_mapper)

    # Step 2: Define schemas
    print("Setting up schemas...")
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

    # Step 3: Generate mappings
    print("Generating field mappings...")
    mappings = healthcare_mapper.find_field_mappings(source_schema, target_schema)
    
    # Print initial mappings
    print("\nInitial Mappings:")
    for source, target, confidence in mappings:
        print(f"{source} -> {target} (confidence: {confidence:.2f})")

    # Step 4: Validate mappings
    print("\nValidating mappings...")
    mapping_dict = {
        field: {
            "target_field": target,
            "confidence": score
        } for field, target, score in mappings
    }
    validation_results = healthcare_mapper.validate_mapping(mapping_dict)

    # Step 5: Generate context report
    print("Generating context report...")
    context_report = context_reporter.generate_context_report(
        source_schema=source_schema,
        target_schema=target_schema,
        mappings=mappings,
        validation_results=validation_results
    )

    # Step 6: Save and display results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save full report
    report_path = output_dir / 'mapping_context_report.json'
    with open(report_path, 'w') as f:
        json.dump(context_report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    # Print summary sections
    print("\nContext Analysis Summary:")
    print("-" * 50)
    
    # Domain Context
    source_domain = context_report['pre_mapping_analysis']['source_schema_context']['primary_domain']
    target_domain = context_report['pre_mapping_analysis']['target_schema_context']['primary_domain']
    print(f"Source Schema Domain: {source_domain}")
    print(f"Target Schema Domain: {target_domain}")

    # Mapping Quality
    quality = context_report['quality_metrics']
    print(f"\nQuality Metrics:")
    print(f"Overall Score: {quality['overall_score']:.2f}")
    print(f"Context Preservation: {quality['metrics']['context_preservation']['score']:.2f}")
    print(f"Data Completeness: {quality['metrics']['data_completeness']['score']:.2f}")
    print(f"Rule Compliance: {quality['metrics']['rule_compliance']['score']:.2f}")

    # Recommendations
    print("\nKey Recommendations:")
    for rec in quality['recommendations']:
        print(f"- {rec['field']}: {rec['suggestion']} (Priority: {rec['priority']})")

if __name__ == "__main__":
    main() 