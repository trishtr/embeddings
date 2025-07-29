#!/usr/bin/env python3

import json
from pathlib import Path
from src.embeddings.embedding_handler import EmbeddingHandler
from src.utils.data_profiler import EnhancedDataProfiler, PostMappingProfiler

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def main():
    # Initialize
    embedding_handler = EmbeddingHandler(use_gpu=False)
    data_profiler = EnhancedDataProfiler(None, embedding_handler)  # None for db_handler as we're using schema only

    # Sample data
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

    # 1. Pre-Mapping Profiling
    print_section("1. Pre-Mapping Profiling")

    # Profile source schema
    source_profile = data_profiler.profile_source_independently(source_schema)
    print("\nSource Schema Profile:")
    print("Field Analysis:")
    for field, details in source_profile["field_analysis"].items():
        print(f"\n{field}:")
        print(f"  Type: {details['type']}")
        print(f"  Pattern: {details.get('pattern', 'N/A')}")
        print(f"  Category: {details.get('category', 'N/A')}")

    # Profile target schema
    target_profile = data_profiler.profile_target_independently(target_schema)
    print("\nTarget Schema Profile:")
    print("Field Analysis:")
    for field, details in target_profile["field_analysis"].items():
        print(f"\n{field}:")
        print(f"  Type: {details['type']}")
        print(f"  Pattern: {details.get('pattern', 'N/A')}")
        print(f"  Category: {details.get('category', 'N/A')}")

    # 2. Analyze Potential Mappings
    print_section("2. Potential Mapping Analysis")
    potential_mappings = data_profiler.analyze_potential_mappings(
        source_profile,
        target_profile
    )
    
    print("\nPotential Field Matches:")
    for source_field, matches in potential_mappings["field_matches"].items():
        print(f"\n{source_field} could map to:")
        for target_field, score in matches.items():
            print(f"  - {target_field} (similarity: {score:.2f})")

    # 3. Field Patterns and Relationships
    print_section("3. Field Patterns and Relationships")
    
    print("\nSource Schema Patterns:")
    for pattern, fields in source_profile["patterns"].items():
        print(f"\n{pattern}:")
        for field in fields:
            print(f"  - {field}")

    print("\nTarget Schema Patterns:")
    for pattern, fields in target_profile["patterns"].items():
        print(f"\n{pattern}:")
        for field in fields:
            print(f"  - {field}")

    # 4. Data Quality Assessment
    print_section("4. Data Quality Assessment")
    
    print("\nSource Schema Quality:")
    quality = source_profile["quality_metrics"]
    print(f"Completeness: {quality['completeness']:.2f}")
    print(f"Consistency: {quality['consistency']:.2f}")
    print(f"Validity: {quality['validity']:.2f}")

    print("\nTarget Schema Quality:")
    quality = target_profile["quality_metrics"]
    print(f"Completeness: {quality['completeness']:.2f}")
    print(f"Consistency: {quality['consistency']:.2f}")
    print(f"Validity: {quality['validity']:.2f}")

    # 5. Mapping Readiness Assessment
    print_section("5. Mapping Readiness Assessment")
    
    readiness = data_profiler.assess_mapping_readiness(
        source_profile,
        target_profile,
        potential_mappings
    )
    
    print("\nReadiness Scores:")
    print(f"Overall Score: {readiness['overall_score']:.2f}")
    print("\nComponent Scores:")
    for component, score in readiness["component_scores"].items():
        print(f"{component}: {score:.2f}")

    print("\nRecommendations:")
    for rec in readiness["recommendations"]:
        print(f"- {rec}")

    # Save detailed results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    detailed_results = {
        "source_profile": source_profile,
        "target_profile": target_profile,
        "potential_mappings": potential_mappings,
        "readiness_assessment": readiness
    }
    
    output_path = output_dir / 'detailed_profiling_results.json'
    with open(output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main() 