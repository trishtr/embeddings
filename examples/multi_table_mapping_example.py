#!/usr/bin/env python3

import json
from pathlib import Path
from src.embeddings.embedding_handler import EmbeddingHandler
from src.schema_mapping.healthcare_mapper import HealthcareSchemaMapper
from src.utils.data_profiler import EnhancedDataProfiler

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def main():
    # Initialize components
    embedding_handler = EmbeddingHandler(use_gpu=False)
    healthcare_mapper = HealthcareSchemaMapper(embedding_handler)
    data_profiler = EnhancedDataProfiler(None, embedding_handler)

    # Define multiple source tables
    source_schemas = {
        "provider_details": {
            "provider_npi": "VARCHAR(10)",
            "doctor_name": "VARCHAR(100)",
            "specialty_code": "VARCHAR(20)",
            "license_number": "VARCHAR(15)",
            "practice_address": "VARCHAR(200)"
        },
        "provider_credentials": {
            "provider_id": "VARCHAR(10)",
            "board_certification": "VARCHAR(50)",
            "certification_date": "DATE",
            "expiration_date": "DATE",
            "status": "VARCHAR(20)"
        },
        "provider_locations": {
            "location_id": "VARCHAR(10)",
            "provider_id": "VARCHAR(10)",
            "facility_name": "VARCHAR(100)",
            "address_line1": "VARCHAR(100)",
            "city": "VARCHAR(50)",
            "state": "VARCHAR(2)",
            "zip_code": "VARCHAR(10)"
        }
    }

    # Define multiple target tables
    target_schemas = {
        "healthcare_providers": {
            "healthcare_provider_id": "VARCHAR(10)",
            "provider_full_name": "VARCHAR(100)",
            "provider_taxonomy": "VARCHAR(20)",
            "state_license": "VARCHAR(15)",
            "primary_location": "VARCHAR(200)"
        },
        "provider_certifications": {
            "provider_id": "VARCHAR(10)",
            "certification_type": "VARCHAR(50)",
            "issue_date": "DATE",
            "valid_until": "DATE",
            "certification_status": "VARCHAR(20)"
        },
        "practice_locations": {
            "practice_id": "VARCHAR(10)",
            "healthcare_provider_id": "VARCHAR(10)",
            "location_name": "VARCHAR(100)",
            "street_address": "VARCHAR(100)",
            "city": "VARCHAR(50)",
            "state_code": "VARCHAR(2)",
            "postal_code": "VARCHAR(10)"
        }
    }

    # Step 1: Profile Each Source Table
    print_section("Source Tables Profiling")
    source_profiles = {}
    
    for table_name, schema in source_schemas.items():
        print(f"\nProfiling source table: {table_name}")
        profile = data_profiler.profile_source_independently(schema)
        source_profiles[table_name] = profile
        
        # Show key metrics
        print(f"Fields: {len(schema)}")
        print("Field Categories:")
        for field, details in profile["field_analysis"].items():
            print(f"- {field}: {details['category']}")

    # Step 2: Profile Each Target Table
    print_section("Target Tables Profiling")
    target_profiles = {}
    
    for table_name, schema in target_schemas.items():
        print(f"\nProfiling target table: {table_name}")
        profile = data_profiler.profile_target_independently(schema)
        target_profiles[table_name] = profile
        
        # Show key metrics
        print(f"Fields: {len(schema)}")
        print("Field Categories:")
        for field, details in profile["field_analysis"].items():
            print(f"- {field}: {details['category']}")

    # Step 3: Analyze Table-Level Relationships
    print_section("Table Relationship Analysis")
    table_relationships = analyze_table_relationships(source_profiles, target_profiles)
    print("\nTable Mapping Suggestions:")
    for source_table, mappings in table_relationships.items():
        print(f"\n{source_table} could map to:")
        for target_table, score in mappings.items():
            print(f"- {target_table} (similarity: {score:.2f})")

    # Step 4: Analyze Cross-Table Field Relationships
    print_section("Cross-Table Field Analysis")
    field_relationships = analyze_field_relationships(source_schemas, target_schemas)
    print("\nCross-Table Field Relationships:")
    for source_table, fields in field_relationships.items():
        print(f"\n{source_table} fields referenced in other tables:")
        for field, references in fields.items():
            print(f"- {field} referenced in: {', '.join(references)}")

    # Step 5: Generate Multi-Table Mapping Suggestions
    print_section("Multi-Table Mapping Suggestions")
    mapping_suggestions = generate_mapping_suggestions(
        source_profiles,
        target_profiles,
        table_relationships,
        field_relationships
    )

    # Save detailed analysis
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    multi_table_analysis = {
        "source_profiles": source_profiles,
        "target_profiles": target_profiles,
        "table_relationships": table_relationships,
        "field_relationships": field_relationships,
        "mapping_suggestions": mapping_suggestions
    }
    
    output_path = output_dir / 'multi_table_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(multi_table_analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_path}")

def analyze_table_relationships(source_profiles, target_profiles):
    """Analyze relationships between source and target tables"""
    relationships = {}
    
    for source_table, source_profile in source_profiles.items():
        relationships[source_table] = {}
        source_fields = set(source_profile["field_analysis"].keys())
        
        for target_table, target_profile in target_profiles.items():
            target_fields = set(target_profile["field_analysis"].keys())
            
            # Calculate table similarity based on field patterns
            common_patterns = len(set(source_profile["patterns"].keys()) & 
                                set(target_profile["patterns"].keys()))
            total_patterns = len(set(source_profile["patterns"].keys()) | 
                               set(target_profile["patterns"].keys()))
            
            # Calculate field name similarity
            common_terms = set()
            for field in source_fields:
                terms = set(field.lower().replace('_', ' ').split())
                common_terms.update(terms)
            
            target_terms = set()
            for field in target_fields:
                terms = set(field.lower().replace('_', ' ').split())
                target_terms.update(terms)
            
            term_similarity = len(common_terms & target_terms) / len(common_terms | target_terms)
            
            # Combined similarity score
            similarity = (0.6 * term_similarity + 0.4 * (common_patterns / total_patterns if total_patterns > 0 else 0))
            relationships[source_table][target_table] = similarity
    
    return relationships

def analyze_field_relationships(source_schemas, target_schemas):
    """Analyze cross-table field relationships"""
    relationships = {}
    
    # Analyze source schemas
    for source_table, schema in source_schemas.items():
        relationships[source_table] = {}
        
        for field in schema.keys():
            base_name = field.replace('_id', '').replace('_number', '')
            references = []
            
            # Look for references in other source tables
            for other_table, other_schema in source_schemas.items():
                if other_table != source_table:
                    for other_field in other_schema.keys():
                        if base_name in other_field.lower():
                            references.append(f"{other_table}.{other_field}")
            
            # Look for references in target tables
            for target_table, target_schema in target_schemas.items():
                for target_field in target_schema.keys():
                    if base_name in target_field.lower():
                        references.append(f"{target_table}.{target_field}")
            
            if references:
                relationships[source_table][field] = references
    
    return relationships

def generate_mapping_suggestions(source_profiles, target_profiles, table_relationships, field_relationships):
    """Generate mapping suggestions for multiple tables"""
    suggestions = {
        "table_mappings": {},
        "field_mappings": {},
        "transformation_rules": {},
        "data_movement": {}
    }
    
    # Generate table-level mappings
    for source_table, targets in table_relationships.items():
        best_target = max(targets.items(), key=lambda x: x[1])
        suggestions["table_mappings"][source_table] = {
            "primary_target": best_target[0],
            "confidence": best_target[1],
            "alternative_targets": [
                {"table": t, "confidence": s}
                for t, s in targets.items()
                if t != best_target[0] and s > 0.5
            ]
        }
    
    # Generate field-level mappings
    for source_table, source_profile in source_profiles.items():
        suggestions["field_mappings"][source_table] = {}
        best_target = suggestions["table_mappings"][source_table]["primary_target"]
        target_profile = target_profiles[best_target]
        
        for source_field in source_profile["field_analysis"].keys():
            field_suggestions = []
            source_category = source_profile["field_analysis"][source_field]["category"]
            
            for target_field, target_details in target_profile["field_analysis"].items():
                if target_details["category"] == source_category:
                    # Calculate field similarity
                    name_similarity = calculate_field_similarity(source_field, target_field)
                    if name_similarity > 0.5:
                        field_suggestions.append({
                            "field": target_field,
                            "confidence": name_similarity,
                            "category_match": True
                        })
            
            if field_suggestions:
                suggestions["field_mappings"][source_table][source_field] = sorted(
                    field_suggestions,
                    key=lambda x: x["confidence"],
                    reverse=True
                )
    
    # Generate transformation rules
    for source_table, mappings in suggestions["field_mappings"].items():
        suggestions["transformation_rules"][source_table] = {}
        for source_field, targets in mappings.items():
            if targets:
                best_target = targets[0]["field"]
                source_type = source_profiles[source_table]["field_analysis"][source_field]["type"]
                target_type = target_profiles[suggestions["table_mappings"][source_table]["primary_target"]]["field_analysis"][best_target]["type"]
                
                if source_type != target_type:
                    suggestions["transformation_rules"][source_table][source_field] = {
                        "type": "type_conversion",
                        "from": source_type,
                        "to": target_type
                    }
    
    # Generate data movement suggestions
    for source_table, mapping in suggestions["table_mappings"].items():
        target_table = mapping["primary_target"]
        suggestions["data_movement"][source_table] = {
            "target": target_table,
            "type": "direct" if mapping["confidence"] > 0.8 else "transform",
            "dependencies": [
                dep.split('.')[0] for deps in field_relationships[source_table].values()
                for dep in deps if dep.startswith(target_table)
            ]
        }
    
    return suggestions

def calculate_field_similarity(field1, field2):
    """Calculate similarity between two field names"""
    terms1 = set(field1.lower().replace('_', ' ').split())
    terms2 = set(field2.lower().replace('_', ' ').split())
    
    intersection = len(terms1 & terms2)
    union = len(terms1 | terms2)
    
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    main() 