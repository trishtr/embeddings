#!/usr/bin/env python3
"""
Enhanced Comparison Example

This example demonstrates how the data profiler handles cases where
all column names in source and target schemas are completely different.

It shows:
1. Basic comparison (limited value when names differ)
2. Enhanced comparison with similarity analysis
3. Potential mapping suggestions
4. Data compatibility analysis

Author: Schema Mapping System
Date: 2024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_profiler import DataProfiler, EnhancedDataProfiler
from db.db_handler import MultiSourceDBHandler
from embeddings.embedding_handler import EmbeddingHandler

def create_different_schemas_example():
    """Create example with completely different column names"""
    
    # Source schema with different naming convention
    source_schema = {
        "columns": {
            "provider_npi": {"type": "VARCHAR(20)", "nullable": False},
            "specialty_code": {"type": "VARCHAR(10)", "nullable": True},
            "phone_number": {"type": "VARCHAR(20)", "nullable": True},
            "email_address": {"type": "VARCHAR(100)", "nullable": True},
            "practice_name": {"type": "VARCHAR(200)", "nullable": True}
        }
    }
    
    # Target schema with completely different naming convention
    target_schema = {
        "columns": {
            "npi_number": {"type": "VARCHAR(20)", "nullable": False},
            "medical_specialty": {"type": "VARCHAR(50)", "nullable": True},
            "contact_phone": {"type": "VARCHAR(15)", "nullable": True},
            "email": {"type": "VARCHAR(100)", "nullable": True},
            "facility_name": {"type": "VARCHAR(200)", "nullable": True}
        }
    }
    
    # Mock statistics for source
    source_stats = {
        "provider_npi": {
            "count": 1000, "unique_count": 1000, "null_count": 0,
            "min": "1000000000", "max": "1999999999", "mean": "1500000000"
        },
        "specialty_code": {
            "count": 1000, "unique_count": 50, "null_count": 10,
            "min": "01", "max": "99", "mean": "45"
        },
        "phone_number": {
            "count": 1000, "unique_count": 950, "null_count": 20,
            "min_length": 10, "max_length": 15, "avg_length": 12.5
        },
        "email_address": {
            "count": 1000, "unique_count": 1000, "null_count": 5,
            "min_length": 15, "max_length": 50, "avg_length": 25.5
        },
        "practice_name": {
            "count": 1000, "unique_count": 800, "null_count": 15,
            "min_length": 10, "max_length": 100, "avg_length": 45.2
        }
    }
    
    # Mock statistics for target
    target_stats = {
        "npi_number": {
            "count": 500, "unique_count": 500, "null_count": 0,
            "min": "1000000000", "max": "1999999999", "mean": "1500000000"
        },
        "medical_specialty": {
            "count": 500, "unique_count": 30, "null_count": 5,
            "min": "01", "max": "99", "mean": "42"
        },
        "contact_phone": {
            "count": 500, "unique_count": 480, "null_count": 10,
            "min_length": 10, "max_length": 15, "avg_length": 12.8
        },
        "email": {
            "count": 500, "unique_count": 500, "null_count": 3,
            "min_length": 15, "max_length": 50, "avg_length": 26.1
        },
        "facility_name": {
            "count": 500, "unique_count": 400, "null_count": 8,
            "min_length": 10, "max_length": 100, "avg_length": 47.5
        }
    }
    
    # Create mock profiles
    source_profile = {
        "schema": source_schema,
        "statistics": {"columns": source_stats}
    }
    
    target_profile = {
        "schema": target_schema,
        "statistics": {"columns": target_stats}
    }
    
    return source_profile, target_profile

def demonstrate_basic_comparison(source_profile, target_profile):
    """Demonstrate basic comparison limitations"""
    print("=" * 60)
    print("BASIC COMPARISON (Limited Value When Names Differ)")
    print("=" * 60)
    
    # Create a basic profiler
    basic_profiler = DataProfiler(None)
    
    # Perform basic comparison
    basic_comparison = basic_profiler.compare_profiles(source_profile, target_profile)
    
    print("\n📊 Basic Comparison Results:")
    print(f"Source-only columns: {basic_comparison['schema_differences']['source_only_columns']}")
    print(f"Target-only columns: {basic_comparison['schema_differences']['target_only_columns']}")
    print(f"Type mismatches: {len(basic_comparison['schema_differences']['type_mismatches'])}")
    print(f"Value ranges compared: {len(basic_comparison['data_distribution']['value_ranges'])}")
    print(f"Null ratios compared: {len(basic_comparison['data_distribution']['null_ratios'])}")
    
    print("\n❌ Limitations:")
    print("• No exact column name matches found")
    print("• Cannot compare data types")
    print("• Cannot analyze value distributions")
    print("• Limited insights provided")
    
    return basic_comparison

def demonstrate_enhanced_comparison(source_profile, target_profile):
    """Demonstrate enhanced comparison capabilities"""
    print("\n" + "=" * 60)
    print("ENHANCED COMPARISON (Intelligent Similarity Analysis)")
    print("=" * 60)
    
    # Create an enhanced profiler
    enhanced_profiler = EnhancedDataProfiler(None)
    
    # Perform enhanced comparison
    enhanced_comparison = enhanced_profiler.compare_profiles_enhanced(source_profile, target_profile)
    
    print("\n🔍 Enhanced Comparison Results:")
    print(f"Overall similarity score: {enhanced_comparison['similarity_analysis']['overall_similarity_score']:.2f}")
    
    print("\n📋 Potential Mappings Found:")
    potential_mappings = enhanced_comparison['schema_differences']['potential_mappings']
    for source_col, mapping_info in potential_mappings.items():
        best_match = mapping_info['best_match']
        confidence = mapping_info['confidence']
        print(f"  {source_col} → {best_match[0]} (confidence: {confidence:.2f})")
    
    print("\n🔧 Data Compatibility Analysis:")
    compatibility = enhanced_comparison['data_distribution']['compatibility_analysis']
    for mapping, compat_info in compatibility.items():
        score = compat_info['overall_compatibility_score']
        type_comp = "✅" if compat_info['type_compatibility'] else "❌"
        range_comp = "✅" if compat_info['range_compatibility'] else "❌"
        dist_comp = "✅" if compat_info['distribution_compatibility'] else "❌"
        
        print(f"  {mapping}:")
        print(f"    Overall Score: {score:.2f}")
        print(f"    Type Compatible: {type_comp}")
        print(f"    Range Compatible: {range_comp}")
        print(f"    Distribution Compatible: {dist_comp}")
    
    print("\n✅ Benefits:")
    print("• Intelligent field matching using similarity analysis")
    print("• Multi-factor comparison (semantic, type, pattern)")
    print("• Data compatibility assessment")
    print("• Confidence scoring for potential mappings")
    print("• Actionable insights for mapping decisions")
    
    return enhanced_comparison

def demonstrate_potential_mapping_analysis(source_profile, target_profile):
    """Demonstrate potential mapping analysis"""
    print("\n" + "=" * 60)
    print("POTENTIAL MAPPING ANALYSIS")
    print("=" * 60)
    
    enhanced_profiler = EnhancedDataProfiler(None)
    
    # Perform potential mapping analysis
    potential_analysis = enhanced_profiler.analyze_potential_mappings(source_profile, target_profile)
    
    print("\n🎯 Top Potential Mappings:")
    potential_mappings = potential_analysis['potential_mappings']
    
    for source_col, mapping_info in potential_mappings.items():
        print(f"\n📌 {source_col}:")
        top_matches = mapping_info['top_matches']
        for i, (target_col, similarity) in enumerate(top_matches, 1):
            print(f"  {i}. {target_col} (similarity: {similarity:.3f})")
    
    print("\n📊 Field Similarity Matrix (Sample):")
    similarities = potential_analysis['field_similarities']
    
    # Show similarity matrix for first few fields
    source_fields = list(similarities.keys())[:3]
    target_fields = list(similarities[source_fields[0]].keys())[:3]
    
    print("          ", end="")
    for target in target_fields:
        print(f"{target:15}", end="")
    print()
    
    for source in source_fields:
        print(f"{source:12}", end="")
        for target in target_fields:
            sim = similarities[source][target]
            print(f"{sim:15.3f}", end="")
        print()
    
    return potential_analysis

def main():
    """Main demonstration function"""
    print("🔍 ENHANCED COMPARISON DEMONSTRATION")
    print("=" * 80)
    print("This example shows how the data profiler handles completely different column names")
    print("=" * 80)
    
    # Create example profiles with different column names
    source_profile, target_profile = create_different_schemas_example()
    
    print("\n📋 Example Schemas:")
    print("Source columns:", list(source_profile['schema']['columns'].keys()))
    print("Target columns:", list(target_profile['schema']['columns'].keys()))
    print("\nNote: All column names are completely different!")
    
    # Demonstrate basic comparison limitations
    basic_result = demonstrate_basic_comparison(source_profile, target_profile)
    
    # Demonstrate enhanced comparison capabilities
    enhanced_result = demonstrate_enhanced_comparison(source_profile, target_profile)
    
    # Demonstrate potential mapping analysis
    potential_result = demonstrate_potential_mapping_analysis(source_profile, target_profile)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ Enhanced comparison provides intelligent analysis even when column names differ")
    print("✅ Potential mappings are suggested based on semantic similarity")
    print("✅ Data compatibility is assessed for suggested mappings")
    print("✅ Confidence scores help prioritize mapping decisions")
    print("✅ Actionable insights are provided for the mapping process")
    
    print("\n🎯 Key Takeaway:")
    print("The enhanced comparison transforms 'no matches found' into")
    print("'intelligent mapping suggestions with confidence scores'")

if __name__ == "__main__":
    main() 