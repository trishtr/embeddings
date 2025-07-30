#!/usr/bin/env python3
"""
Parallel Mapping Pipeline Demonstration

This script demonstrates the parallel processing approach with confidence aggregation
and target field selection for rule-based and embedding engines.

Author: Schema Mapping System
Date: 2024
"""

import sys
import os
import asyncio
import time
import json
from typing import Dict, List, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.parallel_mapping_pipeline import ParallelMappingPipeline

class MockRuleBasedEngine:
    """Mock rule-based engine for demonstration."""
    
    def process(self, source_schema: Dict[str, str], target_schema: Dict[str, str]) -> Dict[str, Any]:
        """Mock rule-based processing."""
        results = {}
        
        # Simulate rule-based processing with realistic scores and targets
        rule_data = {
            "provider_npi": {"confidence": 0.95, "target_field": "npi_number"},
            "specialty_code": {"confidence": 0.70, "target_field": "medical_specialty"},
            "phone_number": {"confidence": 0.85, "target_field": "contact_phone"},
            "email_address": {"confidence": 0.90, "target_field": "email_address"},
            "practice_name": {"confidence": 0.65, "target_field": "practice_name"},
            "license_number": {"confidence": 0.88, "target_field": "license_number"},
            "tax_id": {"confidence": 0.45, "target_field": "employer_id"},
            "date_of_birth": {"confidence": 0.92, "target_field": "date_of_birth"},
            "gender": {"confidence": 0.85, "target_field": "gender"},
            "city": {"confidence": 0.88, "target_field": "city"},
            "state": {"confidence": 0.90, "target_field": "state"},
            "zip_code": {"confidence": 0.87, "target_field": "zip_code"},
            "website_url": {"confidence": 0.82, "target_field": "website"},
            "fax_number": {"confidence": 0.80, "target_field": "fax_number"},
            "education": {"confidence": 0.60, "target_field": "education"},
            "certifications": {"confidence": 0.55, "target_field": "certifications"},
            "experience_years": {"confidence": 0.75, "target_field": "years_experience"},
            "hospital_affiliations": {"confidence": 0.50, "target_field": "hospital_affiliations"},
            "research_interests": {"confidence": 0.40, "target_field": "research_focus"},
            "publications": {"confidence": 0.35, "target_field": "publications"},
            "awards": {"confidence": 0.45, "target_field": "awards"},
            "patient_ratings": {"confidence": 0.70, "target_field": "patient_ratings"},
            "wait_time": {"confidence": 0.65, "target_field": "wait_time_days"},
            "appointment_types": {"confidence": 0.60, "target_field": "appointment_types"},
            "special_services": {"confidence": 0.45, "target_field": "specialized_services"},
            "equipment_available": {"confidence": 0.40, "target_field": "equipment_available"},
            "staff_count": {"confidence": 0.75, "target_field": "staff_count"},
            "patient_volume": {"confidence": 0.70, "target_field": "patient_volume"},
            "accreditation_status": {"confidence": 0.65, "target_field": "accreditation_status"},
            "compliance_score": {"confidence": 0.60, "target_field": "quality_score"},
            "audit_history": {"confidence": 0.35, "target_field": "audit_records"},
            "training_completed": {"confidence": 0.50, "target_field": "training_status"},
            "continuing_education": {"confidence": 0.45, "target_field": "continuing_education"},
            "malpractice_history": {"confidence": 0.30, "target_field": "malpractice_record"},
            "disciplinary_actions": {"confidence": 0.25, "target_field": "disciplinary_history"},
            "peer_reviews": {"confidence": 0.40, "target_field": "peer_reviews"},
            "patient_outcomes": {"confidence": 0.55, "target_field": "patient_outcomes"},
            "quality_metrics": {"confidence": 0.50, "target_field": "quality_metrics"},
            "satisfaction_scores": {"confidence": 0.65, "target_field": "satisfaction_scores"},
            "accessibility_features": {"confidence": 0.45, "target_field": "accessibility_features"},
            "telemedicine_available": {"confidence": 0.70, "target_field": "telemedicine_available"},
            "after_hours_contact": {"confidence": 0.60, "target_field": "after_hours_contact"},
            "holiday_schedule": {"confidence": 0.55, "target_field": "holiday_schedule"},
            "payment_methods": {"confidence": 0.65, "target_field": "payment_methods"},
            "financial_assistance": {"confidence": 0.50, "target_field": "financial_assistance"},
            "insurance_accepted": {"confidence": 0.60, "target_field": "insurance_accepted"},
            "languages_spoken": {"confidence": 0.55, "target_field": "languages_spoken"},
            "accepting_patients": {"confidence": 0.75, "target_field": "accepting_patients"}
        }
        
        for field_name in source_schema.keys():
            if field_name in rule_data:
                data = rule_data[field_name]
                results[field_name] = {
                    "confidence": data["confidence"],
                    "target_field": data["target_field"],
                    "method": "rule_based",
                    "explanation": f"Rule-based mapping for {field_name}"
                }
            else:
                results[field_name] = {
                    "confidence": 0.5,
                    "target_field": f"target_{field_name}",
                    "method": "rule_based",
                    "explanation": f"Default rule-based mapping for {field_name}"
                }
        
        return results

class MockEmbeddingEngine:
    """Mock embedding engine for demonstration."""
    
    def process(self, source_schema: Dict[str, str], target_schema: Dict[str, str]) -> Dict[str, Any]:
        """Mock embedding processing."""
        results = {}
        
        # Simulate embedding processing with realistic scores and targets
        # Note: Some targets are different from rule-based engine to demonstrate disagreement
        embedding_data = {
            "provider_npi": {"confidence": 0.92, "target_field": "npi_number"},  # Same target
            "specialty_code": {"confidence": 0.78, "target_field": "specialty_type"},  # Different target
            "phone_number": {"confidence": 0.88, "target_field": "contact_phone"},  # Same target
            "email_address": {"confidence": 0.94, "target_field": "email_address"},  # Same target
            "practice_name": {"confidence": 0.72, "target_field": "facility_name"},  # Different target
            "license_number": {"confidence": 0.85, "target_field": "license_id"},  # Different target
            "tax_id": {"confidence": 0.52, "target_field": "tax_identifier"},  # Different target
            "date_of_birth": {"confidence": 0.89, "target_field": "birth_date"},  # Different target
            "gender": {"confidence": 0.82, "target_field": "sex"},  # Different target
            "city": {"confidence": 0.85, "target_field": "city_name"},  # Different target
            "state": {"confidence": 0.88, "target_field": "state_code"},  # Different target
            "zip_code": {"confidence": 0.84, "target_field": "postal_code"},  # Different target
            "website_url": {"confidence": 0.78, "target_field": "web_site"},  # Different target
            "fax_number": {"confidence": 0.75, "target_field": "fax_contact"},  # Different target
            "education": {"confidence": 0.68, "target_field": "medical_education"},  # Different target
            "certifications": {"confidence": 0.62, "target_field": "professional_certifications"},  # Different target
            "experience_years": {"confidence": 0.78, "target_field": "years_experience"},  # Same target
            "hospital_affiliations": {"confidence": 0.58, "target_field": "hospital_connections"},  # Different target
            "research_interests": {"confidence": 0.48, "target_field": "research_interests"},  # Different target
            "publications": {"confidence": 0.42, "target_field": "published_works"},  # Different target
            "awards": {"confidence": 0.52, "target_field": "professional_awards"},  # Different target
            "patient_ratings": {"confidence": 0.75, "target_field": "patient_satisfaction"},  # Different target
            "wait_time": {"confidence": 0.68, "target_field": "appointment_wait_time"},  # Different target
            "appointment_types": {"confidence": 0.65, "target_field": "visit_types"},  # Different target
            "special_services": {"confidence": 0.52, "target_field": "specialized_services"},  # Different target
            "equipment_available": {"confidence": 0.48, "target_field": "medical_equipment"},  # Different target
            "staff_count": {"confidence": 0.78, "target_field": "team_size"},  # Different target
            "patient_volume": {"confidence": 0.72, "target_field": "patient_load"},  # Different target
            "accreditation_status": {"confidence": 0.68, "target_field": "accreditation_level"},  # Different target
            "compliance_score": {"confidence": 0.65, "target_field": "quality_score"},  # Same target
            "audit_history": {"confidence": 0.42, "target_field": "audit_records"},  # Different target
            "training_completed": {"confidence": 0.55, "target_field": "training_status"},  # Same target
            "continuing_education": {"confidence": 0.52, "target_field": "ce_credits"},  # Different target
            "malpractice_history": {"confidence": 0.38, "target_field": "malpractice_history"},  # Different target
            "disciplinary_actions": {"confidence": 0.32, "target_field": "disciplinary_history"},  # Different target
            "peer_reviews": {"confidence": 0.48, "target_field": "colleague_reviews"},  # Different target
            "patient_outcomes": {"confidence": 0.62, "target_field": "treatment_outcomes"},  # Different target
            "quality_metrics": {"confidence": 0.58, "target_field": "performance_metrics"},  # Different target
            "satisfaction_scores": {"confidence": 0.68, "target_field": "satisfaction_ratings"},  # Different target
            "accessibility_features": {"confidence": 0.52, "target_field": "accessibility_options"},  # Different target
            "telemedicine_available": {"confidence": 0.75, "target_field": "virtual_visits"},  # Different target
            "after_hours_contact": {"confidence": 0.65, "target_field": "after_hours_phone"},  # Different target
            "holiday_schedule": {"confidence": 0.58, "target_field": "holiday_hours"},  # Different target
            "payment_methods": {"confidence": 0.68, "target_field": "payment_options"},  # Different target
            "financial_assistance": {"confidence": 0.55, "target_field": "financial_support"},  # Different target
            "insurance_accepted": {"confidence": 0.65, "target_field": "insurance_providers"},  # Different target
            "languages_spoken": {"confidence": 0.58, "target_field": "spoken_languages"},  # Different target
            "accepting_patients": {"confidence": 0.78, "target_field": "new_patients_welcome"}  # Different target
        }
        
        for field_name in source_schema.keys():
            if field_name in embedding_data:
                data = embedding_data[field_name]
                results[field_name] = {
                    "confidence": data["confidence"],
                    "target_field": data["target_field"],
                    "method": "embedding",
                    "explanation": f"Embedding-based mapping for {field_name}"
                }
            else:
                results[field_name] = {
                    "confidence": 0.5,
                    "target_field": f"target_{field_name}",
                    "method": "embedding",
                    "explanation": f"Default embedding mapping for {field_name}"
                }
        
        return results

def create_demo_schemas():
    """Create demo schemas for the demonstration."""
    
    source_schema = {
        "provider_npi": "VARCHAR(20)",
        "specialty_code": "VARCHAR(10)",
        "phone_number": "VARCHAR(20)",
        "email_address": "VARCHAR(100)",
        "practice_name": "VARCHAR(200)",
        "license_number": "VARCHAR(20)",
        "tax_id": "VARCHAR(20)",
        "date_of_birth": "DATE",
        "gender": "VARCHAR(10)",
        "city": "VARCHAR(50)",
        "state": "VARCHAR(2)",
        "zip_code": "VARCHAR(10)",
        "website_url": "VARCHAR(200)",
        "fax_number": "VARCHAR(20)",
        "education": "VARCHAR(200)",
        "certifications": "VARCHAR(500)",
        "experience_years": "INTEGER",
        "hospital_affiliations": "VARCHAR(500)",
        "research_interests": "VARCHAR(500)",
        "publications": "VARCHAR(1000)",
        "awards": "VARCHAR(500)",
        "patient_ratings": "DECIMAL(3,2)",
        "wait_time": "INTEGER",
        "appointment_types": "VARCHAR(200)",
        "special_services": "VARCHAR(500)",
        "equipment_available": "VARCHAR(500)",
        "staff_count": "INTEGER",
        "patient_volume": "INTEGER",
        "accreditation_status": "VARCHAR(50)",
        "compliance_score": "DECIMAL(5,2)",
        "audit_history": "VARCHAR(1000)",
        "training_completed": "VARCHAR(500)",
        "continuing_education": "VARCHAR(500)",
        "malpractice_history": "VARCHAR(1000)",
        "disciplinary_actions": "VARCHAR(1000)",
        "peer_reviews": "VARCHAR(1000)",
        "patient_outcomes": "VARCHAR(500)",
        "quality_metrics": "VARCHAR(500)",
        "satisfaction_scores": "DECIMAL(3,2)",
        "accessibility_features": "VARCHAR(500)",
        "telemedicine_available": "BOOLEAN",
        "after_hours_contact": "VARCHAR(200)",
        "holiday_schedule": "VARCHAR(500)",
        "payment_methods": "VARCHAR(500)",
        "financial_assistance": "VARCHAR(500)",
        "insurance_accepted": "VARCHAR(1000)",
        "languages_spoken": "VARCHAR(500)",
        "accepting_patients": "BOOLEAN"
    }
    
    target_schema = {
        "npi_number": "VARCHAR(20)",
        "specialty_type": "VARCHAR(50)",
        "medical_specialty": "VARCHAR(50)",
        "contact_phone": "VARCHAR(15)",
        "email_address": "VARCHAR(100)",
        "facility_name": "VARCHAR(200)",
        "practice_name": "VARCHAR(200)",
        "license_id": "VARCHAR(20)",
        "license_number": "VARCHAR(20)",
        "tax_identifier": "VARCHAR(20)",
        "employer_id": "VARCHAR(20)",
        "birth_date": "DATE",
        "date_of_birth": "DATE",
        "sex": "VARCHAR(10)",
        "gender": "VARCHAR(10)",
        "city_name": "VARCHAR(50)",
        "city": "VARCHAR(50)",
        "state_code": "VARCHAR(2)",
        "state": "VARCHAR(2)",
        "postal_code": "VARCHAR(10)",
        "zip_code": "VARCHAR(10)",
        "web_site": "VARCHAR(200)",
        "website": "VARCHAR(200)",
        "fax_contact": "VARCHAR(20)",
        "fax_number": "VARCHAR(20)",
        "medical_education": "VARCHAR(200)",
        "education": "VARCHAR(200)",
        "professional_certifications": "VARCHAR(500)",
        "certifications": "VARCHAR(500)",
        "years_experience": "INTEGER",
        "hospital_connections": "VARCHAR(500)",
        "hospital_affiliations": "VARCHAR(500)",
        "research_interests": "VARCHAR(500)",
        "research_focus": "VARCHAR(500)",
        "published_works": "VARCHAR(1000)",
        "publications": "VARCHAR(1000)",
        "professional_awards": "VARCHAR(500)",
        "awards": "VARCHAR(500)",
        "patient_satisfaction": "DECIMAL(3,2)",
        "patient_ratings": "DECIMAL(3,2)",
        "appointment_wait_time": "INTEGER",
        "wait_time_days": "INTEGER",
        "visit_types": "VARCHAR(200)",
        "appointment_types": "VARCHAR(200)",
        "specialized_services": "VARCHAR(500)",
        "medical_equipment": "VARCHAR(500)",
        "equipment_available": "VARCHAR(500)",
        "team_size": "INTEGER",
        "staff_count": "INTEGER",
        "patient_load": "INTEGER",
        "patient_volume": "INTEGER",
        "accreditation_level": "VARCHAR(50)",
        "accreditation_status": "VARCHAR(50)",
        "quality_score": "DECIMAL(5,2)",
        "audit_records": "VARCHAR(1000)",
        "training_status": "VARCHAR(500)",
        "ce_credits": "VARCHAR(500)",
        "continuing_education": "VARCHAR(500)",
        "malpractice_history": "VARCHAR(1000)",
        "malpractice_record": "VARCHAR(1000)",
        "disciplinary_history": "VARCHAR(1000)",
        "colleague_reviews": "VARCHAR(1000)",
        "peer_reviews": "VARCHAR(1000)",
        "treatment_outcomes": "VARCHAR(500)",
        "patient_outcomes": "VARCHAR(500)",
        "performance_metrics": "VARCHAR(500)",
        "quality_metrics": "VARCHAR(500)",
        "satisfaction_ratings": "DECIMAL(3,2)",
        "satisfaction_scores": "DECIMAL(3,2)",
        "accessibility_options": "VARCHAR(500)",
        "accessibility_features": "VARCHAR(500)",
        "virtual_visits": "BOOLEAN",
        "telemedicine_available": "BOOLEAN",
        "after_hours_phone": "VARCHAR(200)",
        "after_hours_contact": "VARCHAR(200)",
        "holiday_hours": "VARCHAR(500)",
        "holiday_schedule": "VARCHAR(500)",
        "payment_options": "VARCHAR(500)",
        "payment_methods": "VARCHAR(500)",
        "financial_support": "VARCHAR(500)",
        "financial_assistance": "VARCHAR(500)",
        "insurance_providers": "VARCHAR(1000)",
        "insurance_accepted": "VARCHAR(1000)",
        "spoken_languages": "VARCHAR(500)",
        "languages_spoken": "VARCHAR(500)",
        "new_patients_welcome": "BOOLEAN",
        "accepting_patients": "BOOLEAN"
    }
    
    return source_schema, target_schema

def print_parallel_results(results: Dict[str, Any]):
    """Print comprehensive results of parallel mapping."""
    
    print("\n" + "=" * 80)
    print("⚡ PARALLEL MAPPING PIPELINE - COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    mappings = results["mappings"]
    metrics = results["performance_metrics"]
    
    print(f"\n⏱️  Processing Time: {results['processing_time']:.2f} seconds")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   • Total Fields: {metrics['total_fields']}")
    print(f"   • Efficiency Score: {metrics['performance_metrics']['efficiency_score']}%")
    print(f"   • Quality Score: {metrics['performance_metrics']['quality_score']}%")
    print(f"   • Agreement Score: {metrics['performance_metrics']['agreement_score']}%")
    print(f"   • Target Agreement Rate: {metrics['performance_metrics']['target_agreement_rate']}%")
    
    print(f"\n🎯 Confidence Distribution:")
    conf_dist = metrics['confidence_distribution']
    print(f"   • High Confidence (≥0.8): {conf_dist['high']} fields ({conf_dist['high']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Medium Confidence (0.6-0.8): {conf_dist['medium']} fields ({conf_dist['medium']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Low Confidence (<0.6): {conf_dist['low']} fields ({conf_dist['low']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Average Confidence: {conf_dist['average']}")
    
    print(f"\n🤝 Agreement Distribution:")
    agree_dist = metrics['agreement_distribution']
    print(f"   • High Agreement (≥0.8): {agree_dist['high']} fields ({agree_dist['high']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Medium Agreement (0.6-0.8): {agree_dist['medium']} fields ({agree_dist['medium']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Low Agreement (<0.6): {agree_dist['low']} fields ({agree_dist['low']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Average Agreement: {agree_dist['average']}")
    
    print(f"\n🎯 Target Selection Analysis:")
    target_analysis = metrics['target_selection_analysis']
    print(f"   • Same Targets: {target_analysis['same_targets']} fields ({target_analysis['same_targets']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Embedding Preferred: {target_analysis['embedding_preferred']} fields ({target_analysis['embedding_preferred']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Rule Preferred: {target_analysis['rule_preferred']} fields ({target_analysis['rule_preferred']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Target Disagreements: {target_analysis['target_disagreements']} fields ({target_analysis['target_disagreements']/metrics['total_fields']*100:.1f}%)")
    
    print(f"\n❓ Uncertainty Analysis:")
    uncertainty = metrics['uncertainty_analysis']
    print(f"   • Fields with Uncertainty: {uncertainty['fields_with_uncertainty']} ({uncertainty['fields_with_uncertainty']/metrics['total_fields']*100:.1f}%)")
    print(f"   • Average Uncertainty Factors: {uncertainty['average_uncertainty_factors']}")
    print(f"   • Target Disagreements: {uncertainty['target_disagreements']} ({uncertainty['target_disagreements']/metrics['total_fields']*100:.1f}%)")
    
    print(f"\n🔧 Method Distribution:")
    for method, count in metrics['method_distribution'].items():
        print(f"   • {method.replace('_', ' ').title()}: {count} fields ({count/metrics['total_fields']*100:.1f}%)")
    
    print(f"\n📋 Sample Mappings (Top 15 by Confidence):")
    sorted_mappings = sorted(mappings.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    for i, (source_field, mapping) in enumerate(sorted_mappings[:15]):
        confidence = mapping['confidence']
        method = mapping['method']
        agreement = mapping['confidence_breakdown']['agreement_level']
        target_selection = mapping['confidence_breakdown']['target_selection_method']
        rule_target = mapping['confidence_breakdown']['rule_target']
        embedding_target = mapping['confidence_breakdown']['embedding_target']
        
        # Choose icon based on method
        if 'high_confidence' in method:
            icon = "🟢"
        elif 'agreement' in method:
            icon = "🟡"
        elif 'embedding_preferred' in method:
            icon = "🔵"
        elif 'rule_preferred' in method:
            icon = "🟠"
        elif 'moderate' in method:
            icon = "🟡"
        else:
            icon = "🔴"
        
        print(f"   {i+1}. {icon} {source_field} → {mapping['target_field']}")
        print(f"      Confidence: {confidence:.3f} | Agreement: {agreement:.3f} | Method: {method}")
        
        # Show target selection details
        if rule_target != embedding_target:
            print(f"      Target Selection: {target_selection}")
            print(f"      Rule Target: {rule_target} | Embedding Target: {embedding_target}")
        
        # Show uncertainty factors if any
        uncertainty_factors = mapping['confidence_breakdown']['uncertainty_factors']
        if uncertainty_factors:
            print(f"      Uncertainty: {', '.join(uncertainty_factors)}")
    
    if len(sorted_mappings) > 15:
        print(f"   ... and {len(sorted_mappings) - 15} more mappings")

def print_target_disagreement_analysis(results: Dict[str, Any]):
    """Print detailed analysis of target disagreements."""
    
    print(f"\n" + "=" * 80)
    print("🎯 TARGET DISAGREEMENT ANALYSIS")
    print("=" * 80)
    
    mappings = results["mappings"]
    
    # Find fields with target disagreements
    disagreement_fields = []
    same_target_fields = []
    
    for source_field, mapping in mappings.items():
        rule_target = mapping['confidence_breakdown']['rule_target']
        embedding_target = mapping['confidence_breakdown']['embedding_target']
        
        if rule_target != embedding_target:
            disagreement_fields.append((source_field, mapping))
        else:
            same_target_fields.append((source_field, mapping))
    
    print(f"\n📊 Target Agreement Statistics:")
    print(f"   • Same Targets: {len(same_target_fields)} fields ({len(same_target_fields)/len(mappings)*100:.1f}%)")
    print(f"   • Different Targets: {len(disagreement_fields)} fields ({len(disagreement_fields)/len(mappings)*100:.1f}%)")
    
    print(f"\n🔍 Target Disagreement Examples:")
    for i, (source_field, mapping) in enumerate(disagreement_fields[:10]):
        rule_target = mapping['confidence_breakdown']['rule_target']
        embedding_target = mapping['confidence_breakdown']['embedding_target']
        final_target = mapping['target_field']
        target_selection = mapping['confidence_breakdown']['target_selection_method']
        confidence = mapping['confidence']
        
        print(f"   {i+1}. {source_field}")
        print(f"      Rule Target: {rule_target}")
        print(f"      Embedding Target: {embedding_target}")
        print(f"      Final Target: {final_target}")
        print(f"      Selection Method: {target_selection}")
        print(f"      Final Confidence: {confidence:.3f}")
        
        # Show which engine's target was chosen
        if final_target == rule_target:
            print(f"      Decision: Rule engine target chosen")
        elif final_target == embedding_target:
            print(f"      Decision: Embedding engine target chosen")
        else:
            print(f"      Decision: Neither engine target chosen")
        
        print()
    
    if len(disagreement_fields) > 10:
        print(f"   ... and {len(disagreement_fields) - 10} more disagreement cases")
    
    print(f"\n📈 Target Selection Method Distribution:")
    selection_methods = {}
    for source_field, mapping in disagreement_fields:
        method = mapping['confidence_breakdown']['target_selection_method']
        selection_methods[method] = selection_methods.get(method, 0) + 1
    
    for method, count in selection_methods.items():
        print(f"   • {method.replace('_', ' ').title()}: {count} fields ({count/len(disagreement_fields)*100:.1f}%)")

def compare_sequential_vs_parallel():
    """Compare sequential vs parallel processing approaches."""
    
    print(f"\n" + "=" * 80)
    print("🔄 SEQUENTIAL vs PARALLEL COMPARISON")
    print("=" * 80)
    
    # Simulate sequential processing times
    rule_time = 5.0  # seconds
    embedding_time = 8.0  # seconds
    fields_needing_embedding = 0.3  # 30% of fields
    
    sequential_time = rule_time + (embedding_time * fields_needing_embedding)
    parallel_time = max(rule_time, embedding_time)
    
    print(f"\n⏱️  Processing Time Comparison:")
    print(f"   • Sequential Approach: {sequential_time:.1f} seconds")
    print(f"   • Parallel Approach: {parallel_time:.1f} seconds")
    print(f"   • Time Savings: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")
    
    print(f"\n📊 Quality Comparison:")
    print(f"   • Sequential: Single confidence score per field")
    print(f"   • Parallel: Aggregated confidence with agreement analysis")
    print(f"   • Parallel Advantage: Multi-factor confidence assessment")
    
    print(f"\n🎯 Target Selection Comparison:")
    print(f"   • Sequential: Binary choice (rule OR embedding)")
    print(f"   • Parallel: Intelligent target selection with disagreement penalties")
    print(f"   • Parallel Advantage: Better handling of different targets")
    
    print(f"\n💡 Key Benefits of Parallel Approach:")
    print(f"   • Faster processing for large datasets")
    print(f"   • Better confidence assessment through aggregation")
    print(f"   • Agreement detection for quality assurance")
    print(f"   • Intelligent target selection for different engines")
    print(f"   • Disagreement penalties for uncertainty handling")

async def main():
    """Main demonstration function."""
    
    print("⚡ PARALLEL MAPPING PIPELINE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows parallel processing with confidence aggregation:")
    print("• Rule-based and embedding engines run simultaneously")
    print("• Intelligent confidence score aggregation")
    print("• Agreement-based adjustments")
    print("• Context-aware weight selection")
    print("• Target field selection for different engines")
    print("• Disagreement penalties for uncertainty handling")
    print("=" * 80)
    
    # Create demo schemas
    source_schema, target_schema = create_demo_schemas()
    
    print(f"\n📊 Dataset Information:")
    print(f"   • Source Schema: {len(source_schema)} fields")
    print(f"   • Target Schema: {len(target_schema)} fields")
    print(f"   • Total Possible Combinations: {len(source_schema) * len(target_schema)}")
    
    # Initialize mock engines
    rule_engine = MockRuleBasedEngine()
    embedding_engine = MockEmbeddingEngine()
    
    # Initialize parallel pipeline
    pipeline = ParallelMappingPipeline(rule_engine, embedding_engine)
    
    # Execute parallel mapping
    print(f"\n🔄 Executing Parallel Mapping Pipeline...")
    
    try:
        results = await pipeline.execute_parallel_mapping(source_schema, target_schema)
        
        # Print results
        print_parallel_results(results)
        
        # Print target disagreement analysis
        print_target_disagreement_analysis(results)
        
        # Compare approaches
        compare_sequential_vs_parallel()
        
        # Save results to file
        output_file = "data/reports/parallel_mapping_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = {
            "mappings": results["mappings"],
            "performance_metrics": results["performance_metrics"],
            "processing_time": results["processing_time"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 