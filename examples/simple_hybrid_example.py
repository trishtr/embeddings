#!/usr/bin/env python3
"""
Simple Hybrid Mapping Example

This shows the core concepts of the hybrid approach in a simplified way.
"""

def calculate_pre_embedding_similarity(source_field, target_field):
    """Simple pre-embedding similarity calculation."""
    # Convert to words
    source_words = set(source_field.lower().replace('_', ' ').split())
    target_words = set(target_field.lower().replace('_', ' ').split())
    
    # Jaccard similarity
    intersection = len(source_words & target_words)
    union = len(source_words | target_words)
    
    return intersection / union if union > 0 else 0.0

def categorize_by_confidence(similarity_score):
    """Categorize mapping by confidence level."""
    if similarity_score >= 0.8:
        return "HIGH"
    elif similarity_score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def process_hybrid_mapping(source_fields, target_fields):
    """Main hybrid mapping function."""
    
    print("🔍 Step 1: Pre-Embedding Analysis")
    print("-" * 40)
    
    # Calculate similarities for all combinations
    all_similarities = {}
    for source_field in source_fields:
        all_similarities[source_field] = {}
        for target_field in target_fields:
            similarity = calculate_pre_embedding_similarity(source_field, target_field)
            all_similarities[source_field][target_field] = similarity
    
    # Find best matches and categorize
    categorized_mappings = {
        "HIGH": [],
        "MEDIUM": [],
        "LOW": []
    }
    
    for source_field, similarities in all_similarities.items():
        # Find best match
        best_target = max(similarities.items(), key=lambda x: x[1])
        confidence = best_target[1]
        tier = categorize_by_confidence(confidence)
        
        mapping = {
            "source": source_field,
            "target": best_target[0],
            "confidence": confidence,
            "tier": tier
        }
        
        categorized_mappings[tier].append(mapping)
    
    # Display results
    print(f"📊 Results by Tier:")
    print(f"   HIGH confidence (≥0.8): {len(categorized_mappings['HIGH'])} fields")
    print(f"   MEDIUM confidence (0.5-0.8): {len(categorized_mappings['MEDIUM'])} fields")
    print(f"   LOW confidence (<0.5): {len(categorized_mappings['LOW'])} fields")
    
    print("\n⚡ Step 2: Strategic Processing")
    print("-" * 40)
    
    # Process each tier
    final_mappings = []
    
    # HIGH tier: Accept immediately
    print("   🟢 HIGH tier: Accepting immediately (no embedding needed)")
    for mapping in categorized_mappings["HIGH"]:
        mapping["method"] = "pre_embedding_immediate"
        mapping["processing_time"] = 0.0
        final_mappings.append(mapping)
        print(f"      ✅ {mapping['source']} → {mapping['target']} (confidence: {mapping['confidence']:.3f})")
    
    # MEDIUM tier: Embedding verification
    print("   🟡 MEDIUM tier: Using embedding verification")
    for mapping in categorized_mappings["MEDIUM"]:
        mapping["method"] = "embedding_verification"
        mapping["processing_time"] = 0.1  # Simulated time
        final_mappings.append(mapping)
        print(f"      🔍 {mapping['source']} → {mapping['target']} (confidence: {mapping['confidence']:.3f})")
    
    # LOW tier: Full embedding analysis
    print("   🔴 LOW tier: Full embedding analysis required")
    for mapping in categorized_mappings["LOW"]:
        mapping["method"] = "full_embedding_analysis"
        mapping["processing_time"] = 0.5  # Simulated time
        final_mappings.append(mapping)
        print(f"      🧠 {mapping['source']} → {mapping['target']} (confidence: {mapping['confidence']:.3f})")
    
    return final_mappings

def main():
    """Demonstrate the hybrid approach."""
    
    # Real-world healthcare fields
    source_fields = [
        "provider_npi",
        "specialty_code", 
        "phone_number",
        "email_address",
        "practice_name",
        "license_number",
        "tax_id",
        "date_of_birth",
        "gender",
        "city",
        "state",
        "zip_code",
        "website_url",
        "fax_number",
        "education",
        "certifications",
        "experience_years",
        "hospital_affiliations",
        "research_interests",
        "publications",
        "awards",
        "patient_ratings",
        "wait_time",
        "appointment_types",
        "special_services",
        "equipment_available",
        "staff_count",
        "patient_volume",
        "accreditation_status",
        "compliance_score",
        "audit_history",
        "training_completed",
        "continuing_education",
        "malpractice_history",
        "disciplinary_actions",
        "peer_reviews",
        "patient_outcomes",
        "quality_metrics",
        "satisfaction_scores",
        "accessibility_features",
        "telemedicine_available",
        "after_hours_contact",
        "holiday_schedule",
        "payment_methods",
        "financial_assistance",
        "insurance_accepted",
        "languages_spoken",
        "accepting_patients"
    ]
    
    target_fields = [
        "npi_number",
        "medical_specialty",
        "contact_phone", 
        "email",
        "facility_name",
        "license_id",
        "employer_id",
        "birth_date",
        "sex",
        "city_name",
        "state_code",
        "postal_code",
        "web_site",
        "fax_contact",
        "medical_education",
        "professional_certifications",
        "years_experience",
        "hospital_connections",
        "research_focus",
        "published_works",
        "professional_awards",
        "patient_satisfaction",
        "appointment_wait_time",
        "visit_types",
        "specialized_services",
        "medical_equipment",
        "team_size",
        "patient_load",
        "accreditation_level",
        "quality_score",
        "audit_records",
        "training_status",
        "ce_credits",
        "malpractice_record",
        "disciplinary_history",
        "colleague_reviews",
        "treatment_outcomes",
        "performance_metrics",
        "satisfaction_ratings",
        "accessibility_options",
        "virtual_visits",
        "after_hours_phone",
        "holiday_hours",
        "payment_options",
        "financial_support",
        "insurance_providers",
        "spoken_languages",
        "new_patients_welcome"
    ]
    
    print("🏥 HYBRID MAPPING APPROACH - SIMPLIFIED EXAMPLE")
    print("=" * 60)
    print(f"📊 Source fields: {len(source_fields)}")
    print(f"📊 Target fields: {len(target_fields)}")
    print(f"📊 Total combinations: {len(source_fields) * len(target_fields)}")
    print()
    
    # Execute hybrid mapping
    results = process_hybrid_mapping(source_fields, target_fields)
    
    # Summary
    print("\n📈 SUMMARY")
    print("-" * 40)
    
    high_count = len([r for r in results if r["tier"] == "HIGH"])
    medium_count = len([r for r in results if r["tier"] == "MEDIUM"])
    low_count = len([r for r in results if r["tier"] == "LOW"])
    
    total_time = sum(r["processing_time"] for r in results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    print(f"✅ Total mappings: {len(results)}")
    print(f"🟢 High confidence (immediate): {high_count}")
    print(f"🟡 Medium confidence (verification): {medium_count}")
    print(f"🔴 Low confidence (full analysis): {low_count}")
    print(f"⏱️  Total processing time: {total_time:.2f}s")
    print(f"📊 Average confidence: {avg_confidence:.3f}")
    print(f"💡 Efficiency: {high_count/len(results)*100:.1f}% fields processed immediately")

if __name__ == "__main__":
    main() 