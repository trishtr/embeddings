#!/usr/bin/env python3
"""
Hybrid Mapping Approach for Large Datasets

This example demonstrates the real-world implementation of the hybrid approach:
1. Pre-embedding similarity analysis (fast, lightweight)
2. Tiered embedding application based on confidence scores
3. Strategic resource allocation for large datasets

Author: Schema Mapping System
Date: 2024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, List, Tuple, Any
import time
import json
from dataclasses import dataclass
from enum import Enum

class ConfidenceTier(Enum):
    HIGH = "high"      # 0.8+
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # <0.5

@dataclass
class MappingResult:
    source_field: str
    target_field: str
    confidence: float
    tier: ConfidenceTier
    processing_time: float
    method: str  # "pre_embedding", "embedding_verification", "full_embedding"

class HybridMappingStrategy:
    """
    Implements the hybrid mapping approach for large datasets.
    """
    
    def __init__(self, embedding_handler, data_profiler):
        self.embedding_handler = embedding_handler
        self.data_profiler = data_profiler
        self.results = []
        self.processing_stats = {
            "high_confidence": {"count": 0, "time": 0.0},
            "medium_confidence": {"count": 0, "time": 0.0},
            "low_confidence": {"count": 0, "time": 0.0}
        }
    
    def execute_hybrid_mapping(self, source_schema: Dict[str, str], 
                             target_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute the complete hybrid mapping strategy.
        """
        print("🚀 Starting Hybrid Mapping Strategy")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Pre-embedding similarity analysis
        print("\n📊 Step 1: Pre-Embedding Similarity Analysis")
        pre_embedding_results = self._perform_pre_embedding_analysis(source_schema, target_schema)
        
        # Step 2: Categorize results by confidence tiers
        print("\n🏷️  Step 2: Categorizing by Confidence Tiers")
        tiered_results = self._categorize_by_confidence_tiers(pre_embedding_results)
        
        # Step 3: Process each tier strategically
        print("\n⚡ Step 3: Strategic Processing by Tier")
        final_results = self._process_tiers_strategically(tiered_results, source_schema, target_schema)
        
        # Step 4: Generate comprehensive report
        print("\n📈 Step 4: Generating Final Report")
        final_report = self._generate_final_report(final_results, time.time() - start_time)
        
        return final_report
    
    def _perform_pre_embedding_analysis(self, source_schema: Dict[str, str], 
                                      target_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform pre-embedding similarity analysis using data profiler.
        """
        print("   Performing similarity analysis for all field combinations...")
        
        # Mock source and target statistics for demonstration
        source_stats = self._generate_mock_stats(source_schema)
        target_stats = self._generate_mock_stats(target_schema)
        
        # Calculate similarities for all combinations
        similarities = {}
        for source_field in source_schema.keys():
            similarities[source_field] = {}
            for target_field in target_schema.keys():
                similarity = self._calculate_field_similarity(
                    source_field, target_field,
                    source_stats.get(source_field, {}),
                    target_stats.get(target_field, {})
                )
                similarities[source_field][target_field] = similarity
        
        # Generate potential mappings
        potential_mappings = self._generate_potential_mappings(similarities)
        
        return {
            "similarities": similarities,
            "potential_mappings": potential_mappings
        }
    
    def _calculate_field_similarity(self, source_field: str, target_field: str,
                                  source_stats: Dict, target_stats: Dict) -> float:
        """
        Calculate similarity using the same logic as data profiler.
        """
        # Semantic similarity (50% weight)
        semantic_sim = self._get_string_similarity(source_field, target_field)
        
        # Type similarity (30% weight)
        type_sim = self._get_type_similarity(source_stats.get('type', ''), target_stats.get('type', ''))
        
        # Pattern similarity (20% weight)
        pattern_sim = self._get_pattern_similarity(source_stats, target_stats)
        
        # Weighted combination
        similarity = (0.5 * semantic_sim + 0.3 * type_sim + 0.2 * pattern_sim)
        return similarity
    
    def _get_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaccard similarity between field names."""
        str1_lower = str1.lower().replace('_', ' ')
        str2_lower = str2.lower().replace('_', ' ')
        
        set1 = set(str1_lower.split())
        set2 = set(str2_lower.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate type similarity."""
        if not type1 or not type2:
            return 0.0
        
        type1_lower = type1.lower()
        type2_lower = type2.lower()
        
        if type1_lower == type2_lower:
            return 1.0
        
        if any(t in type1_lower for t in ['varchar', 'text', 'string']) and \
           any(t in type2_lower for t in ['varchar', 'text', 'string']):
            return 0.8
        
        if any(t in type1_lower for t in ['int', 'integer', 'number']) and \
           any(t in type2_lower for t in ['int', 'integer', 'number']):
            return 0.8
        
        return 0.0
    
    def _get_pattern_similarity(self, source_stats: Dict, target_stats: Dict) -> float:
        """Calculate pattern similarity."""
        source_has_numeric = 'min' in source_stats and 'max' in source_stats
        target_has_numeric = 'min' in target_stats and 'max' in target_stats
        
        if source_has_numeric == target_has_numeric:
            return 0.8
        return 0.2
    
    def _generate_potential_mappings(self, similarities: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate potential mappings based on similarities."""
        potential_mappings = {}
        
        for source_field, target_similarities in similarities.items():
            sorted_targets = sorted(target_similarities.items(), key=lambda x: x[1], reverse=True)
            
            potential_mappings[source_field] = {
                "top_matches": sorted_targets[:3],
                "best_match": sorted_targets[0] if sorted_targets else None,
                "confidence": sorted_targets[0][1] if sorted_targets else 0.0
            }
        
        return potential_mappings
    
    def _categorize_by_confidence_tiers(self, pre_embedding_results: Dict[str, Any]) -> Dict[str, List]:
        """
        Categorize results into confidence tiers.
        """
        tiered_results = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        potential_mappings = pre_embedding_results["potential_mappings"]
        
        for source_field, mapping_info in potential_mappings.items():
            confidence = mapping_info["confidence"]
            best_match = mapping_info["best_match"]
            
            if best_match:
                target_field = best_match[0]
                
                result = MappingResult(
                    source_field=source_field,
                    target_field=target_field,
                    confidence=confidence,
                    tier=self._get_confidence_tier(confidence),
                    processing_time=0.0,
                    method="pre_embedding"
                )
                
                if confidence >= 0.8:
                    tiered_results["high"].append(result)
                elif confidence >= 0.5:
                    tiered_results["medium"].append(result)
                else:
                    tiered_results["low"].append(result)
        
        return tiered_results
    
    def _get_confidence_tier(self, confidence: float) -> ConfidenceTier:
        """Determine confidence tier based on score."""
        if confidence >= 0.8:
            return ConfidenceTier.HIGH
        elif confidence >= 0.5:
            return ConfidenceTier.MEDIUM
        else:
            return ConfidenceTier.LOW
    
    def _process_tiers_strategically(self, tiered_results: Dict[str, List], 
                                   source_schema: Dict[str, str], 
                                   target_schema: Dict[str, str]) -> List[MappingResult]:
        """
        Process each tier using appropriate strategy.
        """
        final_results = []
        
        # Process HIGH confidence tier (accept immediately)
        print(f"   Processing HIGH confidence tier ({len(tiered_results['high'])} fields)")
        start_time = time.time()
        for result in tiered_results["high"]:
            result.processing_time = 0.0  # Immediate acceptance
            result.method = "pre_embedding_immediate"
            final_results.append(result)
            self.processing_stats["high_confidence"]["count"] += 1
        
        self.processing_stats["high_confidence"]["time"] = time.time() - start_time
        print(f"   ✅ Accepted {len(tiered_results['high'])} high-confidence mappings")
        
        # Process MEDIUM confidence tier (embedding verification)
        print(f"   Processing MEDIUM confidence tier ({len(tiered_results['medium'])} fields)")
        start_time = time.time()
        medium_results = self._process_medium_confidence_tier(tiered_results["medium"], source_schema, target_schema)
        final_results.extend(medium_results)
        self.processing_stats["medium_confidence"]["time"] = time.time() - start_time
        print(f"   ✅ Processed {len(medium_results)} medium-confidence mappings")
        
        # Process LOW confidence tier (full embedding analysis)
        print(f"   Processing LOW confidence tier ({len(tiered_results['low'])} fields)")
        start_time = time.time()
        low_results = self._process_low_confidence_tier(tiered_results["low"], source_schema, target_schema)
        final_results.extend(low_results)
        self.processing_stats["low_confidence"]["time"] = time.time() - start_time
        print(f"   ✅ Processed {len(low_results)} low-confidence mappings")
        
        return final_results
    
    def _process_medium_confidence_tier(self, medium_results: List[MappingResult], 
                                      source_schema: Dict[str, str], 
                                      target_schema: Dict[str, str]) -> List[MappingResult]:
        """
        Process medium confidence tier with embedding verification.
        """
        processed_results = []
        
        for result in medium_results:
            start_time = time.time()
            
            # Generate embeddings for verification
            source_context = f"database field {result.source_field} of type {source_schema[result.source_field]}"
            target_context = f"database field {result.target_field} of type {target_schema[result.target_field]}"
            
            # Mock embedding verification (in real implementation, use actual embeddings)
            embedding_similarity = self._mock_embedding_verification(source_context, target_context)
            
            # Update result based on embedding verification
            if embedding_similarity > result.confidence:
                result.confidence = embedding_similarity
                result.method = "embedding_verification_improved"
            elif embedding_similarity < result.confidence * 0.8:  # Significant drop
                result.method = "embedding_verification_rejected"
                continue  # Skip this mapping
            
            result.processing_time = time.time() - start_time
            processed_results.append(result)
            self.processing_stats["medium_confidence"]["count"] += 1
        
        return processed_results
    
    def _process_low_confidence_tier(self, low_results: List[MappingResult], 
                                   source_schema: Dict[str, str], 
                                   target_schema: Dict[str, str]) -> List[MappingResult]:
        """
        Process low confidence tier with full embedding analysis.
        """
        processed_results = []
        
        for result in low_results:
            start_time = time.time()
            
            # Full embedding analysis
            source_context = f"database field {result.source_field} of type {source_schema[result.source_field]}"
            
            # Find best match using embeddings
            best_embedding_match = self._find_best_embedding_match(result.source_field, target_schema, source_context)
            
            if best_embedding_match:
                result.target_field = best_embedding_match[0]
                result.confidence = best_embedding_match[1]
                result.method = "full_embedding_analysis"
            else:
                result.method = "full_embedding_no_match"
                continue  # Skip this mapping
            
            result.processing_time = time.time() - start_time
            processed_results.append(result)
            self.processing_stats["low_confidence"]["count"] += 1
        
        return processed_results
    
    def _mock_embedding_verification(self, source_context: str, target_context: str) -> float:
        """
        Mock embedding verification (in real implementation, use actual embeddings).
        """
        # Simulate embedding similarity calculation
        import random
        base_similarity = 0.6
        variation = random.uniform(-0.1, 0.2)
        return min(1.0, max(0.0, base_similarity + variation))
    
    def _find_best_embedding_match(self, source_field: str, target_schema: Dict[str, str], 
                                 source_context: str) -> Tuple[str, float]:
        """
        Find best match using full embedding analysis.
        """
        # Mock embedding analysis (in real implementation, use actual embeddings)
        import random
        
        best_match = None
        best_score = 0.0
        
        for target_field in target_schema.keys():
            # Simulate embedding similarity
            similarity = random.uniform(0.1, 0.8)
            if similarity > best_score:
                best_score = similarity
                best_match = target_field
        
        return (best_match, best_score) if best_match else None
    
    def _generate_final_report(self, final_results: List[MappingResult], total_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive final report.
        """
        # Calculate statistics
        total_fields = len(final_results)
        high_confidence_count = len([r for r in final_results if r.tier == ConfidenceTier.HIGH])
        medium_confidence_count = len([r for r in final_results if r.tier == ConfidenceTier.MEDIUM])
        low_confidence_count = len([r for r in final_results if r.tier == ConfidenceTier.LOW])
        
        avg_confidence = sum(r.confidence for r in final_results) / total_fields if total_fields > 0 else 0.0
        
        # Calculate time savings
        total_processing_time = sum(r.processing_time for r in final_results)
        
        report = {
            "summary": {
                "total_fields_processed": total_fields,
                "high_confidence_mappings": high_confidence_count,
                "medium_confidence_mappings": medium_confidence_count,
                "low_confidence_mappings": low_confidence_count,
                "average_confidence": round(avg_confidence, 3),
                "total_processing_time": round(total_time, 2),
                "total_embedding_time": round(total_processing_time, 2)
            },
            "processing_stats": self.processing_stats,
            "mappings": [
                {
                    "source_field": r.source_field,
                    "target_field": r.target_field,
                    "confidence": round(r.confidence, 3),
                    "tier": r.tier.value,
                    "method": r.method,
                    "processing_time": round(r.processing_time, 3)
                }
                for r in final_results
            ],
            "performance_metrics": {
                "time_savings_percentage": round((1 - total_processing_time / total_time) * 100, 1),
                "efficiency_score": round(high_confidence_count / total_fields * 100, 1),
                "quality_score": round(avg_confidence * 100, 1)
            }
        }
        
        return report
    
    def _generate_mock_stats(self, schema: Dict[str, str]) -> Dict[str, Dict]:
        """Generate mock statistics for demonstration."""
        stats = {}
        for field_name, field_type in schema.items():
            stats[field_name] = {
                "type": field_type,
                "count": 1000,
                "unique_count": 800,
                "null_count": 50
            }
            
            # Add numeric stats for some fields
            if any(word in field_name.lower() for word in ['id', 'number', 'code']):
                stats[field_name].update({
                    "min": "1000000000",
                    "max": "1999999999",
                    "mean": "1500000000"
                })
        
        return stats

def demonstrate_hybrid_approach():
    """
    Demonstrate the hybrid mapping approach with a real-world scenario.
    """
    print("🏥 HYBRID MAPPING APPROACH DEMONSTRATION")
    print("=" * 80)
    
    # Mock schemas (simplified version of our real-world scenario)
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
        "medical_specialty": "VARCHAR(50)",
        "contact_phone": "VARCHAR(15)",
        "email": "VARCHAR(100)",
        "facility_name": "VARCHAR(200)",
        "license_id": "VARCHAR(20)",
        "employer_id": "VARCHAR(20)",
        "birth_date": "DATE",
        "sex": "VARCHAR(10)",
        "city_name": "VARCHAR(50)",
        "state_code": "VARCHAR(2)",
        "postal_code": "VARCHAR(10)",
        "web_site": "VARCHAR(200)",
        "fax_contact": "VARCHAR(20)",
        "medical_education": "VARCHAR(200)",
        "professional_certifications": "VARCHAR(500)",
        "years_experience": "INTEGER",
        "hospital_connections": "VARCHAR(500)",
        "research_focus": "VARCHAR(500)",
        "published_works": "VARCHAR(1000)",
        "professional_awards": "VARCHAR(500)",
        "patient_satisfaction": "DECIMAL(3,2)",
        "appointment_wait_time": "INTEGER",
        "visit_types": "VARCHAR(200)",
        "specialized_services": "VARCHAR(500)",
        "medical_equipment": "VARCHAR(500)",
        "team_size": "INTEGER",
        "patient_load": "INTEGER",
        "accreditation_level": "VARCHAR(50)",
        "quality_score": "DECIMAL(5,2)",
        "audit_records": "VARCHAR(1000)",
        "training_status": "VARCHAR(500)",
        "ce_credits": "VARCHAR(500)",
        "malpractice_record": "VARCHAR(1000)",
        "disciplinary_history": "VARCHAR(1000)",
        "colleague_reviews": "VARCHAR(1000)",
        "treatment_outcomes": "VARCHAR(500)",
        "performance_metrics": "VARCHAR(500)",
        "satisfaction_ratings": "DECIMAL(3,2)",
        "accessibility_options": "VARCHAR(500)",
        "virtual_visits": "BOOLEAN",
        "after_hours_phone": "VARCHAR(200)",
        "holiday_hours": "VARCHAR(500)",
        "payment_options": "VARCHAR(500)",
        "financial_support": "VARCHAR(500)",
        "insurance_providers": "VARCHAR(1000)",
        "spoken_languages": "VARCHAR(500)",
        "new_patients_welcome": "BOOLEAN"
    }
    
    print(f"📊 Source Schema: {len(source_schema)} fields")
    print(f"📊 Target Schema: {len(target_schema)} fields")
    print(f"📊 Total Possible Combinations: {len(source_schema) * len(target_schema)}")
    
    # Create mock handlers (in real implementation, use actual handlers)
    class MockEmbeddingHandler:
        def __init__(self):
            pass
    
    class MockDataProfiler:
        def __init__(self):
            pass
    
    # Initialize hybrid strategy
    strategy = HybridMappingStrategy(MockEmbeddingHandler(), MockDataProfiler())
    
    # Execute hybrid mapping
    results = strategy.execute_hybrid_mapping(source_schema, target_schema)
    
    # Display results
    print("\n" + "=" * 80)
    print("📈 FINAL RESULTS")
    print("=" * 80)
    
    summary = results["summary"]
    print(f"\n🎯 Summary:")
    print(f"   • Total Fields Processed: {summary['total_fields_processed']}")
    print(f"   • High Confidence Mappings: {summary['high_confidence_mappings']}")
    print(f"   • Medium Confidence Mappings: {summary['medium_confidence_mappings']}")
    print(f"   • Low Confidence Mappings: {summary['low_confidence_mappings']}")
    print(f"   • Average Confidence: {summary['average_confidence']}")
    print(f"   • Total Processing Time: {summary['total_processing_time']}s")
    
    metrics = results["performance_metrics"]
    print(f"\n⚡ Performance Metrics:")
    print(f"   • Time Savings: {metrics['time_savings_percentage']}%")
    print(f"   • Efficiency Score: {metrics['efficiency_score']}%")
    print(f"   • Quality Score: {metrics['quality_score']}%")
    
    print(f"\n📋 Sample Mappings:")
    for i, mapping in enumerate(results["mappings"][:10]):  # Show first 10
        print(f"   {i+1}. {mapping['source_field']} → {mapping['target_field']} "
              f"(confidence: {mapping['confidence']}, tier: {mapping['tier']})")
    
    if len(results["mappings"]) > 10:
        print(f"   ... and {len(results['mappings']) - 10} more mappings")
    
    print(f"\n💡 Key Insights:")
    print(f"   • {summary['high_confidence_mappings']} fields mapped immediately (no embedding needed)")
    print(f"   • {summary['medium_confidence_mappings']} fields verified with embeddings")
    print(f"   • {summary['low_confidence_mappings']} fields required full embedding analysis")
    print(f"   • Overall efficiency: {metrics['efficiency_score']}% of fields processed optimally")
    
    return results

if __name__ == "__main__":
    results = demonstrate_hybrid_approach() 