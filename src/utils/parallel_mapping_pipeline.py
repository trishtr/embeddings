#!/usr/bin/env python3
"""
Parallel Mapping Pipeline with Confidence Aggregation

This module implements parallel processing of rule-based and embedding engines
with intelligent confidence score aggregation and target field selection.

Author: Schema Mapping System
Date: 2024
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import time

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceBreakdown:
    rule_based_confidence: float
    embedding_confidence: float
    aggregated_confidence: float
    agreement_level: float
    uncertainty_factors: List[str]
    weights_used: Dict[str, float]
    rule_target: str
    embedding_target: str
    final_target: str
    target_selection_method: str

@dataclass
class FieldMapping:
    source_field: str
    target_field: str
    confidence_breakdown: ConfidenceBreakdown
    method: str
    rule_based_result: Dict[str, Any]
    embedding_result: Dict[str, Any]

class ParallelMappingPipeline:
    """
    Parallel mapping pipeline with confidence aggregation and target selection.
    """
    
    def __init__(self, rule_based_engine, embedding_engine):
        self.rule_based_engine = rule_based_engine
        self.embedding_engine = embedding_engine
        
        # Default confidence weights
        self.confidence_weights = {
            "rule_based": 0.4,
            "embedding": 0.6
        }
        
        # Context-aware weight adjustments
        self.field_category_weights = {
            "standard": {"rule_based": 0.6, "embedding": 0.4},
            "domain_specific": {"rule_based": 0.3, "embedding": 0.7},
            "complex": {"rule_based": 0.2, "embedding": 0.8}
        }
        
        # Agreement thresholds
        self.agreement_thresholds = {
            "high_agreement": 0.8,
            "medium_agreement": 0.6,
            "low_agreement": 0.4
        }
        
        # Agreement adjustments
        self.agreement_adjustments = {
            "high_bonus": 0.1,
            "medium_bonus": 0.05,
            "low_penalty": 0.08,
            "very_low_penalty": 0.15
        }
        
        # Target selection thresholds
        self.target_selection_thresholds = {
            "confidence_difference": 0.1,  # Minimum difference to prefer one engine
            "embedding_preference": 0.05   # Small bias toward embedding for semantic understanding
        }
    
    async def execute_parallel_mapping(self, source_schema: Dict[str, str], 
                                     target_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute parallel mapping with confidence aggregation and target selection.
        """
        logger.info("🚀 Starting Parallel Mapping Pipeline")
        start_time = time.time()
        
        # Stage 1: Parallel Processing
        logger.info("📊 Stage 1: Parallel Processing")
        rule_results, embedding_results = await self._parallel_processing(source_schema, target_schema)
        
        # Stage 2: Confidence Aggregation with Target Selection
        logger.info("🔗 Stage 2: Confidence Aggregation with Target Selection")
        aggregated_results = self._aggregate_confidence_scores_with_targets(rule_results, embedding_results)
        
        # Stage 3: Generate Final Mappings
        logger.info("🎯 Stage 3: Generate Final Mappings")
        final_mappings = self._generate_final_mappings(aggregated_results)
        
        # Stage 4: Performance Analysis
        logger.info("📈 Stage 4: Performance Analysis")
        performance_metrics = self._analyze_performance(aggregated_results, time.time() - start_time)
        
        return {
            "mappings": final_mappings,
            "performance_metrics": performance_metrics,
            "aggregated_results": aggregated_results,
            "processing_time": time.time() - start_time
        }
    
    async def _parallel_processing(self, source_schema: Dict[str, str], 
                                 target_schema: Dict[str, str]) -> Tuple[Dict, Dict]:
        """
        Execute rule-based and embedding engines in parallel.
        """
        loop = asyncio.get_event_loop()
        
        logger.info(f"🔄 Starting parallel processing for {len(source_schema)} fields")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            rule_future = loop.run_in_executor(
                executor, self.rule_based_engine.process, source_schema, target_schema
            )
            embedding_future = loop.run_in_executor(
                executor, self.embedding_engine.process, source_schema, target_schema
            )
            
            # Wait for both to complete
            rule_results, embedding_results = await asyncio.gather(rule_future, embedding_future)
        
        logger.info(f"✅ Parallel processing completed")
        logger.info(f"   • Rule-based results: {len(rule_results)} fields")
        logger.info(f"   • Embedding results: {len(embedding_results)} fields")
        
        return rule_results, embedding_results
    
    def _aggregate_confidence_scores_with_targets(self, rule_results: Dict, embedding_results: Dict) -> List[FieldMapping]:
        """
        Aggregate confidence scores with intelligent target selection.
        """
        aggregated_results = []
        
        # Get all unique source fields
        all_source_fields = set(rule_results.keys()) | set(embedding_results.keys())
        
        logger.info(f"🔗 Aggregating confidence scores with target selection for {len(all_source_fields)} fields")
        
        for source_field in all_source_fields:
            # Get results from both engines
            rule_result = rule_results.get(source_field, {})
            embedding_result = embedding_results.get(source_field, {})
            
            rule_score = rule_result.get("confidence", 0.0)
            embedding_score = embedding_result.get("confidence", 0.0)
            
            # Get target fields
            rule_target = rule_result.get("target_field", f"target_{source_field}")
            embedding_target = embedding_result.get("target_field", f"target_{source_field}")
            
            # Calculate agreement level
            agreement_level = 1.0 - abs(rule_score - embedding_score)
            
            # Get context-aware weights
            weights = self._get_context_aware_weights(source_field)
            
            # Select final target field
            final_target, target_selection_method = self._select_target_field(
                rule_score, embedding_score, rule_target, embedding_target, weights
            )
            
            # Calculate aggregated confidence
            aggregated_score = self._calculate_aggregated_confidence(
                rule_score, embedding_score, weights, agreement_level, 
                rule_target, embedding_target, final_target
            )
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_uncertainty_factors(
                rule_score, embedding_score, agreement_level, rule_target, embedding_target
            )
            
            # Create confidence breakdown
            confidence_breakdown = ConfidenceBreakdown(
                rule_based_confidence=rule_score,
                embedding_confidence=embedding_score,
                aggregated_confidence=aggregated_score,
                agreement_level=agreement_level,
                uncertainty_factors=uncertainty_factors,
                weights_used=weights,
                rule_target=rule_target,
                embedding_target=embedding_target,
                final_target=final_target,
                target_selection_method=target_selection_method
            )
            
            # Determine method
            method = self._determine_mapping_method(confidence_breakdown)
            
            # Create field mapping
            field_mapping = FieldMapping(
                source_field=source_field,
                target_field=final_target,
                confidence_breakdown=confidence_breakdown,
                method=method,
                rule_based_result=rule_result,
                embedding_result=embedding_result
            )
            
            aggregated_results.append(field_mapping)
        
        # Sort by aggregated confidence (highest first)
        aggregated_results.sort(key=lambda x: x.confidence_breakdown.aggregated_confidence, reverse=True)
        
        logger.info(f"✅ Confidence aggregation with target selection completed")
        logger.info(f"   • High confidence (≥0.8): {len([r for r in aggregated_results if r.confidence_breakdown.aggregated_confidence >= 0.8])}")
        logger.info(f"   • Medium confidence (0.6-0.8): {len([r for r in aggregated_results if 0.6 <= r.confidence_breakdown.aggregated_confidence < 0.8])}")
        logger.info(f"   • Low confidence (<0.6): {len([r for r in aggregated_results if r.confidence_breakdown.aggregated_confidence < 0.6])}")
        
        return aggregated_results
    
    def _select_target_field(self, rule_score: float, embedding_score: float, 
                           rule_target: str, embedding_target: str, 
                           weights: Dict[str, float]) -> Tuple[str, str]:
        """
        Select the final target field based on confidence scores and field characteristics.
        """
        # Check if targets are the same
        if rule_target == embedding_target:
            return rule_target, "same_targets"
        
        # Calculate confidence difference
        confidence_difference = abs(rule_score - embedding_score)
        
        # If confidence difference is small, prefer embedding for semantic understanding
        if confidence_difference < self.target_selection_thresholds["confidence_difference"]:
            # Small bias toward embedding for semantic understanding
            if embedding_score + self.target_selection_thresholds["embedding_preference"] > rule_score:
                return embedding_target, "embedding_preferred_semantic"
            else:
                return rule_target, "rule_preferred_pattern"
        
        # If confidence difference is significant, choose the higher confidence
        if embedding_score > rule_score:
            return embedding_target, "embedding_higher_confidence"
        else:
            return rule_target, "rule_higher_confidence"
    
    def _get_context_aware_weights(self, field_name: str) -> Dict[str, float]:
        """
        Get context-aware weights based on field characteristics.
        """
        field_lower = field_name.lower()
        
        # Standard fields (rule-based engine excels)
        if any(word in field_lower for word in ['email', 'phone', 'name', 'id', 'date', 'address', 'city', 'state', 'zip']):
            return self.field_category_weights["standard"]
        
        # Domain-specific fields (embedding engine excels)
        elif any(word in field_lower for word in ['npi', 'specialty', 'license', 'malpractice', 'accreditation', 'phi']):
            return self.field_category_weights["domain_specific"]
        
        # Complex fields (embedding engine excels)
        elif any(word in field_lower for word in ['research', 'publications', 'history', 'outcomes', 'metrics']):
            return self.field_category_weights["complex"]
        
        # Default weights
        else:
            return self.confidence_weights
    
    def _calculate_aggregated_confidence(self, rule_score: float, embedding_score: float, 
                                       weights: Dict[str, float], agreement_level: float,
                                       rule_target: str, embedding_target: str, 
                                       final_target: str) -> float:
        """
        Calculate aggregated confidence with target disagreement penalties.
        """
        # Calculate weighted average
        base_score = (
            rule_score * weights["rule_based"] + 
            embedding_score * weights["embedding"]
        )
        
        # Apply agreement-based adjustments
        adjusted_score = self._apply_agreement_adjustments(base_score, agreement_level)
        
        # Apply target disagreement penalty if targets differ
        if rule_target != embedding_target:
            target_penalty = self._calculate_target_disagreement_penalty(
                rule_score, embedding_score, rule_target, embedding_target, final_target
            )
            adjusted_score = max(0.0, adjusted_score - target_penalty)
        
        return adjusted_score
    
    def _calculate_target_disagreement_penalty(self, rule_score: float, embedding_score: float,
                                            rule_target: str, embedding_target: str,
                                            final_target: str) -> float:
        """
        Calculate penalty for target disagreement between engines.
        """
        # Base penalty for different targets
        base_penalty = 0.15
        
        # Additional penalty based on confidence difference
        confidence_difference = abs(rule_score - embedding_score)
        if confidence_difference > 0.3:
            base_penalty += 0.05  # Extra penalty for high confidence difference
        
        # Penalty reduction if final target matches higher confidence engine
        if final_target == rule_target and rule_score > embedding_score:
            base_penalty -= 0.03  # Reduce penalty if we chose the higher confidence target
        elif final_target == embedding_target and embedding_score > rule_score:
            base_penalty -= 0.03  # Reduce penalty if we chose the higher confidence target
        
        return max(0.0, base_penalty)
    
    def _apply_agreement_adjustments(self, base_score: float, agreement_level: float) -> float:
        """
        Apply agreement-based adjustments to confidence scores.
        """
        adjusted_score = base_score
        
        # High agreement bonus
        if agreement_level > self.agreement_thresholds["high_agreement"]:
            adjusted_score = min(1.0, adjusted_score + self.agreement_adjustments["high_bonus"])
        # Medium agreement bonus
        elif agreement_level > self.agreement_thresholds["medium_agreement"]:
            adjusted_score = min(1.0, adjusted_score + self.agreement_adjustments["medium_bonus"])
        # Low agreement penalty
        elif agreement_level < self.agreement_thresholds["low_agreement"]:
            adjusted_score = max(0.0, adjusted_score - self.agreement_adjustments["very_low_penalty"])
        # Very low agreement penalty
        elif agreement_level < self.agreement_thresholds["medium_agreement"]:
            adjusted_score = max(0.0, adjusted_score - self.agreement_adjustments["low_penalty"])
        
        return adjusted_score
    
    def _identify_uncertainty_factors(self, rule_score: float, embedding_score: float, 
                                    agreement_level: float, rule_target: str, 
                                    embedding_target: str) -> List[str]:
        """
        Identify factors contributing to uncertainty.
        """
        uncertainty_factors = []
        
        # Engine disagreement
        if agreement_level < self.agreement_thresholds["medium_agreement"]:
            uncertainty_factors.append("engine_disagreement")
        
        # Target disagreement
        if rule_target != embedding_target:
            uncertainty_factors.append("target_disagreement")
        
        # Low rule confidence
        if rule_score < 0.5:
            uncertainty_factors.append("low_rule_confidence")
        
        # Low embedding confidence
        if embedding_score < 0.5:
            uncertainty_factors.append("low_embedding_confidence")
        
        # High score variance
        if abs(rule_score - embedding_score) > 0.3:
            uncertainty_factors.append("high_score_variance")
        
        # Very low agreement
        if agreement_level < self.agreement_thresholds["low_agreement"]:
            uncertainty_factors.append("very_low_agreement")
        
        return uncertainty_factors
    
    def _determine_mapping_method(self, confidence_breakdown: ConfidenceBreakdown) -> str:
        """
        Determine the mapping method based on confidence breakdown.
        """
        confidence = confidence_breakdown.aggregated_confidence
        agreement_level = confidence_breakdown.agreement_level
        target_selection = confidence_breakdown.target_selection_method
        
        if confidence >= 0.8 and agreement_level >= 0.7:
            return "high_confidence_agreement"
        elif confidence >= 0.7:
            return "high_confidence"
        elif agreement_level >= 0.8:
            return "high_agreement"
        elif target_selection == "same_targets":
            return "same_targets"
        elif "embedding" in target_selection:
            return "embedding_preferred"
        elif "rule" in target_selection:
            return "rule_preferred"
        elif len(confidence_breakdown.uncertainty_factors) <= 1:
            return "moderate_confidence"
        else:
            return "low_confidence"
    
    def _generate_final_mappings(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Generate final mappings with detailed information.
        """
        final_mappings = {}
        
        for mapping in aggregated_results:
            final_mappings[mapping.source_field] = {
                "target_field": mapping.target_field,
                "confidence": mapping.confidence_breakdown.aggregated_confidence,
                "method": mapping.method,
                "confidence_breakdown": {
                    "rule_based_confidence": mapping.confidence_breakdown.rule_based_confidence,
                    "embedding_confidence": mapping.confidence_breakdown.embedding_confidence,
                    "agreement_level": mapping.confidence_breakdown.agreement_level,
                    "weights_used": mapping.confidence_breakdown.weights_used,
                    "uncertainty_factors": mapping.confidence_breakdown.uncertainty_factors,
                    "rule_target": mapping.confidence_breakdown.rule_target,
                    "embedding_target": mapping.confidence_breakdown.embedding_target,
                    "target_selection_method": mapping.confidence_breakdown.target_selection_method
                },
                "rule_based_result": mapping.rule_based_result,
                "embedding_result": mapping.embedding_result
            }
        
        return final_mappings
    
    def _analyze_performance(self, aggregated_results: List[FieldMapping], processing_time: float) -> Dict[str, Any]:
        """
        Analyze performance metrics including target selection patterns.
        """
        total_fields = len(aggregated_results)
        
        # Confidence distribution
        high_confidence = len([r for r in aggregated_results if r.confidence_breakdown.aggregated_confidence >= 0.8])
        medium_confidence = len([r for r in aggregated_results if 0.6 <= r.confidence_breakdown.aggregated_confidence < 0.8])
        low_confidence = len([r for r in aggregated_results if r.confidence_breakdown.aggregated_confidence < 0.6])
        
        # Agreement distribution
        high_agreement = len([r for r in aggregated_results if r.confidence_breakdown.agreement_level >= 0.8])
        medium_agreement = len([r for r in aggregated_results if 0.6 <= r.confidence_breakdown.agreement_level < 0.8])
        low_agreement = len([r for r in aggregated_results if r.confidence_breakdown.agreement_level < 0.6])
        
        # Target selection analysis
        same_targets = len([r for r in aggregated_results if r.confidence_breakdown.target_selection_method == "same_targets"])
        embedding_preferred = len([r for r in aggregated_results if "embedding" in r.confidence_breakdown.target_selection_method])
        rule_preferred = len([r for r in aggregated_results if "rule" in r.confidence_breakdown.target_selection_method])
        
        # Uncertainty analysis
        fields_with_uncertainty = len([r for r in aggregated_results if r.confidence_breakdown.uncertainty_factors])
        target_disagreements = len([r for r in aggregated_results if "target_disagreement" in r.confidence_breakdown.uncertainty_factors])
        avg_uncertainty_factors = np.mean([len(r.confidence_breakdown.uncertainty_factors) for r in aggregated_results])
        
        # Average scores
        avg_confidence = np.mean([r.confidence_breakdown.aggregated_confidence for r in aggregated_results])
        avg_agreement = np.mean([r.confidence_breakdown.agreement_level for r in aggregated_results])
        
        # Method distribution
        method_counts = {}
        for mapping in aggregated_results:
            method = mapping.method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "processing_time": processing_time,
            "total_fields": total_fields,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
                "average": round(avg_confidence, 3)
            },
            "agreement_distribution": {
                "high": high_agreement,
                "medium": medium_agreement,
                "low": low_agreement,
                "average": round(avg_agreement, 3)
            },
            "target_selection_analysis": {
                "same_targets": same_targets,
                "embedding_preferred": embedding_preferred,
                "rule_preferred": rule_preferred,
                "target_disagreements": target_disagreements
            },
            "uncertainty_analysis": {
                "fields_with_uncertainty": fields_with_uncertainty,
                "average_uncertainty_factors": round(avg_uncertainty_factors, 2),
                "target_disagreements": target_disagreements
            },
            "method_distribution": method_counts,
            "performance_metrics": {
                "efficiency_score": round(high_confidence / total_fields * 100, 1),
                "quality_score": round(avg_confidence * 100, 1),
                "agreement_score": round(avg_agreement * 100, 1),
                "target_agreement_rate": round(same_targets / total_fields * 100, 1)
            }
        }
    
    def get_detailed_analysis(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Get detailed analysis of the mapping results including target selection patterns.
        """
        analysis = {
            "field_categories": self._analyze_field_categories(aggregated_results),
            "confidence_patterns": self._analyze_confidence_patterns(aggregated_results),
            "agreement_patterns": self._analyze_agreement_patterns(aggregated_results),
            "target_selection_patterns": self._analyze_target_selection_patterns(aggregated_results),
            "uncertainty_patterns": self._analyze_uncertainty_patterns(aggregated_results),
            "recommendations": self._generate_recommendations(aggregated_results)
        }
        
        return analysis
    
    def _analyze_target_selection_patterns(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Analyze patterns in target selection.
        """
        same_targets = [r for r in aggregated_results if r.confidence_breakdown.target_selection_method == "same_targets"]
        embedding_preferred = [r for r in aggregated_results if "embedding" in r.confidence_breakdown.target_selection_method]
        rule_preferred = [r for r in aggregated_results if "rule" in r.confidence_breakdown.target_selection_method]
        
        # Analyze confidence patterns for each selection method
        same_targets_confidence = np.mean([r.confidence_breakdown.aggregated_confidence for r in same_targets]) if same_targets else 0
        embedding_preferred_confidence = np.mean([r.confidence_breakdown.aggregated_confidence for r in embedding_preferred]) if embedding_preferred else 0
        rule_preferred_confidence = np.mean([r.confidence_breakdown.aggregated_confidence for r in rule_preferred]) if rule_preferred else 0
        
        return {
            "same_targets": {
                "count": len(same_targets),
                "average_confidence": round(same_targets_confidence, 3),
                "percentage": round(len(same_targets) / len(aggregated_results) * 100, 1)
            },
            "embedding_preferred": {
                "count": len(embedding_preferred),
                "average_confidence": round(embedding_preferred_confidence, 3),
                "percentage": round(len(embedding_preferred) / len(aggregated_results) * 100, 1)
            },
            "rule_preferred": {
                "count": len(rule_preferred),
                "average_confidence": round(rule_preferred_confidence, 3),
                "percentage": round(len(rule_preferred) / len(aggregated_results) * 100, 1)
            }
        }
    
    def _analyze_field_categories(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Analyze results by field categories.
        """
        categories = {
            "standard": [],
            "domain_specific": [],
            "complex": [],
            "other": []
        }
        
        for mapping in aggregated_results:
            field_name = mapping.source_field.lower()
            
            if any(word in field_name for word in ['email', 'phone', 'name', 'id', 'date', 'address']):
                categories["standard"].append(mapping)
            elif any(word in field_name for word in ['npi', 'specialty', 'license', 'malpractice']):
                categories["domain_specific"].append(mapping)
            elif any(word in field_name for word in ['research', 'publications', 'history', 'outcomes']):
                categories["complex"].append(mapping)
            else:
                categories["other"].append(mapping)
        
        # Calculate averages for each category
        category_analysis = {}
        for category, mappings in categories.items():
            if mappings:
                avg_confidence = np.mean([m.confidence_breakdown.aggregated_confidence for m in mappings])
                avg_agreement = np.mean([m.confidence_breakdown.agreement_level for m in mappings])
                target_disagreements = len([m for m in mappings if "target_disagreement" in m.confidence_breakdown.uncertainty_factors])
                
                category_analysis[category] = {
                    "count": len(mappings),
                    "average_confidence": round(avg_confidence, 3),
                    "average_agreement": round(avg_agreement, 3),
                    "target_disagreements": target_disagreements
                }
        
        return category_analysis
    
    def _analyze_confidence_patterns(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Analyze confidence patterns.
        """
        confidences = [r.confidence_breakdown.aggregated_confidence for r in aggregated_results]
        
        return {
            "distribution": {
                "min": min(confidences),
                "max": max(confidences),
                "mean": np.mean(confidences),
                "median": np.median(confidences),
                "std": np.std(confidences)
            },
            "percentiles": {
                "25th": np.percentile(confidences, 25),
                "50th": np.percentile(confidences, 50),
                "75th": np.percentile(confidences, 75),
                "90th": np.percentile(confidences, 90)
            }
        }
    
    def _analyze_agreement_patterns(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Analyze agreement patterns.
        """
        agreements = [r.confidence_breakdown.agreement_level for r in aggregated_results]
        
        return {
            "distribution": {
                "min": min(agreements),
                "max": max(agreements),
                "mean": np.mean(agreements),
                "median": np.median(agreements),
                "std": np.std(agreements)
            },
            "high_agreement_fields": [
                r.source_field for r in aggregated_results 
                if r.confidence_breakdown.agreement_level >= 0.8
            ],
            "low_agreement_fields": [
                r.source_field for r in aggregated_results 
                if r.confidence_breakdown.agreement_level < 0.4
            ]
        }
    
    def _analyze_uncertainty_patterns(self, aggregated_results: List[FieldMapping]) -> Dict[str, Any]:
        """
        Analyze uncertainty patterns.
        """
        uncertainty_factors = {}
        for mapping in aggregated_results:
            for factor in mapping.confidence_breakdown.uncertainty_factors:
                uncertainty_factors[factor] = uncertainty_factors.get(factor, 0) + 1
        
        high_uncertainty_fields = [
            r.source_field for r in aggregated_results 
            if len(r.confidence_breakdown.uncertainty_factors) >= 2
        ]
        
        target_disagreement_fields = [
            r.source_field for r in aggregated_results 
            if "target_disagreement" in r.confidence_breakdown.uncertainty_factors
        ]
        
        return {
            "factor_frequency": uncertainty_factors,
            "high_uncertainty_fields": high_uncertainty_fields,
            "target_disagreement_fields": target_disagreement_fields,
            "uncertainty_distribution": {
                "no_uncertainty": len([r for r in aggregated_results if not r.confidence_breakdown.uncertainty_factors]),
                "low_uncertainty": len([r for r in aggregated_results if len(r.confidence_breakdown.uncertainty_factors) == 1]),
                "high_uncertainty": len([r for r in aggregated_results if len(r.confidence_breakdown.uncertainty_factors) >= 2])
            }
        }
    
    def _generate_recommendations(self, aggregated_results: List[FieldMapping]) -> List[str]:
        """
        Generate recommendations based on analysis.
        """
        recommendations = []
        
        # Analyze overall performance
        avg_confidence = np.mean([r.confidence_breakdown.aggregated_confidence for r in aggregated_results])
        avg_agreement = np.mean([r.confidence_breakdown.agreement_level for r in aggregated_results])
        
        # Target disagreement analysis
        target_disagreements = len([r for r in aggregated_results if "target_disagreement" in r.confidence_breakdown.uncertainty_factors])
        target_disagreement_rate = target_disagreements / len(aggregated_results)
        
        if avg_confidence < 0.7:
            recommendations.append("Consider improving rule-based patterns for better initial confidence")
        
        if avg_agreement < 0.6:
            recommendations.append("Engines show significant disagreement - review field definitions")
        
        if target_disagreement_rate > 0.2:
            recommendations.append("High target disagreement rate - consider improving semantic understanding")
        
        # Analyze uncertainty
        high_uncertainty_count = len([r for r in aggregated_results if len(r.confidence_breakdown.uncertainty_factors) >= 2])
        if high_uncertainty_count > len(aggregated_results) * 0.2:
            recommendations.append("High uncertainty detected - consider domain-specific training")
        
        # Analyze field categories
        category_analysis = self._analyze_field_categories(aggregated_results)
        for category, stats in category_analysis.items():
            if stats["average_confidence"] < 0.6:
                recommendations.append(f"Low confidence in {category} fields - review mapping strategies")
            if stats["target_disagreements"] > stats["count"] * 0.3:
                recommendations.append(f"High target disagreements in {category} fields - improve semantic understanding")
        
        return recommendations 