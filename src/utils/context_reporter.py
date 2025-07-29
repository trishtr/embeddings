import json
from typing import Dict, List, Any
from datetime import datetime

class ContextReporter:
    """Utility class for generating context-enhanced mapping reports"""
    
    def __init__(self, healthcare_mapper):
        self.mapper = healthcare_mapper
        self.rules = healthcare_mapper.rules
    
    def generate_context_report(self, 
                              source_schema: Dict[str, str],
                              target_schema: Dict[str, str],
                              mappings: List[tuple],
                              validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive context-enhanced mapping report"""
        return {
            "report_metadata": self._generate_metadata(),
            "pre_mapping_analysis": self._analyze_schemas(source_schema, target_schema),
            "mapping_results": self._enhance_mappings(mappings),
            "validation_results": self._enhance_validation(validation_results),
            "transformation_instructions": self._generate_transformations(mappings),
            "quality_metrics": self._calculate_metrics(mappings, validation_results)
        }
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            "timestamp": datetime.now().isoformat(),
            "rules_version": self.rules.get("version", "1.0"),
            "domains_analyzed": list(self.rules.get("domain_contexts", {}).keys())
        }
    
    def _analyze_schemas(self, 
                        source_schema: Dict[str, str],
                        target_schema: Dict[str, str]) -> Dict[str, Any]:
        """Analyze schemas with context"""
        return {
            "source_schema_context": self._analyze_single_schema(source_schema, "source"),
            "target_schema_context": self._analyze_single_schema(target_schema, "target")
        }
    
    def _analyze_single_schema(self, 
                             schema: Dict[str, str],
                             schema_type: str) -> Dict[str, Any]:
        """Analyze a single schema with context"""
        field_contexts = {}
        domain_indicators = {
            "provider": 0,
            "patient": 0,
            "facility": 0
        }
        
        for field_name, field_type in schema.items():
            # Get domain context
            context = self.mapper._get_domain_context(field_name)
            if context in domain_indicators:
                domain_indicators[context] += 1
            
            # Analyze field context
            field_contexts[field_name] = {
                "context": context,
                "domain": context,
                "sensitivity": "phi" if field_name in self.mapper.phi_fields else "normal",
                "format_rule": self._get_format_rule(field_name, field_type),
                "business_rules": self._get_business_rules(field_name, context)
            }
        
        # Determine primary domain
        primary_domain = max(domain_indicators.items(), key=lambda x: x[1])[0]
        
        return {
            "schema_type": schema_type,
            "primary_domain": primary_domain,
            "confidence": self._calculate_domain_confidence(domain_indicators),
            "field_contexts": field_contexts
        }
    
    def _get_format_rule(self, field_name: str, field_type: str) -> str:
        """Get format rule for field"""
        type_rules = self.rules.get("field_type_rules", {})
        
        # Check identifiers
        for id_type, rules in type_rules.get("identifiers", {}).items():
            if id_type in field_name.lower():
                return rules.get("format", "")
        
        # Check dates
        if field_type.upper() == "DATE":
            for date_type, rules in type_rules.get("dates", {}).items():
                if date_type in field_name.lower():
                    return rules.get("format", "")
        
        return ""
    
    def _get_business_rules(self, field_name: str, context: str) -> List[str]:
        """Get business rules for field"""
        rules = []
        context_rules = self.rules.get("mapping_rules", {}).get(f"{context}_rules", [])
        
        for rule in context_rules:
            if field_name in rule.get("fields", []):
                rules.append(rule.get("rule", ""))
        
        return [r for r in rules if r]
    
    def _calculate_domain_confidence(self, indicators: Dict[str, int]) -> float:
        """Calculate confidence in domain determination"""
        total = sum(indicators.values())
        if total == 0:
            return 0.0
        
        max_count = max(indicators.values())
        return round(max_count / total, 2)
    
    def _enhance_mappings(self, mappings: List[tuple]) -> Dict[str, Any]:
        """Enhance mappings with context information"""
        enhanced_mappings = []
        
        for source_field, target_field, confidence in mappings:
            context = self.mapper._get_domain_context(source_field)
            enhanced_mappings.append({
                "source_field": source_field,
                "target_field": target_field,
                "context_match": {
                    "domain": context,
                    "confidence": confidence,
                    "rules_satisfied": self._get_satisfied_rules(source_field, target_field, context),
                    "transformation_required": self._get_transformation_requirements(
                        source_field, target_field, context
                    )
                }
            })
        
        return {"mappings": enhanced_mappings}
    
    def _get_satisfied_rules(self, 
                           source_field: str,
                           target_field: str,
                           context: str) -> List[str]:
        """Get list of satisfied business rules"""
        satisfied = []
        context_rules = self.rules.get("mapping_rules", {}).get(f"{context}_rules", [])
        
        for rule in context_rules:
            if source_field in rule.get("fields", []) or target_field in rule.get("fields", []):
                satisfied.append(rule.get("rule", ""))
        
        return [r for r in satisfied if r]
    
    def _get_transformation_requirements(self,
                                      source_field: str,
                                      target_field: str,
                                      context: str) -> Dict[str, Any]:
        """Get transformation requirements"""
        transform_rules = self.rules.get("transformation_rules", {})
        
        # Check name handling
        if "name" in source_field.lower() or "name" in target_field.lower():
            return transform_rules.get("name_handling", [])[0]
        
        # Check address handling
        if "address" in source_field.lower() or "address" in target_field.lower():
            return transform_rules.get("address_handling", [])[0]
        
        return {}
    
    def _enhance_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance validation results with context"""
        return {
            "domain_validation": self._validate_domain_rules(validation_results),
            "field_validation": self._validate_field_rules(validation_results)
        }
    
    def _validate_domain_rules(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate domain-level rules"""
        return {
            "status": validation_results.get("status", "unknown"),
            "score": validation_results.get("score", 0.0),
            "checks": self._get_domain_checks(validation_results)
        }
    
    def _validate_field_rules(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field-level rules"""
        field_validations = {}
        
        for field, results in validation_results.get("fields", {}).items():
            field_validations[field] = {
                "status": results.get("status", "unknown"),
                "rules_checked": self._get_field_checks(field, results)
            }
        
        return field_validations
    
    def _get_domain_checks(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get domain-level validation checks"""
        checks = []
        for check in validation_results.get("domain_checks", []):
            checks.append({
                "rule": check.get("rule", ""),
                "status": check.get("status", "unknown"),
                "details": check.get("details", "")
            })
        return checks
    
    def _get_field_checks(self, 
                         field: str,
                         results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get field-level validation checks"""
        checks = []
        for check in results.get("checks", []):
            checks.append({
                "rule": check.get("rule", ""),
                "status": check.get("status", "unknown"),
                "confidence": check.get("confidence", 0.0)
            })
        return checks
    
    def _generate_transformations(self, mappings: List[tuple]) -> Dict[str, Any]:
        """Generate transformation instructions"""
        transformations = {}
        
        for source_field, target_field, _ in mappings:
            transformations[source_field] = self._get_field_transformation(
                source_field, target_field
            )
        
        return {"transformations": transformations}
    
    def _get_field_transformation(self,
                                source_field: str,
                                target_field: str) -> Dict[str, Any]:
        """Get transformation instructions for a field"""
        transform_rules = self.rules.get("transformation_rules", {})
        
        # Handle identifiers
        if any(id_type in source_field.lower() for id_type in ["npi", "id", "identifier"]):
            return {
                "type": "identifier_standardization",
                "steps": [
                    {"operation": "format_check", "rule": self._get_format_rule(source_field, "")},
                    {"operation": "standardize", "reference": "identifier_standards"}
                ]
            }
        
        # Handle names
        if "name" in source_field.lower():
            return transform_rules.get("name_handling", [])[0]
        
        return {"type": "direct_copy", "steps": [{"operation": "copy", "validation": "none"}]}
    
    def _calculate_metrics(self,
                         mappings: List[tuple],
                         validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics"""
        return {
            "overall_score": self._calculate_overall_score(mappings, validation_results),
            "metrics": {
                "context_preservation": self._calculate_context_preservation(mappings),
                "data_completeness": self._calculate_completeness(validation_results),
                "rule_compliance": self._calculate_compliance(validation_results)
            },
            "recommendations": self._generate_recommendations(mappings, validation_results)
        }
    
    def _calculate_overall_score(self,
                               mappings: List[tuple],
                               validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = [
            self._calculate_context_preservation(mappings)["score"],
            self._calculate_completeness(validation_results)["score"],
            self._calculate_compliance(validation_results)["score"]
        ]
        return round(sum(scores) / len(scores), 2)
    
    def _calculate_context_preservation(self, mappings: List[tuple]) -> Dict[str, Any]:
        """Calculate context preservation score"""
        preserved = 0
        for source_field, target_field, _ in mappings:
            if self.mapper._get_domain_context(source_field) == self.mapper._get_domain_context(target_field):
                preserved += 1
        
        score = round(preserved / len(mappings), 2) if mappings else 0.0
        return {
            "score": score,
            "details": f"Context preserved in {preserved} of {len(mappings)} mappings"
        }
    
    def _calculate_completeness(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data completeness score"""
        total = len(validation_results.get("fields", {}))
        valid = sum(1 for f in validation_results.get("fields", {}).values() 
                   if f.get("status") == "passed")
        
        score = round(valid / total, 2) if total > 0 else 0.0
        return {
            "score": score,
            "details": f"Valid mappings for {valid} of {total} fields"
        }
    
    def _calculate_compliance(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate rule compliance score"""
        return {
            "score": validation_results.get("compliance_score", 0.0),
            "details": "Business rules compliance assessment"
        }
    
    def _generate_recommendations(self,
                                mappings: List[tuple],
                                validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Check for low confidence mappings
        for source_field, target_field, confidence in mappings:
            if confidence < 0.8:
                recommendations.append({
                    "field": source_field,
                    "suggestion": f"Review mapping to {target_field} (low confidence)",
                    "priority": "high"
                })
        
        # Check for failed validations
        for field, results in validation_results.get("fields", {}).items():
            if results.get("status") != "passed":
                recommendations.append({
                    "field": field,
                    "suggestion": "Add additional validation rules",
                    "priority": "medium"
                })
        
        return recommendations 