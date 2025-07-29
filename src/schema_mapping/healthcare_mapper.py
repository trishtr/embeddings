import yaml
from typing import Dict, List, Tuple, Any
import re
from .schema_mapper import SchemaMapper
from ..embeddings.embedding_handler import EmbeddingHandler

class HealthcareSchemaMapper(SchemaMapper):
    """Healthcare-specific schema mapper that incorporates domain rules and context"""
    
    def __init__(self, embedding_handler: EmbeddingHandler, rules_file: str = 'config/healthcare_rules.yaml'):
        super().__init__(embedding_handler)
        self.rules = self._load_rules(rules_file)
        self.phi_fields = set(self._get_phi_fields())
        self.domain_contexts = self.rules.get('domain_contexts', {})
    
    def _load_rules(self, rules_file: str) -> Dict:
        """Load healthcare domain rules from configuration"""
        with open(rules_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_phi_fields(self) -> List[str]:
        """Get list of PHI fields from rules"""
        phi_rules = self.rules.get('privacy_rules', {}).get('phi_fields', [])
        fields = []
        for rule in phi_rules:
            if isinstance(rule, dict) and 'fields' in rule:
                fields.extend(rule['fields'])
        return fields
    
    def _get_domain_context(self, field_name: str) -> str:
        """Determine the domain context for a field"""
        for context, details in self.domain_contexts.items():
            key_terms = details.get('key_terms', [])
            if any(term in field_name.lower() for term in key_terms):
                return context
        return 'general'
    
    def _validate_field_type(self, field_name: str, field_value: Any, field_type: str) -> bool:
        """Validate field value against type rules"""
        type_rules = self.rules.get('field_type_rules', {})
        
        # Check identifier formats
        if 'identifiers' in type_rules:
            for id_type, rules in type_rules['identifiers'].items():
                if id_type in field_name.lower():
                    pattern = rules.get('format')
                    if pattern and not re.match(pattern, str(field_value)):
                        return False
        
        # Check date formats
        if 'dates' in type_rules and field_type.upper() == 'DATE':
            for date_type, rules in type_rules['dates'].items():
                if date_type in field_name.lower():
                    # Add date validation logic here
                    return True
        
        # Check code formats
        if 'codes' in type_rules:
            for code_type, rules in type_rules['codes'].items():
                if code_type in field_name.lower():
                    # Add code validation logic here
                    return True
        
        return True
    
    def _apply_mapping_rules(self, source_field: str, target_field: str, context: str) -> float:
        """Apply domain-specific mapping rules and adjust confidence"""
        base_confidence = super()._calculate_confidence(source_field, target_field)
        
        # Get context-specific rules
        context_rules = self.rules.get('mapping_rules', {}).get(f'{context}_rules', [])
        
        # Apply rules and adjust confidence
        for rule in context_rules:
            fields = rule.get('fields', [])
            if source_field in fields or target_field in fields:
                priority = rule.get('priority', 'medium')
                # Adjust confidence based on priority
                if priority == 'critical':
                    base_confidence *= 1.2
                elif priority == 'high':
                    base_confidence *= 1.1
                
                # Cap confidence at 1.0
                base_confidence = min(base_confidence, 1.0)
        
        return base_confidence
    
    def _enhance_with_context(self, source_field: str, target_field: str, context: str) -> Dict:
        """Enhance mapping with domain context"""
        enhancement_rules = self.rules.get('context_enhancement', {}).get(f'{context}_context', [])
        enhancements = {}
        
        for rule in enhancement_rules:
            fields = rule.get('fields', [])
            if source_field in fields or target_field in fields:
                enhancements['context_source'] = rule.get('context_source')
                enhancements['rule_description'] = rule.get('rule')
        
        return enhancements
    
    def find_field_mappings(self, source_schema: Dict[str, str], target_schema: Dict[str, str]) -> List[Tuple[str, str, float]]:
        """Override to incorporate healthcare-specific rules and context"""
        mappings = []
        
        for source_field, source_type in source_schema.items():
            # Determine domain context
            context = self._get_domain_context(source_field)
            
            # Get base mappings from parent class
            base_mappings = super().find_field_mappings(
                {source_field: source_type},
                target_schema
            )
            
            # Apply healthcare rules and context
            for source, target, confidence in base_mappings:
                # Validate field types
                if not self._validate_field_type(source, None, source_type):
                    continue
                
                # Apply domain-specific rules
                adjusted_confidence = self._apply_mapping_rules(source, target, context)
                
                # Add to mappings if confidence is high enough
                if adjusted_confidence >= self.confidence_threshold:
                    mappings.append((source, target, adjusted_confidence))
        
        return mappings
    
    def transform_data(self, source_data: List[Dict], mapping: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Override to handle PHI fields and apply healthcare transformations"""
        transformed_data = []
        
        for record in source_data:
            transformed_record = {}
            
            for source_field, mapping_info in mapping.items():
                target_field = mapping_info['target_field']
                value = record.get(source_field)
                
                # Handle PHI fields
                if source_field in self.phi_fields or target_field in self.phi_fields:
                    # Add PHI handling logic here
                    pass
                
                # Apply field type validations
                if value is not None:
                    source_type = mapping_info.get('source_type', 'VARCHAR')
                    if self._validate_field_type(source_field, value, source_type):
                        transformed_record[target_field] = value
                
            transformed_data.append(transformed_record)
        
        return transformed_data
    
    def validate_mapping(self, mapping: Dict[str, Dict[str, Any]], threshold: float = None) -> bool:
        """Override to include healthcare-specific validation"""
        # First check basic validation from parent class
        if not super().validate_mapping(mapping, threshold):
            return False
        
        # Additional healthcare-specific validation
        for source_field, mapping_info in mapping.items():
            target_field = mapping_info['target_field']
            
            # Check PHI field mapping
            if source_field in self.phi_fields and target_field not in self.phi_fields:
                return False  # PHI fields must map to PHI fields
            
            # Check domain context compatibility
            source_context = self._get_domain_context(source_field)
            target_context = self._get_domain_context(target_field)
            if source_context != 'general' and target_context != 'general' and source_context != target_context:
                return False  # Incompatible contexts
        
        return True 