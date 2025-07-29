# Sample Context-Enhanced Mapping Output

## 1. Pre-Mapping Context Analysis

### Source Schema Context
```json
{
  "schema_name": "legacy_provider_system",
  "context_analysis": {
    "domain_type": "provider",
    "confidence": 0.95,
    "key_indicators": [
      "provider_npi",
      "doctor_name",
      "specialty_code"
    ],
    "context_rules_applied": [
      "Provider identifiers must follow NPI format",
      "Provider names must preserve credentials",
      "Specialties must use standard taxonomy"
    ]
  },
  "field_contexts": {
    "provider_npi": {
      "context": "identifier",
      "domain": "provider",
      "sensitivity": "phi",
      "format_rule": "^\\d{10}$",
      "validation_status": "valid",
      "business_rules": [
        "Must be valid NPI number",
        "Must be active in NPPES registry"
      ]
    },
    "doctor_name": {
      "context": "name",
      "domain": "provider",
      "sensitivity": "phi",
      "components": ["first_name", "middle_name", "last_name", "credentials"],
      "business_rules": [
        "Preserve name components",
        "Maintain credentials order",
        "Handle multiple credentials"
      ]
    },
    "specialty_code": {
      "context": "taxonomy",
      "domain": "provider",
      "code_system": "healthcare_provider_taxonomy",
      "validation_rules": [
        "Must be valid taxonomy code",
        "Must match provider type",
        "Must be currently active"
      ]
    }
  }
}
```

### Target Schema Context
```json
{
  "schema_name": "unified_healthcare_system",
  "context_analysis": {
    "domain_type": "provider",
    "confidence": 0.98,
    "key_indicators": [
      "healthcare_provider_id",
      "provider_full_name",
      "provider_taxonomy"
    ],
    "context_rules_applied": [
      "Standard healthcare identifiers",
      "Structured name components",
      "Taxonomy code validation"
    ]
  },
  "field_contexts": {
    "healthcare_provider_id": {
      "context": "identifier",
      "domain": "provider",
      "sensitivity": "phi",
      "format_rule": "^(NPI|PROV)\\d{10}$",
      "validation_status": "valid",
      "business_rules": [
        "Support both NPI and internal IDs",
        "Maintain cross-reference mapping"
      ]
    },
    "provider_full_name": {
      "context": "name",
      "domain": "provider",
      "sensitivity": "phi",
      "structure": {
        "format": "structured",
        "components": [
          "title",
          "first_name",
          "middle_name",
          "last_name",
          "suffix",
          "credentials"
        ]
      }
    },
    "provider_taxonomy": {
      "context": "taxonomy",
      "domain": "provider",
      "code_system": "NUCC",
      "version": "current",
      "validation_rules": [
        "Must be current NUCC code",
        "Must match provider type",
        "Must be active code"
      ]
    }
  }
}
```

## 2. Context-Enhanced Mapping Results

### Field Mappings with Context
```json
{
  "mappings": [
    {
      "source_field": "provider_npi",
      "target_field": "healthcare_provider_id",
      "context_match": {
        "domain": "provider",
        "type": "identifier",
        "confidence": 0.95,
        "rules_satisfied": [
          "Format validation",
          "Domain consistency",
          "PHI handling"
        ],
        "transformation_required": {
          "type": "format_standardization",
          "rule": "Prefix NPI with 'NPI'"
        }
      }
    },
    {
      "source_field": "doctor_name",
      "target_field": "provider_full_name",
      "context_match": {
        "domain": "provider",
        "type": "name",
        "confidence": 0.92,
        "rules_satisfied": [
          "Component preservation",
          "Credential handling",
          "PHI compliance"
        ],
        "transformation_required": {
          "type": "name_parsing",
          "components": [
            "title",
            "first_name",
            "middle_name",
            "last_name",
            "suffix",
            "credentials"
          ]
        }
      }
    },
    {
      "source_field": "specialty_code",
      "target_field": "provider_taxonomy",
      "context_match": {
        "domain": "provider",
        "type": "taxonomy",
        "confidence": 0.88,
        "rules_satisfied": [
          "Code system validation",
          "Domain consistency",
          "Active status check"
        ],
        "transformation_required": {
          "type": "code_mapping",
          "source_system": "legacy_taxonomy",
          "target_system": "NUCC",
          "mapping_table": "taxonomy_crosswalk"
        }
      }
    }
  ]
}
```

## 3. Context Validation Results

### Domain-Level Validation
```json
{
  "domain_validation": {
    "status": "passed",
    "score": 0.94,
    "checks": [
      {
        "rule": "Provider domain consistency",
        "status": "passed",
        "details": "All fields maintain provider context"
      },
      {
        "rule": "PHI field handling",
        "status": "passed",
        "details": "All PHI fields properly identified and mapped"
      },
      {
        "rule": "Taxonomy consistency",
        "status": "passed",
        "details": "Specialty codes properly mapped to standard taxonomy"
      }
    ]
  }
}
```

### Field-Level Validation
```json
{
  "field_validation": {
    "provider_npi": {
      "status": "passed",
      "rules_checked": [
        {
          "rule": "NPI format",
          "status": "passed",
          "confidence": 1.0
        },
        {
          "rule": "Active status",
          "status": "passed",
          "confidence": 1.0
        }
      ]
    },
    "doctor_name": {
      "status": "passed",
      "rules_checked": [
        {
          "rule": "Name components",
          "status": "passed",
          "confidence": 0.95
        },
        {
          "rule": "Credentials preservation",
          "status": "passed",
          "confidence": 0.92
        }
      ]
    },
    "specialty_code": {
      "status": "passed",
      "rules_checked": [
        {
          "rule": "Code validity",
          "status": "passed",
          "confidence": 0.98
        },
        {
          "rule": "Taxonomy mapping",
          "status": "passed",
          "confidence": 0.88
        }
      ]
    }
  }
}
```

## 4. Transformation Instructions

### Field-Specific Transformations
```json
{
  "transformations": {
    "provider_npi": {
      "type": "identifier_standardization",
      "steps": [
        {
          "operation": "format_check",
          "rule": "^\\d{10}$"
        },
        {
          "operation": "prefix_add",
          "value": "NPI"
        },
        {
          "operation": "checksum_validate",
          "algorithm": "NPI_checksum"
        }
      ]
    },
    "doctor_name": {
      "type": "name_structuring",
      "steps": [
        {
          "operation": "parse_components",
          "pattern": "^([A-Z][a-z]+)\\s+(?:([A-Z])[.]\\s+)?([A-Z][a-z]+)(?:,\\s*([A-Z\\.\\s]+))?$"
        },
        {
          "operation": "standardize_credentials",
          "reference": "medical_credentials_list"
        },
        {
          "operation": "format_output",
          "template": "{title} {first} {middle} {last}, {credentials}"
        }
      ]
    },
    "specialty_code": {
      "type": "code_mapping",
      "steps": [
        {
          "operation": "validate_source_code",
          "code_system": "legacy_taxonomy"
        },
        {
          "operation": "map_to_nucc",
          "mapping_table": "taxonomy_crosswalk"
        },
        {
          "operation": "validate_target_code",
          "code_system": "NUCC"
        }
      ]
    }
  }
}
```

## 5. Quality Metrics

### Mapping Quality Assessment
```json
{
  "quality_metrics": {
    "overall_score": 0.92,
    "metrics": {
      "context_preservation": {
        "score": 0.95,
        "details": "Domain context maintained across all mappings"
      },
      "data_completeness": {
        "score": 0.94,
        "details": "All required fields mapped with transformations"
      },
      "rule_compliance": {
        "score": 0.96,
        "details": "Business rules satisfied for all mappings"
      },
      "transformation_accuracy": {
        "score": 0.89,
        "details": "Transformations preserve data integrity"
      }
    },
    "recommendations": [
      {
        "field": "specialty_code",
        "suggestion": "Consider adding specialty-specific validation rules",
        "priority": "medium"
      },
      {
        "field": "doctor_name",
        "suggestion": "Add support for international name formats",
        "priority": "low"
      }
    ]
  }
}
```

This output demonstrates:
1. Detailed context analysis of both source and target schemas
2. Context-enhanced field mappings with business rules
3. Comprehensive validation results
4. Specific transformation instructions
5. Quality metrics and recommendations

The format provides:
- Clear visibility into the mapping process
- Traceability of business rule application
- Validation of context preservation
- Detailed transformation instructions
- Quality assessment and improvement suggestions

Would you like me to explain any specific part of this output in more detail? 