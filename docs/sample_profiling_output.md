# Sample Data Profiling Output

## Pre-Mapping Analysis Output

### 1. Source Schema Profile

```json
{
  "source_name": "healthcare_provider_db",
  "table_name": "providers",
  "schema_analysis": {
    "field_count": 8,
    "data_types": {
      "VARCHAR": 5,
      "INTEGER": 2,
      "DATE": 1
    },
    "fields": {
      "provider_id": {
        "type": "VARCHAR(50)",
        "nullable": false,
        "is_primary": true,
        "pattern": "^PRV[0-9]{6}$",
        "sample_values": ["PRV123456", "PRV789012"]
      },
      "doctor_name": {
        "type": "VARCHAR(100)",
        "nullable": false,
        "pattern": "^[A-Z][a-z]+ [A-Z][a-z]+$",
        "sample_values": ["John Smith", "Mary Johnson"]
      },
      "specialization": {
        "type": "VARCHAR(50)",
        "nullable": true,
        "value_distribution": {
          "Cardiology": 25,
          "Pediatrics": 20,
          "Neurology": 15
        }
      },
      "license_number": {
        "type": "VARCHAR(20)",
        "nullable": false,
        "pattern": "^LIC[0-9]{8}$"
      },
      "years_experience": {
        "type": "INTEGER",
        "nullable": true,
        "statistics": {
          "min": 1,
          "max": 45,
          "mean": 15.7,
          "median": 14
        }
      }
    }
  },
  "data_quality": {
    "overall_score": 0.92,
    "completeness": {
      "score": 0.95,
      "details": {
        "provider_id": 1.0,
        "doctor_name": 1.0,
        "specialization": 0.92,
        "license_number": 1.0,
        "years_experience": 0.89
      }
    },
    "uniqueness": {
      "score": 0.98,
      "unique_fields": ["provider_id", "license_number"],
      "duplicate_count": 0
    },
    "consistency": {
      "score": 0.94,
      "pattern_match_rate": {
        "provider_id": 1.0,
        "license_number": 0.98
      }
    },
    "validity": {
      "score": 0.96,
      "invalid_records": {
        "years_experience": 2,
        "license_number": 1
      }
    }
  },
  "field_patterns": {
    "naming_conventions": {
      "snake_case": 5,
      "camel_case": 0
    },
    "identified_patterns": {
      "identifier_fields": ["provider_id", "license_number"],
      "name_fields": ["doctor_name"],
      "categorical_fields": ["specialization"],
      "numeric_fields": ["years_experience"]
    }
  }
}
```

### 2. Potential Mapping Analysis

```json
{
  "field_similarities": {
    "provider_id": {
      "healthcare_provider_id": 0.95,
      "doctor_id": 0.85,
      "provider_number": 0.82
    },
    "doctor_name": {
      "provider_name": 0.92,
      "full_name": 0.88,
      "physician_name": 0.87
    }
  },
  "type_compatibility": {
    "provider_id": {
      "healthcare_provider_id": {
        "compatible": true,
        "conversion_needed": false
      }
    },
    "years_experience": {
      "experience_years": {
        "compatible": true,
        "conversion_needed": false
      }
    }
  },
  "suggested_mappings": {
    "high_confidence": [
      {
        "source": "provider_id",
        "target": "healthcare_provider_id",
        "confidence": 0.95,
        "type_match": true
      },
      {
        "source": "doctor_name",
        "target": "provider_name",
        "confidence": 0.92,
        "type_match": true
      }
    ],
    "medium_confidence": [
      {
        "source": "specialization",
        "target": "specialty",
        "confidence": 0.78,
        "type_match": true
      }
    ]
  }
}
```

### 3. Mapping Readiness Assessment

```json
{
  "overall_score": 0.89,
  "metrics": {
    "schema_compatibility": 0.92,
    "data_quality": 0.94,
    "mapping_confidence": 0.85
  },
  "issues": [
    {
      "type": "data_quality",
      "field": "years_experience",
      "description": "2 invalid values found",
      "severity": "low"
    }
  ],
  "recommendations": [
    {
      "type": "data_cleanup",
      "description": "Clean invalid years_experience values",
      "priority": "medium"
    },
    {
      "type": "mapping_strategy",
      "description": "Consider compound mapping for address fields",
      "priority": "high"
    }
  ]
}
```

## Post-Mapping Analysis Output

### 1. Mapping Validation Results

```json
{
  "mapping_quality": {
    "overall_score": 0.91,
    "coverage": {
      "mapped_fields": 12,
      "total_fields": 14,
      "coverage_ratio": 0.857
    },
    "confidence_distribution": {
      "high": 8,
      "medium": 3,
      "low": 1
    }
  },
  "field_level_validation": {
    "provider_id": {
      "mapped_to": "healthcare_provider_id",
      "confidence": 0.95,
      "data_preservation": 1.0,
      "value_distribution_match": 0.98
    },
    "doctor_name": {
      "mapped_to": "provider_name",
      "confidence": 0.92,
      "data_preservation": 1.0,
      "value_distribution_match": 0.96
    }
  },
  "unmapped_fields": {
    "source": ["created_at"],
    "target": ["last_updated"]
  }
}
```

### 2. Data Compatibility Analysis

```json
{
  "type_compatibility": {
    "compatible_fields": 11,
    "incompatible_fields": 1,
    "details": {
      "provider_id": {
        "source_type": "VARCHAR(50)",
        "target_type": "VARCHAR(50)",
        "compatible": true
      }
    }
  },
  "value_ranges": {
    "years_experience": {
      "source_range": { "min": 1, "max": 45 },
      "target_range": { "min": 0, "max": 50 },
      "compatible": true
    }
  },
  "format_compatibility": {
    "provider_id": {
      "source_format": "^PRV[0-9]{6}$",
      "target_format": "^PRV[0-9]{6}$",
      "compatible": true
    }
  }
}
```

### 3. Transformation Impact Analysis

```json
{
  "data_quality_impact": {
    "overall_impact": "positive",
    "metrics": {
      "completeness": {
        "before": 0.92,
        "after": 0.94,
        "change": "+0.02"
      },
      "consistency": {
        "before": 0.94,
        "after": 0.96,
        "change": "+0.02"
      }
    }
  },
  "value_modifications": {
    "total_records": 1000,
    "modified_records": 15,
    "modification_types": {
      "format_standardization": 10,
      "data_cleaning": 5
    }
  },
  "performance_metrics": {
    "processing_time": "2.5s",
    "memory_usage": "256MB",
    "transformation_rate": "400 records/second"
  }
}
```

### 4. Summary Report

```json
{
  "mapping_summary": {
    "total_fields": 14,
    "mapped_fields": 12,
    "success_rate": 0.857,
    "average_confidence": 0.89
  },
  "quality_metrics": {
    "before_mapping": {
      "completeness": 0.92,
      "consistency": 0.94,
      "validity": 0.96
    },
    "after_mapping": {
      "completeness": 0.94,
      "consistency": 0.96,
      "validity": 0.97
    }
  },
  "recommendations": [
    {
      "type": "mapping_improvement",
      "field": "created_at",
      "suggestion": "Consider mapping to last_updated with transformation"
    },
    {
      "type": "data_quality",
      "field": "years_experience",
      "suggestion": "Implement validation rules for new records"
    }
  ],
  "next_steps": [
    "Review unmapped fields",
    "Implement suggested data quality improvements",
    "Monitor transformation performance"
  ]
}
```

## Visualization Examples

### 1. Field Similarity Matrix

```
                    healthcare_provider_id  doctor_id  provider_number
provider_id                          0.95      0.85             0.82
doctor_name                         0.25      0.82             0.30
specialization                      0.15      0.20             0.18
```

### 2. Data Quality Radar Chart

```
Completeness: 0.94 ★★★★★
Consistency:  0.96 ★★★★★
Validity:     0.97 ★★★★★
Coverage:     0.86 ★★★★☆
Confidence:   0.89 ★★★★☆
```

### 3. Transformation Impact

```
Before Mapping  │████████░░│ 0.92
After Mapping   │█████████░│ 0.94
                0.0    1.0

Modified Records: ▓▓▓░░░░░░░ 15%
Success Rate:     ████████░░ 86%
```

This sample output demonstrates:

- Comprehensive schema analysis
- Detailed data quality metrics
- Potential mapping suggestions
- Post-mapping validation
- Impact analysis
- Visual representations

The output helps in:

1. Understanding data quality
2. Identifying potential issues
3. Validating mappings
4. Tracking transformations
5. Making informed decisions
