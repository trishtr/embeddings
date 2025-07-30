# Confidence Score Aggregation Documentation

## Overview

Confidence score aggregation is a critical component of the advanced mapping pipeline that combines results from multiple engines (rule-based and embedding) to produce a unified, reliable confidence assessment for each field mapping.

## 🔗 Aggregation Strategies

### 1. Weighted Average (Basic)

**Description**: Simple weighted combination of confidence scores from different engines.

**Formula**:

```
Aggregated_Score = (w₁ × Rule_Score) + (w₂ × Embedding_Score)
```

**Default Weights**:

- Rule-based engine: 0.4 (40%)
- Embedding engine: 0.6 (60%)

**Use Case**: Standard scenarios where both engines are equally reliable.

**Example**:

```python
rule_score = 0.75
embedding_score = 0.85
aggregated_score = (0.4 × 0.75) + (0.6 × 0.85) = 0.30 + 0.51 = 0.81
```

### 2. Context-Aware Aggregation - should be used in practice

**Description**: Adjusts weights based on field characteristics and domain context.

**Weight Adjustment Rules**:

#### Standard Fields (Higher rule-based weight) - applied

- **Fields**: email, phone, name, id, date, address, city, state, zip
- **Weights**: Rule-based: 0.6, Embedding: 0.4
- **Rationale**: Standard fields have well-defined patterns that rule-based engines excel at

#### Domain-Specific Fields (Higher embedding weight) - applied

- **Fields**: npi, specialty, license, malpractice, accreditation, phi
- **Weights**: Rule-based: 0.3, Embedding: 0.7
- **Rationale**: Domain-specific fields benefit from semantic understanding

#### Mixed Fields (Default weights)

- **Weights**: Rule-based: 0.4, Embedding: 0.6
- **Rationale**: Balanced approach for fields without clear categorization

**Example**:

# 1. Context-aware weights (primary)

if is_standard_field(field_name):
weights = {"rule_based": 0.6, "embedding": 0.4}
elif is_domain_specific(field_name):
weights = {"rule_based": 0.3, "embedding": 0.7}

```python
# Standard field
field_name = "email_address"
weights = {"rule_based": 0.6, "embedding": 0.4}
aggregated_score = (0.6 × 0.80) + (0.4 × 0.75) = 0.48 + 0.30 = 0.78

# Domain-specific field
field_name = "provider_npi"
weights = {"rule_based": 0.3, "embedding": 0.7}
aggregated_score = (0.3 × 0.70) + (0.7 × 0.90) = 0.21 + 0.63 = 0.84
```

### 3. Agreement-Based Adjustments - should be used in practice

**Description**: Applies bonuses or penalties based on how well the engines agree. answer the question : how confident should we be in this result. Purpose : quality validation, uncertainty detection, reliablity assessment

#### Agreement Bonuses

- **High Agreement** (>0.8): +0.1 bonus
- **Medium Agreement** (0.6-0.8): +0.05 bonus
- **Low Agreement** (<0.4): -0.15 penalty
- **Very Low Agreement** (0.4-0.6): -0.08 penalty

**Formula**:

```
Agreement_Level = 1.0 - |Rule_Score - Embedding_Score|
```

**Example**:

# 2. Agreement adjustments (secondary)

if agreement_level > 0.8:
final_score = min(1.0, aggregated_score + 0.1)
elif agreement_level < 0.4:
final_score = max(0.0, aggregated_score - 0.15)

```python
rule_score = 0.85
embedding_score = 0.87
agreement_level = 1.0 - |0.85 - 0.87| = 1.0 - 0.02 = 0.98

# High agreement bonus
base_score = 0.86
adjusted_score = min(1.0, base_score + 0.1) = 0.96
```

#### Disagreement Penalties

```python
rule_score = 0.30
embedding_score = 0.80
agreement_level = 1.0 - |0.30 - 0.80| = 1.0 - 0.50 = 0.50

# Medium disagreement penalty
base_score = 0.60
adjusted_score = max(0.0, base_score - 0.08) = 0.52
```

### 4. Uncertainty Factor Analysis

**Description**: Identifies and accounts for factors that contribute to uncertainty.

#### Uncertainty Factors

1. **Engine Disagreement**

   - Trigger: Agreement level < 0.6
   - Impact: Reduces confidence

2. **Low Rule Confidence**

   - Trigger: Rule-based score < 0.5
   - Impact: Indicates pattern matching issues

3. **Low Embedding Confidence**

   - Trigger: Embedding score < 0.5
   - Impact: Indicates semantic understanding issues

4. **High Score Variance**
   - Trigger: |Rule_Score - Embedding_Score| > 0.3
   - Impact: Indicates conflicting signals

**Example**:

```python
uncertainty_factors = []
if agreement_level < 0.6:
    uncertainty_factors.append("engine_disagreement")
if rule_score < 0.5:
    uncertainty_factors.append("low_rule_confidence")
if embedding_score < 0.5:
    uncertainty_factors.append("low_embedding_confidence")
if abs(rule_score - embedding_score) > 0.3:
    uncertainty_factors.append("high_score_variance")
```

## 📊 Confidence Tiers

### High Confidence (0.8+)

- **Characteristics**: Clear, unambiguous mappings
- **Processing**: Accept immediately
- **Examples**:
  - `provider_npi` → `npi_number` (0.95)
  - `email_address` → `email` (0.92)
  - `phone_number` → `contact_phone` (0.89)

### Medium Confidence (0.6-0.8)

- **Characteristics**: Good mappings with minor uncertainties
- **Processing**: Embedding verification recommended
- **Examples**:
  - `specialty_code` → `medical_specialty` (0.75)
  - `practice_name` → `facility_name` (0.72)
  - `license_number` → `license_id` (0.68)

### Low Confidence (<0.6)

- **Characteristics**: Uncertain mappings requiring analysis
- **Processing**: Full embedding analysis or LLM processing
- **Examples**:
  - `tax_id` → `employer_id` (0.45)
  - `audit_history` → `audit_records` (0.38)
  - `malpractice_history` → `malpractice_record` (0.32)

## 🔧 Implementation Details

### Aggregation Pipeline

```python
def aggregate_confidence_scores(rule_score, embedding_score, field_name):
    # 1. Determine context-aware weights
    weights = get_context_aware_weights(field_name)

    # 2. Calculate weighted average
    base_score = (weights["rule_based"] * rule_score +
                  weights["embedding"] * embedding_score)

    # 3. Calculate agreement level
    agreement_level = 1.0 - abs(rule_score - embedding_score)

    # 4. Apply agreement adjustments
    adjusted_score = apply_agreement_adjustments(base_score, agreement_level)

    # 5. Identify uncertainty factors
    uncertainty_factors = identify_uncertainty_factors(
        rule_score, embedding_score, agreement_level
    )

    return {
        "aggregated_score": adjusted_score,
        "agreement_level": agreement_level,
        "uncertainty_factors": uncertainty_factors,
        "weights_used": weights
    }
```

### Context-Aware Weight Selection

```python
def get_context_aware_weights(field_name):
    field_lower = field_name.lower()

    # Standard fields
    if any(word in field_lower for word in ['email', 'phone', 'name', 'id', 'date']):
        return {"rule_based": 0.6, "embedding": 0.4}

    # Domain-specific fields
    elif any(word in field_lower for word in ['npi', 'specialty', 'license', 'malpractice']):
        return {"rule_based": 0.3, "embedding": 0.7}

    # Default weights
    else:
        return {"rule_based": 0.4, "embedding": 0.6}
```

### Agreement Adjustment Logic

```python
def apply_agreement_adjustments(base_score, agreement_level):
    adjusted_score = base_score

    # Agreement bonuses
    if agreement_level > 0.8:
        adjusted_score = min(1.0, adjusted_score + 0.1)
    elif agreement_level > 0.6:
        adjusted_score = min(1.0, adjusted_score + 0.05)

    # Disagreement penalties
    elif agreement_level < 0.4:
        adjusted_score = max(0.0, adjusted_score - 0.15)
    elif agreement_level < 0.6:
        adjusted_score = max(0.0, adjusted_score - 0.08)

    return adjusted_score
```

## 📈 Performance Metrics

### Aggregation Quality Metrics

1. **Agreement Rate**: Percentage of fields where engines agree
2. **Confidence Distribution**: Distribution of final confidence scores
3. **Uncertainty Rate**: Percentage of fields with uncertainty factors
4. **Tier Distribution**: Distribution across confidence tiers

### Example Metrics

```python
aggregation_metrics = {
    "total_fields": 100,
    "agreement_rate": 0.75,  # 75% of fields have good agreement
    "confidence_distribution": {
        "high": 0.40,    # 40% high confidence
        "medium": 0.35,  # 35% medium confidence
        "low": 0.25      # 25% low confidence
    },
    "uncertainty_rate": 0.30,  # 30% have uncertainty factors
    "average_confidence": 0.72
}
```

## 🎯 Best Practices

### 1. Weight Tuning

- **Monitor performance** by field category
- **Adjust weights** based on domain characteristics
- **Validate changes** with test datasets

### 2. Agreement Thresholds

- **Set appropriate thresholds** for bonuses/penalties
- **Consider domain-specific patterns**
- **Balance between precision and recall**

### 3. Uncertainty Handling

- **Track uncertainty factors** for analysis
- **Use uncertainty information** for downstream processing
- **Implement fallback strategies** for high-uncertainty cases

### 4. Performance Optimization

- **Cache weight calculations** for repeated fields
- **Batch process** similar field types
- **Parallelize** aggregation for large datasets

## 🔍 Troubleshooting

### Common Issues

1. **Low Agreement Rates**

   - **Cause**: Engines have different strengths
   - **Solution**: Adjust context-aware weights

2. **High Uncertainty**

   - **Cause**: Complex or ambiguous fields
   - **Solution**: Implement additional analysis steps

3. **Confidence Inflation**

   - **Cause**: Overly generous bonuses
   - **Solution**: Review agreement thresholds

4. **Performance Issues**
   - **Cause**: Inefficient weight calculation
   - **Solution**: Implement caching and batching

### Debugging Tools

```python
def debug_aggregation(rule_score, embedding_score, field_name):
    weights = get_context_aware_weights(field_name)
    agreement_level = 1.0 - abs(rule_score - embedding_score)

    print(f"Field: {field_name}")
    print(f"Rule Score: {rule_score}")
    print(f"Embedding Score: {embedding_score}")
    print(f"Weights: {weights}")
    print(f"Agreement Level: {agreement_level}")
    print(f"Uncertainty Factors: {identify_uncertainty_factors(rule_score, embedding_score, agreement_level)}")
```

## 📚 References

- **Multi-Engine Fusion**: Techniques for combining multiple ML models
- **Confidence Calibration**: Methods for reliable confidence estimation
- **Uncertainty Quantification**: Approaches for measuring prediction uncertainty
- **Domain Adaptation**: Strategies for adapting to specific domains
