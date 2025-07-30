# LLM Fallback Logic Documentation

## Overview

LLM fallback logic is a sophisticated decision-making system that determines when to use Large Language Models for schema mapping when traditional rule-based and embedding approaches are insufficient. This system optimizes resource usage while ensuring high-quality mappings for complex cases.

## 🎯 Fallback Decision Framework

### Decision Matrix

The fallback logic uses a multi-criteria decision matrix to determine when LLM processing is necessary:

| Criteria            | Weight | Threshold | Impact |
| ------------------- | ------ | --------- | ------ |
| Low Confidence      | 0.30   | <0.6      | High   |
| Engine Disagreement | 0.25   | <0.7      | High   |
| Semantic Ambiguity  | 0.20   | >0.5      | Medium |
| Business Critical   | 0.15   | True      | High   |
| PHI Related         | 0.10   | True      | Medium |

### Fallback Score Calculation

```python
Fallback_Score = Σ(Criteria_Weight × Criteria_Score)
```

**Example**:

```python
# Low confidence (0.4 < 0.6 threshold)
confidence_score = 0.3 * 1.0 = 0.30

# Engine disagreement (0.5 < 0.7 threshold)
disagreement_score = 0.25 * 1.0 = 0.25

# Semantic ambiguity (0.6 > 0.5 threshold)
ambiguity_score = 0.20 * 1.0 = 0.20

# Business critical field
critical_score = 0.15 * 1.0 = 0.15

# PHI related field
phi_score = 0.10 * 1.0 = 0.10

Total_Fallback_Score = 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.00
```

## 🏗️ Fallback Priority Levels

### 1. Mandatory Fallback (Score ≥ 0.7)

**Characteristics**:

- **Processing**: Immediate LLM processing
- **Priority**: Highest
- **Resource Allocation**: Dedicated processing

**Triggers**:

- Aggregated confidence < 0.4
- Business critical field with low agreement
- PHI-related field with confidence < 0.7
- Semantic ambiguity > 0.8

**Examples**:

```python
mandatory_cases = [
    {
        "field": "provider_npi",
        "confidence": 0.35,
        "business_critical": True,
        "fallback_score": 0.85,
        "reason": "Critical field with low confidence"
    },
    {
        "field": "patient_ssn",
        "confidence": 0.45,
        "phi_related": True,
        "fallback_score": 0.78,
        "reason": "PHI field requiring high accuracy"
    }
]
```

### 2. Conditional Fallback (0.5 ≤ Score < 0.7)

**Characteristics**:

- **Processing**: LLM processing with medium priority
- **Priority**: Medium
- **Resource Allocation**: Shared processing

**Triggers**:

- Aggregated confidence < 0.6
- Engine disagreement with medium confidence
- Complex field characteristics
- Domain-specific ambiguity

**Examples**:

```python
conditional_cases = [
    {
        "field": "specialty_code",
        "confidence": 0.55,
        "semantic_ambiguity": 0.6,
        "fallback_score": 0.65,
        "reason": "Domain-specific field with ambiguity"
    },
    {
        "field": "malpractice_history",
        "confidence": 0.58,
        "field_complexity": 0.7,
        "fallback_score": 0.62,
        "reason": "Complex field requiring semantic understanding"
    }
]
```

### 3. Optional Fallback (0.3 ≤ Score < 0.5)

**Characteristics**:

- **Processing**: Batch LLM processing
- **Priority**: Low
- **Resource Allocation**: Background processing

**Triggers**:

- Aggregated confidence < 0.8
- Minor uncertainties
- Non-critical fields

**Examples**:

```python
optional_cases = [
    {
        "field": "website_url",
        "confidence": 0.72,
        "fallback_score": 0.42,
        "reason": "Standard field with minor uncertainty"
    },
    {
        "field": "fax_number",
        "confidence": 0.68,
        "fallback_score": 0.38,
        "reason": "Non-critical field with low confidence"
    }
]
```

### 4. No Fallback (Score < 0.3)

**Characteristics**:

- **Processing**: No LLM processing
- **Priority**: None
- **Resource Allocation**: None

**Triggers**:

- High confidence mappings
- Clear, unambiguous fields
- Standard field patterns

## 🔧 Implementation Strategies

### 1. Rule-Based Fallback Logic

```python
def determine_llm_fallback_rule_based(candidate):
    fallback_score = 0.0
    reasons = []

    # Criterion 1: Low aggregated confidence
    if candidate["confidence_breakdown"].aggregated_confidence < 0.6:
        fallback_score += 0.30
        reasons.append("low_confidence")

    # Criterion 2: High disagreement between engines
    if candidate["confidence_breakdown"].agreement_level < 0.7:
        fallback_score += 0.25
        reasons.append("engine_disagreement")

    # Criterion 3: Complex field characteristics
    if candidate["field_context"].semantic_ambiguity > 0.5:
        fallback_score += 0.20
        reasons.append("semantic_ambiguity")

    # Criterion 4: Business critical fields
    if candidate["field_context"].is_business_critical:
        fallback_score += 0.15
        reasons.append("business_critical")

    # Criterion 5: PHI-related fields
    if candidate["field_context"].is_phi_related:
        fallback_score += 0.10
        reasons.append("phi_related")

    return {
        "use_llm": fallback_score >= 0.5,
        "fallback_score": fallback_score,
        "reasons": reasons,
        "priority": get_priority_level(fallback_score)
    }
```

### 2. Advanced Fallback Rules

```python
def advanced_fallback_rules(candidate):
    rules = {
        "mandatory_llm": [
            # Always use LLM for these scenarios
            lambda: candidate["confidence_breakdown"].aggregated_confidence < 0.4,
            lambda: (candidate["field_context"].is_business_critical and
                    candidate["confidence_breakdown"].agreement_level < 0.6),
            lambda: (candidate["field_context"].is_phi_related and
                    candidate["confidence_breakdown"].aggregated_confidence < 0.7),
            lambda: candidate["field_context"].semantic_ambiguity > 0.8
        ],

        "conditional_llm": [
            # Use LLM under specific conditions
            lambda: (candidate["confidence_breakdown"].aggregated_confidence < 0.6 and
                    candidate["knn_insights"].similarity_distribution["std"] > 0.3),
            lambda: (candidate["confidence_breakdown"].agreement_level < 0.7 and
                    candidate["field_context"].field_category == "clinical"),
            lambda: (candidate["confidence_breakdown"].aggregated_confidence < 0.7 and
                    candidate["field_context"].type_compatibility < 0.5)
        ],

        "optional_llm": [
            # Consider LLM for these scenarios
            lambda: candidate["confidence_breakdown"].aggregated_confidence < 0.8,
            lambda: candidate["knn_insights"].nearest_neighbors[0]["similarity"] < 0.6,
            lambda: candidate["field_context"].pattern_similarity < 0.4
        ]
    }

    # Check rules in order of priority
    for rule_type, rule_list in rules.items():
        for rule in rule_list:
            if rule():
                return {
                    "use_llm": True,
                    "rule_type": rule_type,
                    "priority": get_priority_from_rule_type(rule_type)
                }

    return {"use_llm": False, "rule_type": "none", "priority": "none"}
```

### 3. Context-Aware Fallback

```python
def context_aware_fallback(candidate, domain_context):
    base_fallback_score = calculate_base_fallback_score(candidate)

    # Adjust based on domain context
    if domain_context["healthcare_domain"]:
        # Healthcare-specific adjustments
        if candidate["field_context"].is_phi_related:
            base_fallback_score *= 1.2  # 20% boost for PHI fields

        if candidate["field_context"].field_category == "clinical":
            base_fallback_score *= 1.1  # 10% boost for clinical fields

    # Adjust based on data volume
    if domain_context["large_dataset"]:
        # Reduce fallback for large datasets to manage costs
        base_fallback_score *= 0.9

    # Adjust based on time constraints
    if domain_context["time_critical"]:
        # Increase fallback for time-critical scenarios
        base_fallback_score *= 1.15

    return base_fallback_score
```

## 📊 Fallback Decision Examples

### Example 1: High-Priority Fallback

```python
candidate = {
    "source_field": "provider_npi",
    "confidence_breakdown": {
        "aggregated_confidence": 0.35,
        "agreement_level": 0.45
    },
    "field_context": {
        "is_business_critical": True,
        "is_phi_related": False,
        "semantic_ambiguity": 0.3
    }
}

fallback_decision = {
    "use_llm": True,
    "fallback_score": 0.85,
    "priority": "mandatory",
    "reasons": ["low_confidence", "engine_disagreement", "business_critical"],
    "processing_strategy": "immediate_llm"
}
```

### Example 2: Medium-Priority Fallback

```python
candidate = {
    "source_field": "specialty_code",
    "confidence_breakdown": {
        "aggregated_confidence": 0.55,
        "agreement_level": 0.65
    },
    "field_context": {
        "is_business_critical": False,
        "is_phi_related": False,
        "semantic_ambiguity": 0.6
    }
}

fallback_decision = {
    "use_llm": True,
    "fallback_score": 0.65,
    "priority": "conditional",
    "reasons": ["low_confidence", "semantic_ambiguity"],
    "processing_strategy": "batch_llm"
}
```

### Example 3: No Fallback

```python
candidate = {
    "source_field": "email_address",
    "confidence_breakdown": {
        "aggregated_confidence": 0.92,
        "agreement_level": 0.95
    },
    "field_context": {
        "is_business_critical": False,
        "is_phi_related": False,
        "semantic_ambiguity": 0.1
    }
}

fallback_decision = {
    "use_llm": False,
    "fallback_score": 0.15,
    "priority": "none",
    "reasons": [],
    "processing_strategy": "accept_mapping"
}
```

## 🚀 Processing Strategies

### 1. Immediate LLM Processing

**Use Case**: Mandatory fallback cases
**Characteristics**:

- Highest priority
- Dedicated resources
- Real-time processing
- Immediate response

**Implementation**:

```python
async def process_immediate_llm(candidate):
    logger.info(f"🧠 Processing mandatory LLM candidate: {candidate['source_field']}")

    # Generate rich context
    rich_context = generate_rich_context(candidate)

    # Process with LLM
    llm_result = await llm_processor.process(rich_context)

    # Update candidate with LLM result
    candidate["llm_result"] = llm_result
    candidate["processing_strategy"] = "immediate_llm"

    return candidate
```

### 2. Batch LLM Processing

**Use Case**: Conditional and optional fallback cases
**Characteristics**:

- Medium priority
- Shared resources
- Batch processing
- Cost optimization

**Implementation**:

```python
async def process_batch_llm(llm_queue):
    logger.info(f"🧠 Processing {len(llm_queue)} queued LLM candidates")

    # Group by priority
    mandatory_batch = [c for c in llm_queue if c["llm_fallback"]["priority"] == "mandatory"]
    conditional_batch = [c for c in llm_queue if c["llm_fallback"]["priority"] == "conditional"]
    optional_batch = [c for c in llm_queue if c["llm_fallback"]["priority"] == "optional"]

    # Process in priority order
    results = []
    results.extend(await process_batch(mandatory_batch, "high_priority"))
    results.extend(await process_batch(conditional_batch, "medium_priority"))
    results.extend(await process_batch(optional_batch, "low_priority"))

    return results
```

### 3. Adaptive Processing

**Use Case**: Dynamic resource allocation
**Characteristics**:

- Resource-aware
- Performance monitoring
- Dynamic adjustment
- Cost optimization

**Implementation**:

```python
async def adaptive_llm_processing(candidates, resource_constraints):
    # Monitor resource usage
    current_load = get_current_resource_load()
    available_budget = resource_constraints["llm_budget"]

    # Adjust processing strategy based on constraints
    if current_load > 0.8 or available_budget < 100:
        # Conservative processing
        return await conservative_llm_processing(candidates)
    else:
        # Aggressive processing
        return await aggressive_llm_processing(candidates)
```

## 📈 Performance Monitoring

### Key Metrics

1. **Fallback Rate**: Percentage of fields requiring LLM processing
2. **Processing Time**: Average time for LLM processing
3. **Cost per Field**: Average cost of LLM processing per field
4. **Quality Improvement**: Improvement in mapping quality after LLM processing
5. **Resource Utilization**: Efficiency of resource usage

### Monitoring Dashboard

```python
fallback_metrics = {
    "total_fields": 1000,
    "fallback_rate": 0.15,  # 15% of fields require LLM
    "processing_time": {
        "immediate": 2.5,    # seconds
        "batch": 15.0,       # seconds
        "average": 8.2       # seconds
    },
    "cost_per_field": 0.05,  # dollars
    "quality_improvement": 0.12,  # 12% improvement
    "resource_utilization": 0.75  # 75% efficiency
}
```

## 🔍 Troubleshooting

### Common Issues

1. **High Fallback Rate**

   - **Cause**: Low-quality initial mappings
   - **Solution**: Improve rule-based and embedding engines

2. **Slow Processing**

   - **Cause**: Inefficient batching or resource constraints
   - **Solution**: Optimize batch sizes and resource allocation

3. **High Costs**

   - **Cause**: Excessive LLM usage
   - **Solution**: Adjust fallback thresholds and optimize processing

4. **Low Quality Improvement**
   - **Cause**: Poor LLM prompts or context
   - **Solution**: Improve context generation and prompt engineering

### Debugging Tools

```python
def debug_fallback_decision(candidate):
    print(f"Field: {candidate['source_field']}")
    print(f"Confidence: {candidate['confidence_breakdown'].aggregated_confidence}")
    print(f"Agreement: {candidate['confidence_breakdown'].agreement_level}")
    print(f"Ambiguity: {candidate['field_context'].semantic_ambiguity}")
    print(f"Business Critical: {candidate['field_context'].is_business_critical}")
    print(f"PHI Related: {candidate['field_context'].is_phi_related}")

    fallback_decision = determine_llm_fallback(candidate)
    print(f"Fallback Decision: {fallback_decision}")
```

## 📚 Best Practices

### 1. Threshold Tuning

- **Monitor performance** by fallback category
- **Adjust thresholds** based on domain requirements
- **Balance quality and cost** appropriately

### 2. Resource Management

- **Implement queuing** for batch processing
- **Monitor resource usage** continuously
- **Scale processing** based on demand

### 3. Quality Assurance

- **Validate LLM results** against ground truth
- **Implement feedback loops** for continuous improvement
- **Track quality metrics** over time

### 4. Cost Optimization

- **Use appropriate batching** strategies
- **Implement caching** for similar cases
- **Monitor and optimize** LLM usage patterns

This comprehensive fallback logic system ensures optimal resource usage while maintaining high mapping quality for complex schema mapping scenarios.
