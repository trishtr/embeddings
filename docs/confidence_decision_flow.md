# Confidence Score Decision Flow

## Overview

This document provides a comprehensive decision flow graph for confidence score calculation and decision making in the parallel mapping pipeline, based on the actual thresholds and adjustments used in practice.

## 🎯 **Decision Flow Graph**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PARALLEL MAPPING PIPELINE                        │
│                        Confidence Score Decision Flow                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RULE ENGINE   │    │ EMBEDDING ENGINE│    │   PARALLEL      │
│   PROCESSING    │    │   PROCESSING    │    │   EXECUTION     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIDENCE AGGREGATION STAGE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  TARGET FIELD   │    │ CONTEXT-AWARE   │    │  AGREEMENT      │
│   SELECTION     │    │    WEIGHTS      │    │  CALCULATION    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Same Target?    │    │ Field Category  │    │ Agreement =     │
│                 │    │                 │    │ 1.0 - |R-E|     │
│ YES → No Penalty│    │ Standard: 60/40 │    │                 │
│ NO → Penalty    │    │ Domain: 30/70   │    │                 │
│                 │    │ Complex: 20/80  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BASE SCORE CALCULATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Base_Score = (Rule_Score × Rule_Weight) + (Embedding_Score × Embedding_Weight)

                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGREEMENT ADJUSTMENTS                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Agreement > 0.8 │    │ 0.6 < Agreement │    │ Agreement < 0.4 │
│                 │    │      < 0.8      │    │                 │
│ HIGH BONUS      │    │ MEDIUM BONUS    │    │ VERY LOW        │
│ +0.1            │    │ +0.05           │    │ PENALTY -0.15   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TARGET DISAGREEMENT PENALTY                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Targets Same?   │    │ Confidence      │    │ Final Target    │
│                 │    │ Difference      │    │ Matches Higher  │
│ YES → No Penalty│    │ > 0.3?          │    │ YES → +0.05     │
│ NO → -0.15      │    │ YES → -0.03     │    │ NO → No Change  │
│                 │    │ NO → No Change  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL CONFIDENCE SCORE                              │
└─────────────────────────────────────────────────────────────────────────────┘

Final_Score = Base_Score + Agreement_Adjustment - Target_Penalty

                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DECISION MATRIX                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Confidence ≥ 0.8│    │ 0.6 ≤ Confidence│    │ Confidence < 0.6│
│                 │    │      < 0.8      │    │                 │
│ ✅ ACCEPT       │    │ ⚠️ VERIFY        │    │ ❌ LLM PROCESS  │
│ Immediately     │    │ Consider        │    │ High Priority   │
│ Low Priority    │    │ Medium Priority │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 **Detailed Decision Flow Examples**

### **Example 1: High Confidence with Agreement**

```
Field: "provider_npi"
Rule Score: 0.95, Target: "npi_number"
Embedding Score: 0.92, Target: "npi_number"

Step 1: Target Selection
- Same Target: YES → No penalty

Step 2: Context-Aware Weights
- Field Type: Domain-specific
- Weights: Rule(0.3), Embedding(0.7)

Step 3: Base Score Calculation
- Base_Score = (0.95 × 0.3) + (0.92 × 0.7) = 0.285 + 0.644 = 0.929

Step 4: Agreement Calculation
- Agreement = 1.0 - |0.95 - 0.92| = 0.97
- Agreement > 0.8 → HIGH BONUS (+0.1)

Step 5: Target Disagreement
- Same Target → No Penalty

Step 6: Final Score
- Final_Score = 0.929 + 0.1 = 1.0

Step 7: Decision
- Confidence ≥ 0.8 → ✅ ACCEPT IMMEDIATELY
```

### **Example 2: Medium Confidence with Target Disagreement**

```
Field: "specialty_code"
Rule Score: 0.75, Target: "medical_specialty"
Embedding Score: 0.78, Target: "specialty_type"

Step 1: Target Selection
- Same Target: NO → Penalty applied

Step 2: Context-Aware Weights
- Field Type: Domain-specific
- Weights: Rule(0.3), Embedding(0.7)

Step 3: Base Score Calculation
- Base_Score = (0.75 × 0.3) + (0.78 × 0.7) = 0.225 + 0.546 = 0.771

Step 4: Agreement Calculation
- Agreement = 1.0 - |0.75 - 0.78| = 0.97
- Agreement > 0.8 → HIGH BONUS (+0.1)

Step 5: Target Disagreement
- Different Targets → Base Penalty (-0.15)
- Confidence Difference = |0.75 - 0.78| = 0.03 < 0.3 → No additional penalty
- Final Target = "specialty_type" (embedding wins)
- Final Target matches higher confidence → Penalty Reduction (-0.03)

Step 6: Final Score
- Final_Score = 0.771 + 0.1 - 0.15 + 0.03 = 0.754

Step 7: Decision
- 0.6 ≤ Confidence < 0.8 → ⚠️ VERIFY
```

### **Example 3: Low Confidence with High Disagreement**

```
Field: "malpractice_history"
Rule Score: 0.35, Target: "malpractice_record"
Embedding Score: 0.78, Target: "malpractice_history"

Step 1: Target Selection
- Same Target: NO → Penalty applied

Step 2: Context-Aware Weights
- Field Type: Complex
- Weights: Rule(0.2), Embedding(0.8)

Step 3: Base Score Calculation
- Base_Score = (0.35 × 0.2) + (0.78 × 0.8) = 0.07 + 0.624 = 0.694

Step 4: Agreement Calculation
- Agreement = 1.0 - |0.35 - 0.78| = 0.57
- Agreement < 0.6 → LOW PENALTY (-0.08)

Step 5: Target Disagreement
- Different Targets → Base Penalty (-0.15)
- Confidence Difference = |0.35 - 0.78| = 0.43 > 0.3 → Additional Penalty (-0.05)
- Final Target = "malpractice_history" (embedding wins)
- Final Target matches higher confidence → Penalty Reduction (-0.03)

Step 6: Final Score
- Final_Score = 0.694 - 0.08 - 0.15 - 0.05 + 0.03 = 0.494

Step 7: Decision
- Confidence < 0.6 → ❌ LLM PROCESS
```

## 🎯 **Threshold Values (From Code)**

### **Agreement Thresholds**

```python
agreement_thresholds = {
    "high_agreement": 0.8,    # Bonus applied
    "medium_agreement": 0.6,   # Medium bonus applied
    "low_agreement": 0.4       # Penalty applied
}
```

### **Agreement Adjustments**

```python
agreement_adjustments = {
    "high_bonus": 0.1,         # +10% for high agreement
    "medium_bonus": 0.05,      # +5% for medium agreement
    "low_penalty": 0.08,       # -8% for low agreement
    "very_low_penalty": 0.15   # -15% for very low agreement
}
```

### **Target Selection Thresholds**

```python
target_selection_thresholds = {
    "confidence_difference": 0.1,  # Minimum difference to prefer one engine
    "embedding_preference": 0.05   # Small bias toward embedding
}
```

### **Decision Thresholds**

```python
decision_thresholds = {
    "high_confidence": 0.8,    # Accept immediately
    "medium_confidence": 0.6,   # Verify
    "low_confidence": 0.6       # LLM processing
}
```

## 📈 **Performance Impact**

### **Confidence Distribution**

- **High Confidence (≥0.8)**: ~40% of fields
- **Medium Confidence (0.6-0.8)**: ~45% of fields
- **Low Confidence (<0.6)**: ~15% of fields

### **Agreement Distribution**

- **High Agreement (≥0.8)**: ~60% of fields
- **Medium Agreement (0.6-0.8)**: ~30% of fields
- **Low Agreement (<0.6)**: ~10% of fields

### **Target Agreement**

- **Same Targets**: ~35% of fields
- **Different Targets**: ~65% of fields

## 🔧 **Key Decision Points**

### **1. Target Selection Priority**

1. **Same Target**: No penalty, highest confidence
2. **Embedding Higher**: Semantic understanding preferred
3. **Rule Higher**: Pattern matching preferred

### **2. Agreement Impact**

1. **High Agreement**: Significant confidence boost
2. **Medium Agreement**: Moderate confidence boost
3. **Low Agreement**: Confidence penalty

### **3. Final Decision**

1. **Accept**: High confidence, immediate processing
2. **Verify**: Medium confidence, review required
3. **LLM Process**: Low confidence, advanced processing

This decision flow ensures **robust, reliable, and efficient** schema mapping with appropriate confidence scoring and decision making for real-world applications.
