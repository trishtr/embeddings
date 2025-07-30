# Decision Matrix Documentation

## Overview

This document explains how we use **aggregated scores** and **agreement scores** to make decisions in the schema mapping pipeline. Understanding this decision matrix is crucial for interpreting results and optimizing the mapping process.

## 🎯 **Primary Decision Matrix: Aggregated Score**

### **The Aggregated Score is the MAIN Decision Maker**

The **aggregated score** (after both context-aware weighting and agreement adjustments) serves as our **primary decision matrix**:

| Aggregated Score Range | Decision           | Action                     | Priority |
| ---------------------- | ------------------ | -------------------------- | -------- |
| **≥0.8**               | ✅ **Accept**      | Accept mapping immediately | Low      |
| **0.6-0.8**            | ⚠️ **Verify**      | Consider for verification  | Medium   |
| **<0.6**               | ❌ **LLM Process** | Send to LLM for processing | High     |

### **Why Aggregated Score is Primary**

- **Comprehensive**: Combines field optimization + quality assurance
- **Actionable**: Directly tells us what to do with each field
- **Scalable**: Works for any field type or domain
- **Consistent**: Provides uniform decision criteria across all fields

## 🤝 **Supporting Indicator: Agreement Score**

### **The Agreement Score is a QUALITY ASSURANCE Tool**

The **agreement score** serves as a **supporting indicator** for analysis and validation, not primary decisions:

### **Agreement Score Roles**

1. **Quality Indicator**: How reliable is this mapping?
2. **Uncertainty Detector**: Are the engines disagreeing?
3. **Debugging Tool**: Why is confidence low?
4. **Process Optimization**: Which fields need attention?

### **Agreement Score Thresholds**

| Agreement Score Range | Interpretation   | Quality Level                         |
| --------------------- | ---------------- | ------------------------------------- |
| **≥0.8**              | High Agreement   | Engines strongly confirm each other   |
| **0.6-0.8**           | Medium Agreement | Moderate confirmation between engines |
| **<0.6**              | Low Agreement    | Engines disagree significantly        |

## 📊 **Complete Decision Matrix**

### **Two-Dimensional Decision Matrix**

| Aggregated Score | Agreement Score | Decision           | Priority | Reasoning                               |
| ---------------- | --------------- | ------------------ | -------- | --------------------------------------- |
| **≥0.8**         | **≥0.8**        | ✅ Accept          | Low      | High confidence with engine agreement   |
| **≥0.8**         | **<0.6**        | ⚠️ Accept + Review | Medium   | High confidence but engines disagree    |
| **0.6-0.8**      | **≥0.7**        | 🔍 Verify          | Medium   | Moderate confidence with good agreement |
| **0.6-0.8**      | **<0.7**        | 🤖 LLM Process     | High     | Moderate confidence but disagreement    |
| **<0.6**         | **≥0.7**        | 🤖 LLM Process     | High     | Low confidence despite agreement        |
| **<0.6**         | **<0.6**        | 🚨 Urgent LLM      | Critical | Low confidence and disagreement         |

### **Decision Flow Examples**

#### **Scenario 1: High Aggregated Score + High Agreement**

- **Aggregated Score**: 0.92 (High confidence)
- **Agreement Score**: 0.95 (High agreement)
- **Decision**: ✅ **Accept immediately**
- **Reasoning**: Both engines agree and field is well-mapped
- **Action**: No further processing needed

#### **Scenario 2: High Aggregated Score + Low Agreement**

- **Aggregated Score**: 0.85 (High confidence)
- **Agreement Score**: 0.45 (Low agreement)
- **Decision**: ⚠️ **Accept but flag for review**
- **Reasoning**: High confidence but engines disagree - investigate
- **Action**: Accept mapping but log for analysis

#### **Scenario 3: Low Aggregated Score + High Agreement**

- **Aggregated Score**: 0.55 (Low confidence)
- **Agreement Score**: 0.88 (High agreement)
- **Decision**: ❌ **Send to LLM**
- **Reasoning**: Engines agree but both are uncertain
- **Action**: LLM processing with high priority

#### **Scenario 4: Low Aggregated Score + Low Agreement**

- **Aggregated Score**: 0.35 (Low confidence)
- **Agreement Score**: 0.30 (Low agreement)
- **Decision**: 🚨 **High priority LLM processing**
- **Reasoning**: Both engines uncertain and disagreeing
- **Action**: Urgent LLM processing with rich context

## 🔄 **Decision Processing Workflow**

### **Step 1: Calculate Scores**

1. **Context-Aware Aggregation**: Apply field-specific weights
2. **Agreement Adjustment**: Apply bonuses/penalties based on agreement
3. **Final Aggregated Score**: Primary decision metric
4. **Agreement Score**: Supporting quality indicator

### **Step 2: Apply Decision Matrix**

1. **Check Aggregated Score**: Determine primary action
2. **Check Agreement Score**: Understand quality context
3. **Apply Decision Rules**: Execute appropriate action
4. **Log Both Scores**: For analysis and improvement

### **Step 3: Execute Actions**

1. **Accept**: No further processing
2. **Verify**: Additional validation steps
3. **LLM Process**: Send to language model
4. **Urgent LLM**: High-priority processing

## 📈 **Analytics and Reporting**

### **Aggregated Score Analytics**

- **Distribution Analysis**: How many fields in each confidence tier
- **Performance Trends**: Mapping quality over time
- **Optimization Insights**: Which fields need improvement
- **Success Metrics**: Overall system performance

### **Agreement Score Analytics**

- **Quality Metrics**: Overall system reliability
- **Engine Performance**: Which engines agree/disagree
- **Domain Insights**: Which field types cause disagreement
- **Process Optimization**: Areas for improvement

### **Combined Analytics**

- **Decision Distribution**: How often each decision is made
- **Quality Correlation**: Relationship between scores and outcomes
- **Process Efficiency**: Resource usage optimization
- **Continuous Improvement**: Data-driven enhancements

## 🎯 **Key Decision Principles**

### **Aggregated Score is the "What"**

- **What** should we do with this field?
- **What** is our confidence level?
- **What** action should we take?

### **Agreement Score is the "Why"**

- **Why** is confidence high/low?
- **Why** should we trust/distrust this mapping?
- **Why** do we need additional processing?

### **Combined Decision Making**

- **Primary**: Use aggregated score for action
- **Secondary**: Use agreement score for context
- **Tertiary**: Use both for optimization

## 🔍 **Decision Matrix Applications**

### **For Individual Fields**

1. **Immediate Action**: Based on aggregated score
2. **Quality Assessment**: Based on agreement score
3. **Processing Priority**: Based on both scores
4. **Resource Allocation**: Optimize based on decision type

### **For System Optimization**

1. **Performance Monitoring**: Track decision distributions
2. **Quality Assurance**: Monitor agreement patterns
3. **Process Improvement**: Identify optimization opportunities
4. **Resource Planning**: Allocate resources based on decision patterns

### **For Domain-Specific Tuning**

1. **Healthcare Fields**: Higher standards for PHI-related fields
2. **Business Critical**: Enhanced validation for important fields
3. **Standard Fields**: Streamlined processing for common fields
4. **Complex Fields**: Specialized handling for ambiguous fields

## 🚀 **Optimization Strategies**

### **Based on Decision Patterns**

#### **High Accept Rate + High Agreement**

- **Strategy**: Optimize for speed
- **Action**: Reduce processing overhead
- **Goal**: Maintain quality while improving efficiency

#### **High LLM Rate + Low Agreement**

- **Strategy**: Improve engine training
- **Action**: Enhance rule-based patterns
- **Goal**: Reduce dependency on LLM processing

#### **Mixed Decisions + Variable Agreement**

- **Strategy**: Domain-specific optimization
- **Action**: Tune weights for specific field types
- **Goal**: Improve decision accuracy

### **Continuous Improvement Loop**

1. **Monitor Decisions**: Track decision distributions
2. **Analyze Patterns**: Identify trends and anomalies
3. **Optimize Parameters**: Adjust thresholds and weights
4. **Validate Changes**: Test improvements
5. **Deploy Updates**: Implement optimizations
6. **Repeat**: Continuous monitoring and improvement

## 📊 **Decision Matrix Metrics**

### **Key Performance Indicators (KPIs)**

#### **Efficiency Metrics**

- **Accept Rate**: Percentage of fields accepted immediately
- **LLM Usage Rate**: Percentage of fields requiring LLM
- **Processing Time**: Average time per decision
- **Resource Utilization**: CPU/memory usage per decision

#### **Quality Metrics**

- **Agreement Rate**: Percentage of high-agreement decisions
- **Confidence Distribution**: Spread of aggregated scores
- **Decision Accuracy**: Validation of accepted mappings
- **Error Rate**: Incorrect mapping decisions

#### **Business Metrics**

- **Cost per Decision**: Financial impact of decisions
- **Throughput**: Fields processed per unit time
- **Scalability**: Performance with increased load
- **Reliability**: Consistency of decision quality

## 🔧 **Decision Matrix Configuration**

### **Configurable Parameters**

#### **Aggregated Score Thresholds**

```yaml
aggregated_score_thresholds:
  high_confidence: 0.8
  medium_confidence: 0.6
  low_confidence: 0.4
```

#### **Agreement Score Thresholds**

```yaml
agreement_score_thresholds:
  high_agreement: 0.8
  medium_agreement: 0.6
  low_agreement: 0.4
```

#### **Decision Priorities**

```yaml
decision_priorities:
  accept: "low"
  verify: "medium"
  llm_process: "high"
  urgent_llm: "critical"
```

### **Domain-Specific Adjustments**

#### **Healthcare Domain**

- **Higher Standards**: Stricter thresholds for PHI fields
- **Enhanced Validation**: More verification steps
- **Quality Focus**: Prioritize accuracy over speed

#### **General Domain**

- **Balanced Approach**: Standard thresholds
- **Efficiency Focus**: Optimize for throughput
- **Flexible Processing**: Adaptive decision making

## 💡 **Best Practices**

### **Decision Matrix Usage**

1. **Consistent Application**: Apply matrix uniformly across all fields
2. **Regular Monitoring**: Track decision patterns continuously
3. **Data-Driven Optimization**: Use analytics to improve decisions
4. **Domain Awareness**: Adjust for specific field characteristics
5. **Quality Assurance**: Validate decisions with ground truth

### **Performance Optimization**

1. **Threshold Tuning**: Optimize based on domain requirements
2. **Weight Adjustment**: Fine-tune aggregation weights
3. **Process Streamlining**: Reduce overhead for common decisions
4. **Resource Allocation**: Prioritize based on decision importance
5. **Continuous Learning**: Improve based on decision outcomes

This decision matrix provides a **systematic, data-driven approach** to schema mapping decisions, ensuring both **quality and efficiency** in the mapping process.

### Decision Logic When Candidates Differ

1. Compare Confidence

# Rule-based result

rule_confidence = 0.75
rule_target = "medical_specialty"

# Embedding result (top candidate)

embedding_confidence = 0.88
embedding_target = "specialty_type"

# The candidates are DIFFERENT!

2. Apply Decision Rules :

# Option A: Embedding Wins:

if embedding_confidence > rule_confidence:
final_target = embedding_target # "specialty_type"
primary_confidence = embedding_confidence # 0.88
secondary_confidence = rule_confidence # 0.75
else:
final_target = rule_target
primary_confidence = rule_confidence
secondary_confidence = embedding_confidence

3. Calculate Agreement Level

# Agreement level is LOW because candidates differ

agreement_level = 1.0 - abs(0.88 - 0.75) = 0.87

# But this doesn't reflect the real disagreement

# The engines chose DIFFERENT targets!

4. Apply Disagreement Penalty

# Context-aware weights

weights = {"rule_based": 0.3, "embedding": 0.7}

# Calculate base aggregated score

base_score = (0.75 _ 0.3) + (0.88 _ 0.7) = 0.225 + 0.616 = 0.841

# Apply disagreement penalty (engines chose different targets)

disagreement_penalty = 0.15
final_score = max(0.0, 0.841 - 0.15) = 0.691

# Final result

aggregated_result = {
"specialty_cd": {
"target_field": "specialty_type", # Embedding's choice wins
"confidence": 0.691,
"agreement_level": 0.87,
"method": "embedding_preferred",
"uncertainty_factors": ["engine_disagreement", "different_targets"],
"rule_target": "medical_specialty",
"embedding_target": "specialty_type"
}
}
