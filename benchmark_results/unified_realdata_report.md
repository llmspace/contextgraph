# Unified Real Data Benchmark Report

**Generated:** 2026-01-23 21:14:55 UTC
**Duration:** 398.7s

## Executive Summary

- **Best Single Embedder:** E3_Periodic (MRR@10: 1.000)
- **E1 Foundation Baseline:** MRR@10: 1.000
- **Best Fusion Strategy:** Pipeline (MRR@10: 1.118, +11.8% over E1)
- **Constitutional Compliance:** FAIL (4/8 rules)

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Chunks | 58895 |
| Chunks Used | 200 |
| Total Documents | 2437 |
| Unique Topics | 100 |
| Queries Generated | 92 |
| Source Datasets | wikipedia |

### Top Topics

| Topic | Chunks | Percentage |
|-------|--------|------------|
| politics | 0 | 0.0% |
| andronicus | 0 | 0.0% |
| bay | 0 | 0.0% |
| baseball | 0 | 0.0% |
| basic | 0 | 0.0% |
| bronze | 0 | 0.0% |
| andronikos | 0 | 0.0% |
| adam | 0 | 0.0% |
| abd | 0 | 0.0% |
| alexis | 0 | 0.0% |

## Per-Embedder Results

### Retrieval Quality

| Embedder | Category | MRR@10 | P@5 | P@10 | R@20 | MAP |
|----------|----------|--------|-----|------|------|-----|
| E12_LateInteraction | semantic_enhancer | 1.000 | 0.996 | 0.992 | 0.577 | 0.000 |
| E6_Sparse | semantic_enhancer | 1.000 | 1.000 | 0.997 | 0.581 | 0.000 |
| E13_SPLADE | semantic_enhancer | 1.000 | 0.996 | 0.988 | 0.580 | 0.000 |
| E1_Semantic | foundation | 1.000 | 1.000 | 1.000 | 0.582 | 0.000 |
| E7_Code | semantic_enhancer | 1.000 | 1.000 | 1.000 | 0.582 | 0.000 |
| E8_Graph | relational_enhancer | 1.000 | 0.998 | 0.987 | 0.576 | 0.000 |
| E3_Periodic | temporal_context | 1.000 | 1.000 | 1.000 | 0.101 | 0.000 |
| E5_Causal | semantic_enhancer | 0.993 | 0.996 | 0.973 | 0.559 | 0.000 |
| E11_Entity | relational_enhancer | 0.990 | 0.985 | 0.978 | 0.572 | 0.000 |
| E10_Multimodal | semantic_enhancer | 0.984 | 0.974 | 0.957 | 0.556 | 0.000 |
| E9_HDC | structural_context | 0.962 | 0.959 | 0.949 | 0.097 | 0.000 |
| E4_Sequence | temporal_context | 0.763 | 0.572 | 0.551 | 0.530 | 0.000 |
| E2_Recency | temporal_context | 0.382 | 0.193 | 0.203 | 0.113 | 0.000 |

### Contribution vs E1 Baseline

| Embedder | MRR Improvement | Is Enhancement? |
|----------|-----------------|-----------------|
| E12_LateInteraction | +0.0% | No |
| E6_Sparse | +0.0% | No |
| E13_SPLADE | +0.0% | No |
| E7_Code | +0.0% | No |
| E8_Graph | +0.0% | No |
| E3_Periodic | +0.0% | No |
| E5_Causal | -0.7% | No |
| E11_Entity | -1.0% | No |
| E10_Multimodal | -1.6% | No |
| E9_HDC | -3.8% | No |
| E4_Sequence | -23.7% | No |
| E2_Recency | -61.8% | No |

### Asymmetric Embedder Ratios

| Embedder | Asymmetric Ratio | Target | Status |
|----------|------------------|--------|--------|
| E8_Graph | 1.000 | 1.50+/-0.15 | WARN |
| E5_Causal | 1.000 | 1.50+/-0.15 | WARN |
| E10_Multimodal | 1.000 | 1.50+/-0.15 | WARN |

## Fusion Strategy Comparison

| Strategy | MRR@10 | P@10 | R@20 | Latency (ms) | Quality/Latency |
|----------|--------|------|------|--------------|-----------------|
| E1Only | 1.000 | 1.000 | 0.582 | 0.4 | 2.5922 |
| MultiSpace | 1.065 | 0.959 | 1.172 | 1.9 | 0.5522 |
| Pipeline * | 1.118 | 1.062 | 1.225 | 3.1 | 0.3624 |

*Best strategy: Pipeline*

### Fusion Recommendations

- Use Pipeline for best quality
- E1Only provides best latency when quality requirements are moderate

## Cross-Embedder Analysis

### Best Complementary Pairs

| Pair | Complementarity Score |
|------|----------------------|
| E1_Semantic + E2_Recency | 0.846 |
| E2_Recency + E3_Periodic | 0.846 |
| E2_Recency + E6_Sparse | 0.846 |
| E2_Recency + E7_Code | 0.846 |
| E2_Recency + E8_Graph | 0.846 |
| E2_Recency + E12_LateInteraction | 0.846 |
| E2_Recency + E13_SPLADE | 0.846 |
| E2_Recency + E5_Causal | 0.838 |
| E2_Recency + E11_Entity | 0.834 |
| E2_Recency + E10_Multimodal | 0.828 |

### Redundancy Warnings

These pairs have high correlation but limited complementarity:

- E1_Semantic and E9_HDC (correlation: 0.963)
- E3_Periodic and E9_HDC (correlation: 0.963)
- E6_Sparse and E9_HDC (correlation: 0.963)
- E7_Code and E9_HDC (correlation: 0.963)
- E8_Graph and E9_HDC (correlation: 0.963)
- E9_HDC and E12_LateInteraction (correlation: 0.963)
- E9_HDC and E13_SPLADE (correlation: 0.963)
- E5_Causal and E9_HDC (correlation: 0.969)
- E9_HDC and E11_Entity (correlation: 0.972)
- E9_HDC and E10_Multimodal (correlation: 0.978)
- E1_Semantic and E10_Multimodal (correlation: 0.985)
- E3_Periodic and E10_Multimodal (correlation: 0.985)
- E6_Sparse and E10_Multimodal (correlation: 0.985)
- E7_Code and E10_Multimodal (correlation: 0.985)
- E8_Graph and E10_Multimodal (correlation: 0.985)
- E10_Multimodal and E12_LateInteraction (correlation: 0.985)
- E10_Multimodal and E13_SPLADE (correlation: 0.985)
- E1_Semantic and E11_Entity (correlation: 0.990)
- E3_Periodic and E11_Entity (correlation: 0.990)
- E6_Sparse and E11_Entity (correlation: 0.990)
- E7_Code and E11_Entity (correlation: 0.990)
- E8_Graph and E11_Entity (correlation: 0.990)
- E11_Entity and E12_LateInteraction (correlation: 0.990)
- E11_Entity and E13_SPLADE (correlation: 0.990)
- E5_Causal and E10_Multimodal (correlation: 0.992)
- E1_Semantic and E5_Causal (correlation: 0.993)
- E3_Periodic and E5_Causal (correlation: 0.993)
- E5_Causal and E6_Sparse (correlation: 0.993)
- E5_Causal and E7_Code (correlation: 0.993)
- E5_Causal and E8_Graph (correlation: 0.993)
- E5_Causal and E12_LateInteraction (correlation: 0.993)
- E5_Causal and E13_SPLADE (correlation: 0.993)
- E10_Multimodal and E11_Entity (correlation: 0.995)
- E5_Causal and E11_Entity (correlation: 0.997)
- E1_Semantic and E3_Periodic (correlation: 1.000)
- E1_Semantic and E6_Sparse (correlation: 1.000)
- E1_Semantic and E7_Code (correlation: 1.000)
- E1_Semantic and E8_Graph (correlation: 1.000)
- E1_Semantic and E12_LateInteraction (correlation: 1.000)
- E1_Semantic and E13_SPLADE (correlation: 1.000)
- E3_Periodic and E6_Sparse (correlation: 1.000)
- E3_Periodic and E7_Code (correlation: 1.000)
- E3_Periodic and E8_Graph (correlation: 1.000)
- E3_Periodic and E12_LateInteraction (correlation: 1.000)
- E3_Periodic and E13_SPLADE (correlation: 1.000)
- E6_Sparse and E7_Code (correlation: 1.000)
- E6_Sparse and E8_Graph (correlation: 1.000)
- E6_Sparse and E12_LateInteraction (correlation: 1.000)
- E6_Sparse and E13_SPLADE (correlation: 1.000)
- E7_Code and E8_Graph (correlation: 1.000)
- E7_Code and E12_LateInteraction (correlation: 1.000)
- E7_Code and E13_SPLADE (correlation: 1.000)
- E8_Graph and E12_LateInteraction (correlation: 1.000)
- E8_Graph and E13_SPLADE (correlation: 1.000)
- E12_LateInteraction and E13_SPLADE (correlation: 1.000)

## Ablation Study

### Impact of Adding Each Embedder to E1

| Embedder | MRR Change | P@10 Change | Significant? |
|----------|------------|-------------|--------------|
| E13_SPLADE | +0.0% | +0.0% | No |
| E7_Code | +0.0% | +0.0% | No |
| E6_Sparse | +0.0% | +0.0% | No |
| E3_Periodic | +0.0% | +0.0% | No |
| E12_LateInteraction | +0.0% | +0.0% | No |
| E8_Graph | +0.0% | +0.0% | No |
| E5_Causal | -0.3% | -0.3% | No |
| E11_Entity | -0.4% | -0.4% | No |
| E10_Multimodal | -0.6% | -0.6% | No |
| E9_HDC | -1.5% | -1.4% | No |
| E4_Sequence | -9.5% | -8.5% | Yes |
| E2_Recency | -24.7% | -22.3% | Yes |

### Critical Embedders

Removing these causes >10% degradation:
- E2_Recency

### Potentially Redundant Embedders

Removing these causes <2% change:
- E3_Periodic
- E5_Causal
- E6_Sparse
- E7_Code
- E8_Graph
- E9_HDC
- E10_Multimodal
- E11_Entity
- E12_LateInteraction
- E13_SPLADE

## Constitutional Compliance

**Overall Status:** FAILED

| Rule | Description | Status | Details |
|------|-------------|--------|---------|
| ARCH-09 | Topic threshold weighted_agreement >= 2.5 | PASS | weighted_sum=8.50, semantic_count=7 |
| AP-73 | Temporal embedders (E2-E4) not in similarity fusion | FAIL | Temporal embedders in fusion: [E2Recency, E3Period |
| ASYMMETRIC-E5_Causal | E5_Causal asymmetric ratio within 1.5 +/- 0.15 | FAIL | ratio=1.000, expected=1.500+/-0.150 |
| ASYMMETRIC-E8_Graph | E8_Graph asymmetric ratio within 1.5 +/- 0.15 | FAIL | ratio=1.000, expected=1.500+/-0.150 |
| ASYMMETRIC-E10_Multimodal | E10_Multimodal asymmetric ratio within 1.5 +/- 0.15 | FAIL | ratio=1.000, expected=1.500+/-0.150 |
| ARCH-14-E2_Recency | E2_Recency has topic_weight=0.0 | PASS | weight=0 |
| ARCH-14-E3_Periodic | E3_Periodic has topic_weight=0.0 | PASS | weight=0 |
| ARCH-14-E4_Sequence | E4_Sequence has topic_weight=0.0 | PASS | weight=0 |

## Recommendations

1. Best single embedder: E12_LateInteraction - use as baseline
2. Use Pipeline fusion for +11.8% improvement over E1

---

*Report generated by context-graph-benchmark unified-realdata-bench*
*Version: 0.1.0*
