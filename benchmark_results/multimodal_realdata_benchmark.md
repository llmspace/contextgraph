# E10 Multimodal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-23T19:42:07.547922770+00:00

## Benchmark Design (E1 Foundation + E10 Enhancement)

Per ARCH-12 and ARCH-16: E1 is THE semantic foundation, E10 enhances it.

This benchmark uses **DOCUMENT-LEVEL** ground truth:
- Relevant = chunks from the **same document**
- Tests E10's ability to enhance E1's semantic retrieval

**Key Features:**
- Real E10 embeddings with NO ground truth leakage
- E1 provides semantic foundation, E10 adds intent/context precision
- Ablation study validates E10 enhancement vs E1 baseline

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 500 |
| Documents | 2437 |
| Topics | 100 |
| E10 Coverage | 100.0% |
| Asymmetric E10 | true |

## E10 Asymmetry Verification

| Metric | Value | Target |
|--------|-------|--------|
| Total Vector Pairs | 500 | - |
| Identical Pairs | 0 | <50% |
| **Identity Rate** | **0.0%** | <50% |
| Avg Distinctness | 0.2488 | >0.1 |
| **Verification** | **PASS** | PASS |

## Intent Retrieval (query_as_intent -> doc_as_context)

**Note:** Uses document-level ground truth (same document = relevant).
E1+E10 blend should maintain E1's performance while adding intent precision.

| Metric | Value | Target |
|--------|-------|--------|
| Queries | 50 | - |
| **MRR Intent->Context** | **1.0000** | >0.15 |
| MRR E1 Baseline | 1.0000 | - |
| Improvement over E1 | 0.0% | >0% |
| Hits@1 | 100.0% | - |
| Hits@5 | 100.0% | - |

## Asymmetric Comparison

| Metric | Value | Target |
|--------|-------|--------|
| **Observed Ratio** | **1.00** | ~1.5 |
| Expected Ratio | 1.50 | 1.5 |
| Formula Compliant | NO | YES |

## E10 Contribution Analysis

**Note:** E10 contribution measures improvement over E1 baseline. When E1 achieves perfect MRR, there's no room for E10 enhancement to show improvement.

| Metric | Value | Target |
|--------|-------|--------|
| MRR E1+E10 Blend | 1.0000 | - |
| MRR E1 Only | 1.0000 | - |
| **E10 Contribution** | **0.0%** | >5% |

## Ablation Study

| Configuration | MRR |
|---------------|-----|
| E1 Only | 1.0000 |
| E10 Only | 0.9892 |
| E1+E10 Blend (0.3) | 1.0000 |
| Full 13-Space | 1.0000 |

### Blend Parameter Sweep

| Blend | MRR | P@5 |
|-------|-----|-----|
| 0.0 | 1.0000 | 0.9850 |
| 0.1 | 1.0000 | 0.9850 |
| 0.2 | 1.0000 | 0.9850 |
| 0.3 | 1.0000 | 0.9860 |
| 0.4 | 1.0000 | 0.9840 |
| 0.5 | 0.9962 | 0.9790 |
| 0.7 | 0.9956 | 0.9650 |
| 1.0 | 0.9892 | 0.9330 |

## Recommendations

- Asymmetry ratio (1.00) outside expected range (1.2-2.0). Consider tuning direction modifiers.
- E10 contribution below 5%. Consider increasing E10 weight in fusion formula.

