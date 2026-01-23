# E10 Multimodal Benchmark with Real HuggingFace Data

**Generated:** 2026-01-23T21:17:45.779595493+00:00

## Benchmark Design (E1 Foundation + E10 Enhancement)

Per ARCH-12 and ARCH-16: E1 is THE semantic foundation, E10 enhances it.

This benchmark uses **ADJACENT CHUNK** ground truth:
- Relevant = only chunks at **chunk_idx +/- 1** from same document
- Creates a challenging retrieval task (0-2 relevant per query)
- Tests E10's ability to find structurally related context

**Key Features:**
- Real E10 embeddings with NO ground truth leakage
- Challenging task prevents ceiling effects (MRR=1.0)
- Ablation study validates E10 enhancement vs E1 baseline

## Configuration

| Parameter | Value |
|-----------|-------|
| Total Chunks | 300 |
| Documents | 2437 |
| Topics | 100 |
| E10 Coverage | 100.0% |
| Asymmetric E10 | true |

## E10 Asymmetry Verification

| Metric | Value | Target |
|--------|-------|--------|
| Total Vector Pairs | 300 | - |
| Identical Pairs | 0 | <50% |
| **Identity Rate** | **0.0%** | <50% |
| Avg Distinctness | 0.2499 | >0.1 |
| **Verification** | **PASS** | PASS |

## Intent Retrieval (query_as_intent -> doc_as_context)

**Note:** Uses adjacent chunk ground truth (chunk_idx +/- 1 from same document).
This creates a challenging retrieval task where E10's context understanding may help.

| Metric | Value | Target |
|--------|-------|--------|
| Queries | 200 | - |
| **MRR Intent->Context** | **0.8975** | >0.15 |
| MRR E1 Baseline | 0.9734 | - |
| Improvement over E1 | -7.8% | >0% |
| Hits@1 | 83.0% | - |
| Hits@5 | 97.0% | - |

## Asymmetric Comparison (Direction Modifier Validation)

**Note:** Direction modifiers (intent→context=1.2x, context→intent=0.8x) are applied per Constitution.
Raw ratio should be ~1.0, modified ratio should be ~1.5 (= 1.2/0.8).

| Metric | Value | Target |
|--------|-------|--------|
| **Raw Asymmetry Ratio** | **1.0111** | ~1.0 |
| **Modified Asymmetry Ratio** | **1.5166** | ~1.5 |
| Expected Ratio | 1.50 | 1.5 |
| Modifiers Applied | YES | YES |
| Formula Compliant | YES | YES |

### Raw vs Modified Similarity Averages

| Direction | Raw | Modified | Modifier |
|-----------|-----|----------|----------|
| Intent→Context | 0.2942 | 0.3531 | 1.2x |
| Context→Intent | 0.2910 | 0.2328 | 0.8x |

## E10 Contribution Analysis

**Note:** E10 contribution measures improvement over E1 baseline.
With adjacent chunk ground truth, the task is challenging enough to show E10's value.

| Metric | Value | Target |
|--------|-------|--------|
| MRR E1+E10 Blend | 0.8869 | - |
| MRR E1 Only | 0.9692 | - |
| **E10 Contribution** | **-8.5%** | >5% |

## Ablation Study

| Configuration | MRR |
|---------------|-----|
| E1 Only | 0.9750 |
| E10 Only | 0.3676 |
| E1+E10 Blend (0.3) | 0.8749 |
| Full 13-Space | 0.8749 |

### Blend Parameter Sweep

| Blend | MRR | P@5 |
|-------|-----|-----|
| 0.0 | 0.9750 | 0.3850 |
| 0.1 | 0.9642 | 0.3800 |
| 0.2 | 0.9301 | 0.3610 |
| 0.3 | 0.8749 | 0.3440 |
| 0.4 | 0.8020 | 0.3080 |
| 0.5 | 0.7301 | 0.2680 |
| 0.7 | 0.5535 | 0.2120 |
| 1.0 | 0.3676 | 0.1360 |

## Recommendations

- E10 intent->context performing worse than E1. Check E10 embedding quality.
- E10 contribution below 5%. Consider increasing E10 weight in fusion formula.

