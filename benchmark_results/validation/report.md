# Unified Real Data Benchmark Validation Report

Generated: 2026-01-23 11:21:50 UTC

## Summary

| Metric | Value |
|--------|-------|
| Total Checks | 73 |
| Passed | 72 |
| Failed | 0 |
| Warnings | 0 |
| Critical Failures | 0 |
| Overall Status | PASS |

## ARCH Compliance

Rules Passed: 11/11

## Phase Results

### ablation

Duration: 0ms | Passed: 6 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| e1_is_foundation | PASS | E1 foundation | E1 semantic, weight=1.0 |
| temporal_excluded | PASS | temporal excluded | E2-E4 weight=0 |
| enhancement_embedders_defined | PASS | enhancers defined | E5, E7, E10 in semantic |
| E5_Causal_measured_contribution | PASS | >= 70% | 72.0% |
| E8_Graph_measured_contribution | PASS | >= 5% | 100.0% |
| E10_Multimodal_measured_contribution | PASS | >= 5% | 30.8% |

### temporal

Duration: 0ms | Passed: 14 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| config_valid | PASS | valid | valid |
| decay_half_life_positive | PASS | > 0 | 86400 |
| e2_timestamp_span | PASS | >= 328.5 days (90%) | 340.7 days (93%) |
| e2_recency_scores_valid | PASS | [0.0, 1.0] | [0.00, 1.00] |
| e3_periodic_clusters | PASS | <= 10 clusters | 3 clusters |
| e3_hour_day_valid | PASS | hour < 24, day < 7 | all valid |
| e4_session_sequences | PASS | ~100 sessions | 43 sessions |
| e4_session_positions_monotonic | PASS | monotonic | monotonic |
| temporal_coverage | PASS | >= 90% | 100% |
| decay_exponential | PASS | 0.5 | 0.500 |
| decay_linear | PASS | 0.5 | 0.500 |
| decay_step | PASS | 1.0, 0.8, 0.5, 0.1 | 1.0, 0.8, 0.5, 0.1 |
| e2_measured_quality | PASS | >= 65% | 68.9% |
| e4_measured_accuracy | PASS | >= 80% before_after | before_after=100.0% |
| symmetry_validated | SKIP | symmetric | not computed |

### config

Duration: 0ms | Passed: 15 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| all_embedders_count | PASS | 13 | 13 |
| semantic_embedders_count | PASS | 7 | 7 |
| temporal_embedders_count | PASS | 3 | 3 |
| relational_embedders_count | PASS | 2 | 2 |
| structural_embedders_count | PASS | 1 | 1 |
| asymmetric_embedders_count | PASS | 3 | 3 |
| category_sum | PASS | 13 | 13 |
| semantic_weights | PASS | 1.0 | 1.0 |
| temporal_weights | PASS | 0.0 | 0.0 |
| relational_structural_weights | PASS | 0.5 | 0.5 |
| max_weighted_agreement | PASS | 8.5 | 8.5 |
| e1_is_foundation | PASS | E1 in semantic | E1 in semantic |
| default_fusion_strategy | PASS | E1Only | E1Only |
| embedder_indices | PASS | 0-12 | 0-12 |
| divergence_embedders | PASS | 7 semantic | 7 embedders |

### ground_truth

Duration: 11ms | Passed: 8 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| all_embedders_have_gt | PASS | 13/13 | 13/13 |
| semantic_gt_topic_based | PASS | 7/7 semantic | 7/7 |
| e4_gt_uses_sessions | PASS | session-based GT | 48 queries |
| e8_gt_uses_documents | PASS | graph-based GT | 48 queries |
| min_relevant_per_query | PASS | >= 3 avg | 95.6 avg |
| no_empty_embedder_gt | PASS | none empty | none empty |
| queries_have_chunk_ids | PASS | all have chunk_id | 48/48 |
| relevant_chunk_ids_valid | PASS | no nil UUIDs | 59625 valid |

### arch

Duration: 0ms | Passed: 11 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| ARCH-01 | PASS | 13 embedders | 13 embedders defined |
| ARCH-02 | PASS | distinct embedder indices | unique indices 0-12 |
| ARCH-09 | PASS | >= 2.5 threshold | max=8.5, threshold=2.5 |
| ARCH-10 | PASS | 7 semantic only | 7 semantic embedders |
| ARCH-12 | PASS | E1 is foundation | E1 in semantic, weight=1.0 |
| ARCH-13 | PASS | E1Only default | E1Only default, 3 strategies |
| ARCH-14 | PASS | temporal weight = 0.0 | E2-E4 = 0.0 |
| ARCH-18 | PASS | E5 asymmetric | E5 in asymmetric list |
| AP-73 | PASS | temporal excluded from fusion | E2-E4 weight=0 (excluded) |
| AP-74 | PASS | E12 for rerank only | E12 defined, rerank-only guidance |
| AP-75 | PASS | E13 for recall only | E13 defined, recall-only guidance |

### fusion

Duration: 0ms | Passed: 5 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| fusion_strategies_defined | PASS | 4 strategies | 4 strategies |
| default_is_e1only | PASS | E1Only | E1Only |
| e1only_semantics | PASS | E1 only | E1 is foundation (index=0, weight=1.0) |
| multispace_excludes_temporal | PASS | temporal excluded | E2-E4 weight=0.0 |
| pipeline_architecture | PASS | 3-stage pipeline | E13 → E1 → E12 |

### e1_foundation

Duration: 0ms | Passed: 5 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| e1_in_semantic | PASS | E1 in semantic | E1 in semantic |
| e1_is_entry_embedder | PASS | E1 is entry point | index=0, default=E1Only |
| default_is_e1only | PASS | E1Only | E1Only |
| e1_max_weight | PASS | 1.0 (max) | 1.0 |
| e1_used_for_divergence | PASS | E1 used | E1 in divergence |

### asymmetric

Duration: 0ms | Passed: 8 | Failed: 0 | Warnings: 0

| Check | Status | Expected | Actual |
|-------|--------|----------|--------|
| asymmetric_embedders_defined | PASS | 3 | 3 |
| e5_is_asymmetric | PASS | E5 asymmetric | E5 in asymmetric |
| e8_is_asymmetric | PASS | E8 asymmetric | E8 in asymmetric |
| e10_is_asymmetric | PASS | E10 asymmetric | E10 in asymmetric |
| direction_modifiers | PASS | 1.2, 0.8, 1.0 | 1.2, 0.8, 1.0 |
| E5_Causal_asymmetric_ratio | PASS | [1.35, 1.65] | 1.500 |
| E10_Multimodal_asymmetric_ratio | PASS | [1.35, 1.65] | 1.500 |
| E8_Graph_asymmetric_ratio | PASS | [1.35, 1.65] | 1.500 |

