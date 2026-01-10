# TASK-LOGIC-004: Teleological Comparator

```xml
<task_spec id="TASK-LOGIC-004" version="6.0">
<metadata>
  <title>Implement TeleologicalComparator</title>
  <status>COMPLETED</status>
  <updated>2026-01-09</updated>
  <layer>logic</layer>
  <sequence>14</sequence>
  <implements>
    <requirement_ref>REQ-COMPARATOR-01</requirement_ref>
    <requirement_ref>REQ-APPLES-TO-APPLES-02</requirement_ref>
    <requirement_ref>ARCH-02: Compare Only Compatible Embedding Types</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-CORE-004</task_ref>
    <task_ref status="COMPLETED">TASK-LOGIC-001</task_ref>
    <task_ref status="COMPLETED">TASK-LOGIC-002</task_ref>
    <task_ref status="COMPLETED">TASK-LOGIC-003</task_ref>
  </depends_on>
</metadata>

<completion_summary>
## IMPLEMENTATION COMPLETE

**Verified**: 2026-01-09
**Tests**: 16 passing
**Commit**: Pending push

### What Was Built

| Component | Location | Status |
|-----------|----------|--------|
| `TeleologicalComparator` | `teleological/comparator.rs:95` | DONE |
| `ComparisonResult` | `teleological/comparator.rs:43` | DONE |
| `BatchComparator` | `teleological/comparator.rs:551` | DONE |
| Module export | `teleological/mod.rs` | DONE |
| rayon dependency | `Cargo.toml` | DONE |

### Test Results

```
running 16 tests
test teleological::comparator::tests::test_adaptive_strategy ... ok
test teleological::comparator::tests::test_apples_to_apples ... ok
test teleological::comparator::tests::test_batch_all_pairs ... ok
test teleological::comparator::tests::test_batch_one_to_many ... ok
test teleological::comparator::tests::test_breakdown_generation ... ok
test teleological::comparator::tests::test_coherence_computation ... ok
test teleological::comparator::tests::test_compare_different ... ok
test teleological::comparator::tests::test_compare_identical ... ok
test teleological::comparator::tests::test_compare_no_overlap ... ok
test teleological::comparator::tests::test_compare_strategies ... ok
test teleological::comparator::tests::test_dominant_embedder ... ok
test teleological::comparator::tests::test_group_hierarchical ... ok
test teleological::comparator::tests::test_invalid_weights_fail_fast ... ok
test teleological::comparator::tests::test_no_unwrap_calls ... ok
test teleological::comparator::tests::test_similarity_range ... ok
test teleological::comparator::tests::test_synergy_weighted ... ok

test result: ok. 16 passed; 0 failed; 0 ignored
```

### Constraints Verified

| Constraint | Status | Evidence |
|------------|--------|----------|
| APPLES-TO-APPLES | PASS | `compare_embedder_slices()` only matches same EmbeddingSlice variants |
| FAIL-FAST | PASS | `self.config.weights.validate()?` at start of `compare()` |
| NO-UNWRAP | PASS | `rg "\.unwrap\(\)"` returns 0 matches |
| WEIGHT-SUM | PASS | Uses `ComponentWeights::validate()` |
| SIMILARITY-RANGE | PASS | All outputs clamped to [0.0, 1.0] |
| RAYON-PARALLEL | PASS | `BatchComparator` uses `par_iter()` |

</completion_summary>

<context>
The central comparison engine that combines all similarity functions to compare
teleological arrays/fingerprints. This is the "apples-to-apples" enforcement layer:
- Routes to correct similarity function based on embedder type (dense/sparse/token-level)
- Applies weights via ComponentWeights and synergy via SynergyMatrix
- Returns detailed comparison results with per-embedder breakdown
- NEVER compares E1 (Semantic) to E5 (Causal) - same embedder type only
</context>

<implementation_details>
## Embedder Type Dispatch (Actual Implementation)

The `compare_embedder_slices()` method matches on `EmbeddingSlice` variants:

```rust
fn compare_embedder_slices(&self, a: &EmbeddingSlice<'_>, b: &EmbeddingSlice<'_>) -> Option<f32> {
    match (a, b) {
        (EmbeddingSlice::Dense(a_dense), EmbeddingSlice::Dense(b_dense)) => {
            // Uses cosine_similarity from similarity/dense.rs
            match cosine_similarity(a_dense, b_dense) {
                Ok(sim) => Some(sim.clamp(0.0, 1.0)),
                Err(_) => None,
            }
        }
        (EmbeddingSlice::Sparse(a_sparse), EmbeddingSlice::Sparse(b_sparse)) => {
            // Uses sparse_cosine_similarity from similarity/sparse.rs
            Some(sparse_cosine_similarity(a_sparse, b_sparse).clamp(0.0, 1.0))
        }
        (EmbeddingSlice::TokenLevel(a_tokens), EmbeddingSlice::TokenLevel(b_tokens)) => {
            // Uses max_sim from similarity/token_level.rs
            Some(max_sim(a_tokens, b_tokens).clamp(0.0, 1.0))
        }
        _ => None, // Type mismatch = no comparison
    }
}
```

## Aggregation Strategies (All 7 Implemented)

| Strategy | Method | Description |
|----------|--------|-------------|
| Cosine | `aggregate_mean()` | Simple weighted mean |
| Euclidean | `aggregate_euclidean()` | RMS distance converted to similarity |
| SynergyWeighted | `aggregate_synergy()` | Uses SynergyMatrix diagonal weights |
| GroupHierarchical | `aggregate_hierarchical()` | Groups embedders by type, averages groups |
| CrossCorrelationDominant | `aggregate_correlation()` | Uses off-diagonal synergy for pair weighting |
| TuckerCompressed | `aggregate_tucker()` | Weights by distance from mean |
| Adaptive | `aggregate_adaptive()` | Chooses strategy based on CoV |

## Key Files

| File | Purpose |
|------|---------|
| `crates/context-graph-core/src/teleological/comparator.rs` | Main implementation (1085 lines) |
| `crates/context-graph-core/src/teleological/mod.rs` | Module exports |
| `crates/context-graph-core/Cargo.toml` | rayon dependency |
| `crates/context-graph-core/src/similarity/dense.rs` | `cosine_similarity()` |
| `crates/context-graph-core/src/similarity/sparse.rs` | `sparse_cosine_similarity()` |
| `crates/context-graph-core/src/similarity/token_level.rs` | `max_sim()` |

</implementation_details>

<api_reference>
## Public API

### TeleologicalComparator

```rust
pub struct TeleologicalComparator {
    config: MatrixSearchConfig,
}

impl TeleologicalComparator {
    pub fn new() -> Self;
    pub fn with_config(config: MatrixSearchConfig) -> Self;
    pub fn config(&self) -> &MatrixSearchConfig;

    pub fn compare(
        &self,
        a: &SemanticFingerprint,
        b: &SemanticFingerprint,
    ) -> ComparisonValidationResult<ComparisonResult>;

    pub fn compare_with_strategy(
        &self,
        a: &SemanticFingerprint,
        b: &SemanticFingerprint,
        strategy: SearchStrategy,
    ) -> ComparisonValidationResult<ComparisonResult>;
}
```

### ComparisonResult

```rust
pub struct ComparisonResult {
    pub overall: f32,                           // [0.0, 1.0]
    pub per_embedder: [Option<f32>; 13],        // Per-embedder scores
    pub strategy: SearchStrategy,
    pub coherence: Option<f32>,                 // 1/(1+CoV)
    pub dominant_embedder: Option<Embedder>,
    pub breakdown: Option<SimilarityBreakdown>,
}

impl ComparisonResult {
    pub fn valid_score_count(&self) -> usize;
}
```

### BatchComparator

```rust
pub struct BatchComparator {
    comparator: TeleologicalComparator,
}

impl BatchComparator {
    pub fn new() -> Self;
    pub fn with_config(config: MatrixSearchConfig) -> Self;
    pub fn comparator(&self) -> &TeleologicalComparator;

    pub fn compare_one_to_many(
        &self,
        reference: &SemanticFingerprint,
        targets: &[SemanticFingerprint],
    ) -> Vec<ComparisonValidationResult<ComparisonResult>>;

    pub fn compare_all_pairs(
        &self,
        fingerprints: &[SemanticFingerprint],
    ) -> Vec<Vec<f32>>;

    pub fn compare_above_threshold(
        &self,
        reference: &SemanticFingerprint,
        targets: &[SemanticFingerprint],
        min_similarity: f32,
    ) -> Vec<(usize, f32)>;
}
```

</api_reference>

<downstream_tasks>
## Tasks Unblocked by This Implementation

| Task | What It Uses | Status |
|------|--------------|--------|
| TASK-LOGIC-005 | `TeleologicalComparator` for single-embedder search | Ready to start |
| TASK-LOGIC-006 | `ComparisonResult` for search result ranking | Ready to start |
| TASK-LOGIC-007 | `MatrixSearchConfig` integration | Ready to start |
| TASK-LOGIC-008 | `BatchComparator` for parallel search | Ready to start |
| TASK-INTEG-001 | Memory MCP handlers (uses comparator for dedup) | Ready to start |
| TASK-INTEG-003 | Consciousness MCP handlers (uses coherence) | Ready to start |

</downstream_tasks>

<verification_commands>
## Verification (All Pass)

```bash
# Compile check
cargo check -p context-graph-core
# Result: Finished `dev` profile

# Run comparator tests
cargo test -p context-graph-core comparator -- --nocapture
# Result: 16 passed; 0 failed

# Verify no .unwrap() calls
rg "\.unwrap\(\)" crates/context-graph-core/src/teleological/comparator.rs
# Result: 0 matches

# Verify module export
rg "pub use comparator" crates/context-graph-core/src/teleological/mod.rs
# Result: pub use comparator::{BatchComparator, ComparisonResult, TeleologicalComparator};

# Verify rayon dependency
rg "rayon" crates/context-graph-core/Cargo.toml
# Result: rayon = "1.10"
```

</verification_commands>

<constitution_compliance>
## Constitution Compliance

| Rule | Required | Status | Evidence |
|------|----------|--------|----------|
| ARCH-02: Apples-to-apples | YES | PASS | EmbeddingSlice matching prevents cross-type comparison |
| AP-14: No .unwrap() | YES | PASS | 0 unwrap calls in comparator.rs |
| FAIL FAST | YES | PASS | validate()? at start of compare() |
| NO MOCK DATA | YES | PASS | Tests use real SemanticFingerprint with actual vectors |
| NO BACKWARDS COMPAT | YES | PASS | No deprecated shims |
| AP-01: All 13 embedders | YES | PASS | per_embedder is [Option<f32>; NUM_EMBEDDERS] where NUM_EMBEDDERS=13 |

</constitution_compliance>

</task_spec>
```
