# TASK-CORE-004: Define Comparison Types

```xml
<task_spec id="TASK-CORE-004" version="5.0">
<metadata>
  <title>Define Comparison Types and Search Matrices</title>
  <status>COMPLETED</status>
  <updated>2026-01-09</updated>
  <completed>2026-01-09</completed>
  <layer>foundation</layer>
  <sequence>4</sequence>
  <implements>
    <requirement_ref>REQ-COMPARISON-01</requirement_ref>
    <requirement_ref>REQ-SEARCH-MATRIX-01</requirement_ref>
    <requirement_ref>ARCH-02: Apples-to-apples comparison only</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-CORE-002</task_ref><!-- Embedder enum -->
    <task_ref status="COMPLETED">TASK-CORE-003</task_ref><!-- TeleologicalArray/SemanticFingerprint -->
  </depends_on>
</metadata>

<verified_codebase_state date="2026-01-09">
## Current Implementation (VERIFIED WORKING)

All code compiles. All tests pass. 53+ related tests verified.

### File Locations (All paths relative to `crates/context-graph-core/src/`)

| File | Purpose | Key Types |
|------|---------|-----------|
| `teleological/comparison_error.rs` | Validation error types | `ComparisonValidationError` (7 variants), `WeightValues`, `ComparisonValidationResult<T>` |
| `teleological/matrix_search.rs` | Comparison types & search | `SearchStrategy` (7 variants), `ComparisonScope` (7 variants), `ComponentWeights`, `SimilarityBreakdown`, `TeleologicalMatrixSearch` |
| `teleological/synergy_matrix.rs` | 13x13 synergy matrix | `SynergyMatrix`, `SYNERGY_DIM=13`, `CROSS_CORRELATION_COUNT=78`, 8 predefined constructors |
| `teleological/mod.rs` | Module exports | Re-exports all public types |

### Types Implemented

#### `ComparisonValidationError` (comparison_error.rs)
```rust
pub enum ComparisonValidationError {
    WeightsNotNormalized { actual_sum, expected_sum, tolerance, weights: WeightValues },
    WeightOutOfRange { field_name: &'static str, value, min, max },
    MatrixNotSymmetric { row, col, value_ij, value_ji, tolerance },
    DiagonalNotUnity { index, actual, expected, tolerance },
    SynergyOutOfRange { row, col, value, min, max },
    SimilarityOutOfRange { component: &'static str, value },
    BreakdownInconsistent { computed_overall, stored_overall, tolerance },
}
```

#### `SearchStrategy` (matrix_search.rs)
```rust
pub enum SearchStrategy {
    Cosine,              // Simple cosine similarity
    Euclidean,           // Euclidean distance converted to similarity
    SynergyWeighted,     // Apply synergy matrix weights
    GroupHierarchical,   // Hierarchical group comparison
    CrossCorrelationDominant, // Cross-correlation dominant
    TuckerCompressed,    // Tucker decomposition
    Adaptive,            // Adaptive selection
}
```

#### `ComponentWeights` (matrix_search.rs)
```rust
pub struct ComponentWeights {
    pub purpose_vector: f32,       // default: 0.4
    pub cross_correlations: f32,   // default: 0.35
    pub group_alignments: f32,     // default: 0.15
    pub confidence: f32,           // default: 0.1
}

impl ComponentWeights {
    pub fn validate(&self) -> ComparisonValidationResult<()>; // Validates sum = 1.0 ± 0.001
    pub fn is_valid(&self) -> bool;
    pub fn normalize(&mut self);
}
```

#### `SynergyMatrix` Predefined Constructors (synergy_matrix.rs)
```rust
impl SynergyMatrix {
    pub fn with_base_synergies() -> Self;    // Constitution-defined base values
    pub fn semantic_focused() -> Self;        // Boosts E1 (Semantic)
    pub fn code_heavy() -> Self;              // Boosts E6 (Code), E4 (Causal)
    pub fn temporal_focused() -> Self;        // Boosts E2, E3 (Temporal)
    pub fn causal_reasoning() -> Self;        // Boosts E4, E7 (Causal/Why)
    pub fn relational() -> Self;              // Boosts E5, E8, E9 (Relations)
    pub fn qualitative() -> Self;             // Boosts E10, E11 (Quality)
    pub fn balanced() -> Self;                // Uniform 0.6 off-diagonal
    pub fn identity() -> Self;                // Identity (0.0 off-diagonal)

    pub fn validate(&self) -> ComparisonValidationResult<()>;
    pub fn is_valid(&self) -> bool;
    pub fn assert_valid(&self);  // Panics if invalid
}
```
</verified_codebase_state>

<verification_commands>
## Verification Commands (All Must Pass)

```bash
# 1. Compile check
cargo check -p context-graph-core
# Expected: Compiles with at most warnings, no errors

# 2. Verify ComparisonValidationError exists
rg "pub enum ComparisonValidationError" crates/context-graph-core/src/teleological/
# Expected: comparison_error.rs

# 3. Verify ComponentWeights.validate() exists
rg "pub fn validate.*ComparisonValidationResult" crates/context-graph-core/src/teleological/matrix_search.rs
# Expected: 1 match

# 4. Verify SynergyMatrix.validate() exists
rg "pub fn validate.*ComparisonValidationResult" crates/context-graph-core/src/teleological/synergy_matrix.rs
# Expected: 2 matches (validate and validate_with_tolerance)

# 5. Verify predefined constructors
rg "pub fn semantic_focused|pub fn code_heavy|pub fn temporal_focused" crates/context-graph-core/src/teleological/synergy_matrix.rs
# Expected: 3 matches

# 6. Run all comparison-related tests
cargo test -p context-graph-core -- comparison --nocapture
# Expected: 10+ tests pass

# 7. Run all synergy tests
cargo test -p context-graph-core -- synergy --nocapture
# Expected: 51+ tests pass

# 8. Run all matrix tests
cargo test -p context-graph-core -- matrix --nocapture
# Expected: 53+ tests pass
```
</verification_commands>

<test_coverage>
## Test Coverage (All Pass)

| Test Category | Count | Status |
|---------------|-------|--------|
| ComparisonValidationError Display | 7 | PASS |
| ComparisonValidationError Traits | 2 | PASS |
| ComponentWeights Validation | 5 | PASS |
| SynergyMatrix Validation | 3 | PASS |
| SynergyMatrix Constructors | 8 | PASS |
| SynergyMatrix Core | 17 | PASS |
| MatrixSearch Integration | 11 | PASS |
| **Total** | **53+** | **ALL PASS** |
</test_coverage>

<source_of_truth>
## Source of Truth (For Full State Verification)

| Component | Location | Verification Command |
|-----------|----------|---------------------|
| ComparisonValidationError | `src/teleological/comparison_error.rs` | `rg "pub enum ComparisonValidationError" --type rust` |
| ComponentWeights | `src/teleological/matrix_search.rs:60-80` | `rg "pub struct ComponentWeights" --type rust` |
| SynergyMatrix | `src/teleological/synergy_matrix.rs:22-30` | `rg "pub struct SynergyMatrix" --type rust` |
| SearchStrategy | `src/teleological/matrix_search.rs:15-26` | `rg "pub enum SearchStrategy" --type rust` |
| Re-exports | `src/teleological/mod.rs:111-113` | `rg "ComparisonValidationError" src/teleological/mod.rs` |

**Critical**: All paths are relative to `crates/context-graph-core/`
**Note**: Directory is `teleological/` NOT `teleology/`
</source_of_truth>

<no_action_required>
## What This Task Already Implemented (NO FURTHER ACTION)

1. **ComparisonValidationError enum** - 7 variants with actionable error messages
2. **ComponentWeights.validate()** - Validates weights sum to 1.0 ± tolerance
3. **ComponentWeights.normalize()** - Auto-normalizes weights
4. **SynergyMatrix.validate()** - Validates symmetry, diagonal, ranges
5. **SynergyMatrix predefined constructors** - 8 domain-specific matrices
6. **WeightValues struct** - Debug data for weight errors
7. **ComparisonValidationResult<T>** - Type alias for Result
8. **All re-exports in mod.rs** - Types accessible from `crate::teleological::`
</no_action_required>

<downstream_tasks>
## Downstream Tasks Unblocked

| Task | Dependency | What It Uses |
|------|------------|--------------|
| TASK-LOGIC-001 | ComponentWeights | Validated weights for similarity calculations |
| TASK-LOGIC-004 | All types | TeleologicalComparator uses SearchStrategy, SynergyMatrix |
| TASK-LOGIC-005 | SynergyMatrix | Single-embedder search with synergy weights |
| TASK-LOGIC-007 | All predefined matrices | Matrix strategy selection |
</downstream_tasks>

<constitution_compliance>
## Constitution Compliance (VERIFIED)

| Rule | Compliance | Evidence |
|------|------------|----------|
| ARCH-02: Apples-to-apples | YES | SearchStrategy operates on same embedder types only |
| AP-14: No .unwrap() | YES | All methods return Result or Option |
| FAIL FAST | YES | validate() returns Err immediately on any violation |
| NO MOCK DATA | YES | Tests use real type instances |
| NO BACKWARDS COMPAT | YES | No deprecated shims or type aliases |
</constitution_compliance>

<edge_cases_tested>
## Edge Cases Tested

| Scenario | Test | Result |
|----------|------|--------|
| Weights sum = 0.999 | test_component_weights_default_validates | VALID (within tolerance) |
| Weights sum = 1.002 | test_weights_not_normalized_error_display | INVALID, clear error |
| Negative weight | test_weight_out_of_range_error_display | INVALID, identifies field |
| Diagonal != 1.0 | test_diagonal_not_unity_error_display | INVALID, identifies index |
| Synergy > 1.0 | test_synergy_out_of_range_error_display | INVALID |
| Asymmetric matrix | test_matrix_not_symmetric_error_display | INVALID, identifies pair |
| BASE_SYNERGIES | test_synergy_matrix_with_base_synergies | VALID |
| All predefined matrices | Multiple constructor tests | ALL VALID |
</edge_cases_tested>

<full_state_verification_protocol>
## Full State Verification Protocol

After any changes to this code, execute this verification:

### 1. Define Source of Truth
- **Primary**: Rust types compile without errors
- **Secondary**: All tests pass with real data

### 2. Execute & Inspect
```bash
# Compile
cargo check -p context-graph-core

# Run tests
cargo test -p context-graph-core -- comparison synergy matrix --nocapture

# Verify no regressions
cargo test -p context-graph-core 2>&1 | grep -E "(FAILED|passed|failed)"
```

### 3. Boundary & Edge Case Audit

**Case 1: Empty/Zero Weights**
```rust
let weights = ComponentWeights { purpose_vector: 0.0, cross_correlations: 0.0, group_alignments: 0.0, confidence: 0.0 };
assert!(weights.validate().is_err()); // Sum = 0.0, invalid
```

**Case 2: Maximum Weights**
```rust
let weights = ComponentWeights { purpose_vector: 1.0, cross_correlations: 0.0, group_alignments: 0.0, confidence: 0.0 };
assert!(weights.validate().is_ok()); // Sum = 1.0, valid
```

**Case 3: Invalid Matrix**
```rust
let mut matrix = SynergyMatrix::identity();
matrix.values[0][0] = 0.5; // Diagonal not 1.0
assert!(matrix.validate().is_err());
```

### 4. Evidence of Success
```
running 53 tests
...
test result: ok. 53 passed; 0 failed; 0 ignored
```
</full_state_verification_protocol>

</task_spec>
```
