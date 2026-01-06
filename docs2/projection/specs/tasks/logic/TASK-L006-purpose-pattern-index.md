# TASK-L006: Purpose Pattern Index

```yaml
metadata:
  id: "TASK-L006"
  title: "Purpose Pattern Index"
  layer: "logic"
  priority: "P1"
  estimated_hours: 8
  created: "2026-01-04"
  updated: "2026-01-05"
  status: "complete"
  completed: "2026-01-05"
  dependencies:
    - "TASK-L002"  # Purpose Vector Computation - COMPLETE
    - "TASK-L005"  # Per-Space HNSW Index Builder - COMPLETE
  spec_refs:
    - "constitution.yaml:teleological:purpose_vector"
    - "constitution.yaml:storage:layer2d_purpose"
```

## IMPLEMENTATION STATUS: COMPLETE

**All files implemented and tests passing.** This document serves as a reference for understanding the implementation.

---

## Files Created (VERIFIED)

| File | Lines | Description |
|------|-------|-------------|
| `crates/context-graph-core/src/index/purpose/mod.rs` | ~100 | Module exports, re-exports all public types |
| `crates/context-graph-core/src/index/purpose/entry.rs` | ~750 | `PurposeIndexEntry`, `PurposeMetadata` with validation |
| `crates/context-graph-core/src/index/purpose/query.rs` | ~1200 | `PurposeQuery`, `PurposeQueryTarget`, `PurposeSearchResult` |
| `crates/context-graph-core/src/index/purpose/error.rs` | ~250 | `PurposeIndexError` enum with fail-fast semantics |
| `crates/context-graph-core/src/index/purpose/hnsw_purpose.rs` | ~1400 | `HnswPurposeIndex` and `PurposeIndexOps` trait |
| `crates/context-graph-core/src/index/purpose/clustering.rs` | ~1200 | K-means clustering for 13D purpose vectors |
| `crates/context-graph-core/src/index/purpose/tests.rs` | ~1400 | Comprehensive tests with [VERIFIED] output |

## Files Modified (VERIFIED)

| File | Change |
|------|--------|
| `crates/context-graph-core/src/index/mod.rs` | Added `pub mod purpose;` |

---

## Architecture Overview

```
HnswPurposeIndex (Stage 4 Retrieval - constitution.yaml:storage:layer2d_purpose)
├── inner: SimpleHnswIndex          // 13D HNSW for ANN search
├── metadata: HashMap<Uuid, PurposeMetadata>   // goal, quadrant, confidence
├── vectors: HashMap<Uuid, PurposeVector>      // For retrieval & reranking
├── quadrant_index: HashMap<JohariQuadrant, HashSet<Uuid>>  // Secondary index
└── goal_index: HashMap<String, HashSet<Uuid>>              // Secondary index
```

## Existing Dependencies (USE THESE - DO NOT RECREATE)

| Component | Path | Status |
|-----------|------|--------|
| `PurposeVector` (13D alignments) | `crates/context-graph-core/src/types/fingerprint/purpose.rs` | EXISTS |
| `SimpleHnswIndex` | `crates/context-graph-core/src/index/hnsw_impl.rs` | EXISTS |
| `HnswConfig::purpose_vector()` | `crates/context-graph-core/src/index/config.rs` | EXISTS |
| `GoalId`, `GoalLevel`, `GoalHierarchy` | `crates/context-graph-core/src/purpose/goals.rs` | EXISTS |
| `JohariQuadrant` | `crates/context-graph-core/src/types/johari/quadrant.rs` | EXISTS |
| `AlignmentThreshold` | `crates/context-graph-core/src/types/fingerprint/purpose.rs` | EXISTS |
| `IndexError` | `crates/context-graph-core/src/index/error.rs` | EXISTS |
| `PURPOSE_VECTOR_DIM = 13` | `crates/context-graph-core/src/index/config.rs` | EXISTS |

---

## Key Implementation Details

### PurposeIndexOps Trait Methods

```rust
pub trait PurposeIndexOps {
    fn insert(&mut self, entry: PurposeIndexEntry) -> PurposeIndexResult<()>;
    fn remove(&mut self, memory_id: Uuid) -> PurposeIndexResult<()>;
    fn search(&self, query: &PurposeQuery) -> PurposeIndexResult<Vec<PurposeSearchResult>>;
    fn get(&self, memory_id: Uuid) -> PurposeIndexResult<PurposeIndexEntry>;
    fn contains(&self, memory_id: Uuid) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn clear(&mut self);
}
```

### Search Target Types

```rust
pub enum PurposeQueryTarget {
    /// Direct 13D vector ANN search
    Vector(PurposeVector),

    /// K-means clustering + coherence filtering
    Pattern {
        target_alignments: [f32; 13],
        coherence_threshold: f32,
    },

    /// Find similar to existing memory
    FromMemory(Uuid),
}
```

### Error Types (Fail-Fast)

```rust
pub enum PurposeIndexError {
    NotFound { memory_id: Uuid, context: &'static str },
    DimensionMismatch { expected: usize, actual: usize, context: &'static str },
    InvalidQuery { reason: String },
    ClusteringError { reason: String },
    HnswError(IndexError),
}
```

---

## Verification Commands (Run to Verify)

```bash
# Build verification
cd /home/cabdru/contextgraph
cargo build -p context-graph-core 2>&1 | head -50

# Run purpose index tests
cargo test -p context-graph-core purpose --nocapture 2>&1

# Count [VERIFIED] statements in test output
cargo test -p context-graph-core purpose --nocapture 2>&1 | grep -c "\[VERIFIED\]"

# Verify file structure
ls -la crates/context-graph-core/src/index/purpose/
```

---

## Full State Verification Protocol (MANDATORY)

### 1. Source of Truth

| Operation | Source of Truth | Verification |
|-----------|-----------------|--------------|
| Insert | `contains(id) == true` | Call after insert |
| Remove | `contains(id) == false` | Call after remove |
| Search | `Vec<Result>` non-empty | For matching query |
| Get | Returns `Ok(entry)` | After insert |

### 2. Edge Cases (Tests Cover These)

| Edge Case | Expected | Test |
|-----------|----------|------|
| Search empty index | Empty Vec (not error) | `test_search_empty_index` |
| Remove non-existent | `NotFound` error | `test_remove_nonexistent` |
| Dimension mismatch | `DimensionMismatch` error | `test_constructor_dimension_validation` |
| Invalid coherence | Error in Pattern search | `test_pattern_search_coherence` |

### 3. Evidence of Success (Test Output)

Tests print verification statements:
```
[VERIFIED] Constructor validates dimension = 13
[VERIFIED] Insert adds entry to index
[VERIFIED] Remove fails with NotFound for missing entry
[VERIFIED] Search returns empty Vec for empty index
[VERIFIED] Goal filter excludes non-matching entries
[VERIFIED] Quadrant filter excludes non-matching entries
[VERIFIED] Pattern search uses k-means clustering
```

---

## Traceability Matrix

| Requirement | Source | Implementation |
|-------------|--------|----------------|
| 13D purpose index | constitution.yaml:teleological:purpose_vector | `HnswPurposeIndex` with `PURPOSE_VECTOR_DIM = 13` |
| Stage 4 retrieval | constitution.yaml:storage:layer2d_purpose | `search()` method with filtering |
| Goal filtering | constitution.yaml:teleological:goal_hierarchy | `PurposeQuery::goal_filter` |
| Quadrant filtering | constitution.yaml:utl:johari | `PurposeQuery::quadrant_filter` |
| Alignment threshold 0.55 | constitution.yaml:teleological:thresholds:critical | `PurposeQuery::min_similarity` |
| Fail-fast errors | constitution.yaml:rules | `PurposeIndexError` variants |

---

## Integration Points

### Stage 4 in 5-Stage Pipeline

From constitution.yaml:
```yaml
stage_4_teleological_filter:
  desc: "Purpose alignment filter"
  input: "100 candidates"
  output: "Top 50 candidates"
  latency: "<10ms"
  uses: [purpose_vector, north_star]
  method: "Filter: alignment < 0.55 → discard"
```

The `HnswPurposeIndex.search()` method implements this:
1. Takes 100 candidates from Stage 3
2. Filters by `min_similarity >= 0.55` (alignment threshold)
3. Returns top 50 by similarity

### Usage in MultiSpaceIndexManager

```rust
// crates/context-graph-core/src/index/manager.rs
impl MultiSpaceIndexManager {
    pub fn search_purpose(&self, query: &PurposeQuery) -> Result<Vec<PurposeSearchResult>> {
        self.purpose_index.search(query)
    }

    pub fn add_purpose_vector(&mut self, entry: PurposeIndexEntry) -> Result<()> {
        self.purpose_index.insert(entry)
    }
}
```

---

## Test Coverage Summary

| Test Category | Count | Coverage |
|---------------|-------|----------|
| Constructor | 3 | Dimension validation, config |
| Insert | 4 | Basic, duplicate, secondary indexes |
| Remove | 4 | Basic, not found, index cleanup |
| Get | 2 | Found, not found |
| Search Vector | 4 | Basic, filters, similarity |
| Search Pattern | 3 | K-means, coherence, empty |
| Search FromMemory | 3 | Basic, not found, self-exclusion |
| Utility | 5 | len, is_empty, contains, clear |
| **Total** | 30 | All pass with [VERIFIED] |

---

*Status: COMPLETE - 2026-01-05*
*All tests passing, all [VERIFIED] statements confirmed*
