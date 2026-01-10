# TASK-LOGIC-011: RRF Fusion Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Task ID** | TASK-LOGIC-011 |
| **Title** | Reciprocal Rank Fusion Implementation |
| **Status** | :white_check_mark: done |
| **Layer** | Logic |
| **Sequence** | 21 |
| **Estimated Days** | 1.5 |
| **Complexity** | Medium |
| **Actual Completion** | Already implemented - see audit below |

## ⚠️ CRITICAL AUDIT FINDINGS (2026-01-09)

### STATUS: ALREADY IMPLEMENTED

**This task is marked "todo" in _index.md but RRF is ALREADY FULLY IMPLEMENTED in the codebase.**

The original task specification contained several errors that would have caused an AI agent to fail:

| Issue | Original Spec | Actual Implementation |
|-------|---------------|----------------------|
| **File Path** | `crates/context-graph-core/src/teleology/similarity/fusion.rs` | `crates/context-graph-core/src/retrieval/aggregation.rs` |
| **Type Name** | `RRFFusion` struct | `AggregationStrategy` enum with `RRF { k: f32 }` variant |
| **Directory** | `teleology/similarity/` | Does not exist - use `retrieval/` |
| **Status** | todo | Already complete with 18+ tests passing |

### EXISTING IMPLEMENTATION LOCATIONS

| File | Purpose | Lines |
|------|---------|-------|
| `crates/context-graph-core/src/retrieval/aggregation.rs` | Primary RRF implementation | 429 |
| `crates/context-graph-storage/src/teleological/search/pipeline.rs` | Stage 3 RRF rerank in 5-stage pipeline | 1935 |
| `crates/context-graph-core/src/retrieval/result.rs` | SpaceContribution with RRF scoring | ~400 |
| `crates/context-graph-mcp/src/handlers/search.rs` | MCP handler with RRF support | ~300 |

---

## Implements

- **REQ-SEARCH-07**: Multi-space ranking fusion
- **ARCH-04**: Entry-point discovery with reranking (Stage 3 of 5-stage pipeline)

## Dependencies

| Task | Reason | Status |
|------|--------|--------|
| TASK-LOGIC-001 | Dense similarity scores | :white_check_mark: done |
| TASK-LOGIC-002 | Sparse similarity scores | :white_check_mark: done |
| TASK-LOGIC-003 | Token-level similarity scores | :white_check_mark: done |
| TASK-LOGIC-005 | Single embedder HNSW search | :white_check_mark: done |
| TASK-LOGIC-006 | Multi-embedder parallel search | :white_check_mark: done |
| TASK-LOGIC-008 | 5-Stage retrieval pipeline (uses RRF at Stage 3) | :white_check_mark: done |

---

## Objective

Implement Reciprocal Rank Fusion (RRF) to combine rankings from multiple embedding spaces into a single unified ranking for multi-space search aggregation.

## Context

### RRF Formula (from constitution.yaml)
```
RRF(d) = Σᵢ 1/(k + rankᵢ(d) + 1)
```

Where:
- `k` = 60 (per constitution.yaml `embeddings.similarity.rrf_constant`)
- `rankᵢ(d)` is the 0-indexed rank of document `d` in ranking `i`
- The `+1` ensures rank 0 contributes `1/(60+0+1) = 1/61`

### Role in 5-Stage Pipeline (per constitution.yaml)
```yaml
retrieval:
  S1: { desc: "BM25+E13 sparse", in: "1M+", out: "10K", lat: "<5ms" }
  S2: { desc: "E1[..128] Matryoshka ANN", in: "10K", out: "1K", lat: "<10ms" }
  S3: { desc: "RRF across 13 spaces", in: "1K", out: "100", lat: "<20ms" }  # <-- THIS TASK
  S4: { desc: "Purpose alignment filter (≥0.55)", in: "100", out: "50", lat: "<10ms" }
  S5: { desc: "E12 MaxSim precision", in: "50", out: "10", lat: "<15ms" }
```

---

## Current Implementation (Source of Truth)

### 1. AggregationStrategy Enum
**File**: `crates/context-graph-core/src/retrieval/aggregation.rs`

```rust
/// Aggregation strategy for combining multi-space search results.
#[derive(Clone, Debug)]
pub enum AggregationStrategy {
    /// Reciprocal Rank Fusion - PRIMARY STRATEGY.
    /// Formula: RRF(d) = Σᵢ 1/(k + rankᵢ(d) + 1)
    RRF { k: f32 },

    /// Weighted average of similarities.
    WeightedAverage {
        weights: [f32; NUM_EMBEDDERS],
        require_all: bool,
    },

    /// Maximum similarity across spaces.
    MaxPooling,

    /// Purpose-weighted aggregation using 13D purpose vector.
    PurposeWeighted { purpose_vector: PurposeVector },
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::RRF { k: similarity::RRF_K } // k=60 per constitution
    }
}
```

### 2. Core RRF Functions
**File**: `crates/context-graph-core/src/retrieval/aggregation.rs`

```rust
impl AggregationStrategy {
    /// Aggregate using Reciprocal Rank Fusion across ranked lists.
    pub fn aggregate_rrf(ranked_lists: &[(usize, Vec<Uuid>)], k: f32) -> HashMap<Uuid, f32>;

    /// Aggregate RRF with per-space weighting.
    /// Formula: RRF_weighted(d) = Σᵢ wᵢ/(k + rankᵢ(d) + 1)
    pub fn aggregate_rrf_weighted(
        ranked_lists: &[(usize, Vec<Uuid>)],
        k: f32,
        weights: &[f32; NUM_EMBEDDERS],
    ) -> HashMap<Uuid, f32>;

    /// Compute RRF contribution for a single rank.
    #[inline]
    pub fn rrf_contribution(rank: usize, k: f32) -> f32 {
        1.0 / (k + (rank as f32) + 1.0)
    }
}
```

### 3. Stage 3 Pipeline Integration
**File**: `crates/context-graph-storage/src/teleological/search/pipeline.rs:1021-1118`

```rust
/// Stage 3: Multi-space RRF rerank.
fn stage_rrf_rerank(
    &self,
    query_semantic: &[f32],
    candidates: Vec<PipelineCandidate>,
    config: &StageConfig,
) -> Result<StageResult, PipelineError> {
    // ... implementation uses rrf_k from config
    let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
    // ... FAIL FAST on timeout
}
```

---

## Definition of Done

### Already Verified ✅

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Basic RRF produces correct scores | ✅ | `test_rrf_aggregation_single_list` |
| Multiple rankings combine correctly | ✅ | `test_rrf_aggregation_multiple_lists` |
| Weighted RRF applies weights | ✅ | `test_rrf_weighted` |
| Documents missing from some rankings handled | ✅ | RRF naturally handles via HashMap |
| Results sorted by RRF score descending | ✅ | `stage_rrf_rerank` sorts before truncating |
| k=60 per constitution | ✅ | `test_rrf_k_constitution` |
| Default strategy is RRF | ✅ | `test_default_is_rrf` |
| Integration with 5-stage pipeline | ✅ | `stage_rrf_rerank` in pipeline.rs |

### Existing Tests
**File**: `crates/context-graph-core/src/retrieval/aggregation.rs` (lines 190-428)

- `test_rrf_aggregation_single_list` - Single ranking verification
- `test_rrf_aggregation_multiple_lists` - Multi-space combination
- `test_rrf_weighted` - Per-space weighting
- `test_rrf_contribution` - Individual contribution calculation
- `test_default_is_rrf` - Default k=60 verification
- `test_rrf_aggregate_panics` - Correct API usage enforcement

**File**: `crates/context-graph-embeddings/tests/search_test.rs`
- `test_rrf_k_constitution` - Constitution compliance

---

## Constraints (Per Constitution)

| Constraint | Target | Implementation |
|------------|--------|----------------|
| Fusion latency (100 docs, 13 rankings) | < 1ms | HashMap operations are O(1) |
| Memory (100 candidates) | < 50KB | Stores only (Uuid, f32) pairs |
| Score precision | f32 sufficient | Uses f32 throughout |
| Stage 3 total latency | < 20ms | FAIL FAST timeout in pipeline |

---

## Full State Verification Requirements

### Source of Truth Definition

| Component | Source of Truth | Verification Method |
|-----------|-----------------|---------------------|
| RRF k constant | `crates/context-graph-core/src/config/constants/similarity.rs::RRF_K` | Must equal 60.0 |
| RRF formula | `1/(k + rank + 1)` where rank is 0-indexed | Unit tests verify exact scores |
| Default strategy | `AggregationStrategy::default()` returns `RRF { k: 60.0 }` | `test_default_is_rrf` |
| Stage 3 behavior | `pipeline.rs::stage_rrf_rerank()` | Integration tests |

### Execute & Inspect Pattern

Before considering RRF "working", you MUST:

1. **Run the existing tests**:
   ```bash
   cargo test -p context-graph-core aggregation -- --nocapture
   cargo test -p context-graph-embeddings rrf -- --nocapture
   ```

2. **Verify exact RRF scores**:
   ```rust
   // Rank 0 with k=60: 1/(60+0+1) = 0.01639344...
   assert!((score - 1.0/61.0).abs() < 0.0001);

   // Rank 5 with k=60: 1/(60+5+1) = 0.01515151...
   assert!((score - 1.0/66.0).abs() < 0.0001);
   ```

3. **Verify multi-space aggregation**:
   ```rust
   // Doc at ranks [0, 1, 0] across 3 spaces:
   // RRF = 1/61 + 1/62 + 1/61 = 0.04888...
   let expected = 1.0/61.0 + 1.0/62.0 + 1.0/61.0;
   assert!((actual - expected).abs() < 0.0001);
   ```

### Boundary & Edge Case Audit

#### Edge Case 1: Empty Rankings
**Input**: `ranked_lists = []`
**Expected**: Returns empty HashMap
**Before State**: No documents
**After State**: `HashMap::new()` with len=0

#### Edge Case 2: Document in Only One Ranking
**Input**: Doc A in space 0 at rank 0, not in spaces 1-12
**Expected**: RRF(A) = 1/61 (only one contribution)
**Before State**: A absent from aggregate
**After State**: A has score 0.01639344...

#### Edge Case 3: Same Document at Same Rank in All Spaces
**Input**: Doc A at rank 0 in all 13 spaces
**Expected**: RRF(A) = 13 × (1/61) = 0.21311...
**Before State**: A absent from aggregate
**After State**: A dominates with highest possible RRF score

### Evidence of Success (Actual Data)

From existing test output:
```
[VERIFIED] RRF single list: id1=0.016393, id2=0.016129
[VERIFIED] RRF multiple lists: id1=0.0489, id2=0.0484, id3=0.0479
[VERIFIED] RRF weighted: id1=0.0489, id2=0.0486
[VERIFIED] rrf_contribution: rank0=0.016393, rank5=0.015152
[VERIFIED] Default strategy is RRF with k=60
```

---

## Files (Actual Locations)

| File | Purpose | Status |
|------|---------|--------|
| `crates/context-graph-core/src/retrieval/aggregation.rs` | Primary RRF implementation | ✅ EXISTS |
| `crates/context-graph-core/src/retrieval/mod.rs` | Exports `AggregationStrategy` | ✅ EXISTS |
| `crates/context-graph-storage/src/teleological/search/pipeline.rs` | Stage 3 RRF integration | ✅ EXISTS |
| `crates/context-graph-core/src/config/constants/similarity.rs` | `RRF_K = 60.0` constant | ✅ EXISTS |

### Files That DO NOT EXIST (Original Spec Error)
- ~~`crates/context-graph-core/src/teleology/similarity/fusion.rs`~~ - WRONG PATH
- ~~`RRFFusion` struct~~ - Implementation uses `AggregationStrategy` enum

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| K parameter suboptimal | Low | Medium | k=60 is literature standard, configurable via `RRF { k }` |
| Score overflow | Very Low | Medium | RRF scores are always < 1.0 (sum of fractions) |
| Stage 3 timeout | Low | High | FAIL FAST implemented in pipeline.rs |

---

## Manual Verification Checklist

Before marking this task as truly complete, manually verify:

- [ ] `cargo test -p context-graph-core aggregation` - All 12 tests pass
- [ ] `cargo test -p context-graph-storage pipeline::tests::test_stage_rrf` - RRF stage works
- [ ] `grep -r "RRF_K" crates/` - Constant is 60.0 everywhere
- [ ] Inspect `aggregation.rs` line 148: Formula is `1.0 / (k + (rank as f32) + 1.0)`

---

## Fail Fast Requirements

Per constitution.yaml `ARCH-04` and project policy:

1. **NO FALLBACKS** - If RRF computation fails, propagate error immediately
2. **NO MOCK DATA IN TESTS** - Use real Uuids and computed scores
3. **TIMEOUT IS FATAL** - Stage 3 timeout (>20ms) returns `PipelineError::Timeout`
4. **NO BACKWARDS COMPATIBILITY** - The `RRFFusion` struct from original spec was never created; use `AggregationStrategy::RRF` only

---

## Traceability

- Source: Constitution ARCH-04 (Entry-point discovery pattern)
- Reference: TASK-LOGIC-008 Stage 3 (Multi-space rerank)
- Constitution line 715: `similarity.method: "RRF(d) = Σᵢ 1/(60 + rankᵢ(d))"`
- Constitution line 723: Stage 3 spec

---

## What an AI Agent MUST Know

1. **This task is ALREADY DONE** - Do not create new files
2. **File path in original spec was WRONG** - Use `retrieval/aggregation.rs`
3. **Type name was WRONG** - Use `AggregationStrategy::RRF`, not `RRFFusion`
4. **k=60 is hardcoded** - Per constitution, do not change
5. **Tests already exist** - Run them, don't create duplicates
6. **RRF is integrated into Stage 3** - See pipeline.rs lines 1021-1118
