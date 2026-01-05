# TASK-F008: TeleologicalMemoryStore Trait

**Status**: SPEC READY
**Priority**: P0 (Critical Path)
**Dependencies**: TASK-F001 (COMPLETE), TASK-F002 (COMPLETE), TASK-F003 (COMPLETE), TASK-F004 (COMPLETE), TASK-F005 (COMPLETE), TASK-F007 (COMPLETE)
**Estimated Effort**: L (Large)
**Last Updated**: 2026-01-05

---

## CRITICAL DIRECTIVES (READ FIRST)

1. **NO BACKWARDS COMPATIBILITY**: Delete old code completely. No fallbacks, no `_deprecated` wrappers, no optional feature flags for legacy behavior.
2. **FAIL FAST**: All errors must propagate immediately with full context. No silent swallowing of errors. Use `CoreError` variants.
3. **NO MOCK DATA IN TESTS**: Tests use real `InMemoryTeleologicalStore` (not mocks). Tests must verify actual storage operations by reading back data.
4. **SHERLOCK-HOLMES VERIFICATION REQUIRED**: Final step MUST spawn sherlock-holmes agent to audit implementation completeness.

---

## Executive Summary

Implement `TeleologicalMemoryStore` trait providing CRUD and multi-array search for `TeleologicalFingerprint` with 13 embeddings. This trait **REPLACES** the legacy `MemoryStore` trait (which operates on `MemoryNode` with single embedding).

**Key Change**: The system now uses 13 embeddings (E1-E13) instead of 12. The old task spec incorrectly stated 12 embeddings.

---

## Current Codebase State (VERIFIED 2026-01-05)

### Existing Types (DO NOT MODIFY - CONSUME AS-IS)

| Type | File Path | Key Info |
|------|-----------|----------|
| `TeleologicalFingerprint` | `crates/context-graph-core/src/types/fingerprint/teleological/types.rs` | Contains `semantic`, `purpose_vector`, `johari`, `purpose_evolution`, `theta_to_north_star`, `content_hash`, timestamps, `access_count` |
| `SemanticFingerprint` | `crates/context-graph-core/src/types/fingerprint/semantic/fingerprint.rs` | 13 embeddings: `e1_semantic` through `e13_splade`. Uses `Vec<f32>` for dense, `SparseVector` for E6/E13 |
| `PurposeVector` | `crates/context-graph-core/src/types/fingerprint/purpose.rs` | `alignments: [f32; 13]`, `dominant_embedder: u8`, `coherence: f32`, `stability: f32`. Has `similarity()` method |
| `JohariFingerprint` | `crates/context-graph-core/src/types/fingerprint/johari/core.rs` | `quadrants: [[f32; 4]; 13]`, `confidence: [f32; 13]`, `transition_probs: [[[f32; 4]; 4]; 13]` |
| `JohariQuadrant` | `crates/context-graph-core/src/types/johari/quadrant.rs` | Enum: `Open`, `Hidden`, `Blind`, `Unknown` |
| `SparseVector` | `crates/context-graph-core/src/types/fingerprint/sparse.rs` | `indices: Vec<u16>`, `values: Vec<f32>` |
| `CoreError` | `crates/context-graph-core/src/error.rs` | Variants: `StorageError`, `ValidationError`, `SerializationError`, `FeatureDisabled`, etc. |
| `NUM_EMBEDDERS` | `crates/context-graph-core/src/types/fingerprint/semantic/constants.rs` | `pub const NUM_EMBEDDERS: usize = 13;` |

### Existing Storage Infrastructure (TASK-F004 COMPLETE)

| File | Contents |
|------|----------|
| `crates/context-graph-storage/src/teleological/column_families.rs` | Constants: `CF_FINGERPRINTS`, `CF_PURPOSE_VECTORS`, `CF_E13_SPLADE_INVERTED`, `CF_E1_MATRYOSHKA_128` |
| `crates/context-graph-storage/src/teleological/schema.rs` | Functions: `fingerprint_key(id)`, `purpose_vector_key(id)`, `e13_splade_inverted_key(term_id)`, `e1_matryoshka_128_key(id)` |

### File to DELETE

```
crates/context-graph-core/src/traits/memory_store.rs
```

This file contains the old `MemoryStore` trait operating on `MemoryNode`. **DELETE ENTIRE FILE**.

### Current traits/mod.rs (WILL BE REPLACED)

```rust
// Current content - DO NOT preserve memory_store exports
mod memory_store;  // DELETE THIS MODULE
pub use memory_store::{MemoryStore, SearchOptions, StorageBackend, StorageConfig};  // DELETE
```

---

## Deliverables

### 1. New Trait File

**File**: `crates/context-graph-core/src/traits/teleological_memory_store.rs` (CREATE NEW)

```rust
//! TeleologicalMemoryStore trait for fingerprint persistence.
//!
//! Provides CRUD operations and multi-array search for TeleologicalFingerprint.
//! REPLACES legacy MemoryStore trait.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::error::CoreResult;
use crate::types::fingerprint::{
    PurposeVector, SemanticFingerprint, SparseVector, TeleologicalFingerprint, NUM_EMBEDDERS,
};
use crate::types::johari::JohariQuadrant;

/// Storage backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeleologicalStorageBackend {
    /// In-memory storage (testing only - ephemeral)
    InMemory,
    /// RocksDB with teleological column families (production)
    RocksDB,
}

/// Configuration for TeleologicalMemoryStore.
#[derive(Debug, Clone)]
pub struct TeleologicalStorageConfig {
    pub backend: TeleologicalStorageBackend,
    pub path: Option<PathBuf>,
    pub cache_size_mb: usize,
    pub enable_wal: bool,
}

impl TeleologicalStorageConfig {
    pub fn rocksdb(path: impl Into<PathBuf>) -> Self {
        Self {
            backend: TeleologicalStorageBackend::RocksDB,
            path: Some(path.into()),
            cache_size_mb: 256,
            enable_wal: true,
        }
    }

    pub fn in_memory() -> Self {
        Self {
            backend: TeleologicalStorageBackend::InMemory,
            path: None,
            cache_size_mb: 64,
            enable_wal: false,
        }
    }
}

/// Search filter for teleological queries.
#[derive(Debug, Clone, Default)]
pub struct TeleologicalSearchFilter {
    pub min_alignment: Option<f32>,
    pub johari_quadrant: Option<JohariQuadrant>,
    pub dominant_embedder: Option<u8>,
    pub include_deleted: bool,
}

/// Search options for teleological queries.
#[derive(Debug, Clone)]
pub struct TeleologicalSearchOptions {
    pub top_k: usize,
    pub min_similarity: f32,
    pub embedder_index: Option<usize>,
    pub use_5_stage_pipeline: bool,
    pub filter: TeleologicalSearchFilter,
}

impl Default for TeleologicalSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            embedder_index: None,
            use_5_stage_pipeline: true,
            filter: TeleologicalSearchFilter::default(),
        }
    }
}

impl TeleologicalSearchOptions {
    pub fn new(top_k: usize) -> Self {
        Self { top_k, ..Default::default() }
    }

    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold;
        self
    }

    pub fn with_embedder(mut self, idx: usize) -> Self {
        self.embedder_index = Some(idx);
        self
    }

    pub fn with_johari_filter(mut self, quadrant: JohariQuadrant) -> Self {
        self.filter.johari_quadrant = Some(quadrant);
        self
    }

    pub fn with_min_alignment(mut self, alignment: f32) -> Self {
        self.filter.min_alignment = Some(alignment);
        self
    }
}

/// Search result with multi-array scores.
#[derive(Debug, Clone)]
pub struct TeleologicalSearchResult {
    pub fingerprint: TeleologicalFingerprint,
    pub per_embedder_scores: [f32; NUM_EMBEDDERS],
    pub fused_score: f32,
    pub purpose_score: f32,
    pub final_score: f32,
    pub stage_scores: Option<[f32; 5]>,
}

/// Teleological Memory Store trait.
///
/// All methods return CoreError on failure (FAIL FAST).
#[async_trait]
pub trait TeleologicalMemoryStore: Send + Sync {
    // CRUD
    async fn store(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<Uuid>;
    async fn retrieve(&self, id: Uuid) -> CoreResult<Option<TeleologicalFingerprint>>;
    async fn update(&self, fingerprint: TeleologicalFingerprint) -> CoreResult<bool>;
    async fn delete(&self, id: Uuid, soft: bool) -> CoreResult<bool>;

    // Search
    async fn search_semantic(
        &self,
        query: &SemanticFingerprint,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    async fn search_purpose(
        &self,
        query: &PurposeVector,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    async fn search_text(
        &self,
        text: &str,
        options: TeleologicalSearchOptions,
    ) -> CoreResult<Vec<TeleologicalSearchResult>>;

    async fn search_sparse(
        &self,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> CoreResult<Vec<(Uuid, f32)>>;

    // Batch
    async fn store_batch(&self, fingerprints: Vec<TeleologicalFingerprint>) -> CoreResult<Vec<Uuid>>;
    async fn retrieve_batch(&self, ids: &[Uuid]) -> CoreResult<Vec<Option<TeleologicalFingerprint>>>;

    // Stats
    async fn count(&self) -> CoreResult<usize>;
    async fn count_by_quadrant(&self) -> CoreResult<[usize; 4]>;
    fn storage_size_bytes(&self) -> usize;
    fn backend_type(&self) -> TeleologicalStorageBackend;

    // Persistence
    async fn flush(&self) -> CoreResult<()>;
    async fn checkpoint(&self) -> CoreResult<PathBuf>;
    async fn restore(&self, checkpoint_path: &Path) -> CoreResult<()>;
    async fn compact(&self) -> CoreResult<()>;
}
```

### 2. Update Module Exports

**File**: `crates/context-graph-core/src/traits/mod.rs` (REPLACE ENTIRE FILE)

```rust
//! Core trait definitions for the Context Graph system.

mod embedding_provider;
mod graph_index;
mod multi_array_embedding;
mod nervous_layer;
mod teleological_memory_store;
mod utl_processor;

pub use embedding_provider::{EmbeddingOutput, EmbeddingProvider};
pub use graph_index::GraphIndex;
pub use multi_array_embedding::{
    MultiArrayEmbeddingOutput, MultiArrayEmbeddingProvider, SingleEmbedder, SparseEmbedder,
    TokenEmbedder,
};
pub use nervous_layer::NervousLayer;
pub use teleological_memory_store::{
    TeleologicalMemoryStore, TeleologicalSearchFilter, TeleologicalSearchOptions,
    TeleologicalSearchResult, TeleologicalStorageBackend, TeleologicalStorageConfig,
};
pub use utl_processor::UtlProcessor;
```

### 3. Stub Implementation

**File**: `crates/context-graph-core/src/stubs/teleological_store_stub.rs` (CREATE NEW)

Implement `InMemoryTeleologicalStore`:
- `fingerprints: RwLock<HashMap<Uuid, TeleologicalFingerprint>>`
- `deleted: RwLock<HashSet<Uuid>>`
- `search_semantic`: Compute cosine similarity for dense embeddings (E1-E5, E7-E11)
- `search_purpose`: Use `PurposeVector::similarity()` method
- `search_text`: Return `CoreError::FeatureDisabled` (needs embedding provider)
- `checkpoint`: Serialize to JSON temp file
- `restore`: Deserialize from JSON

### 4. Update Stubs Module

**File**: `crates/context-graph-core/src/stubs/mod.rs` (REPLACE ENTIRE FILE)

```rust
//! Stub implementations for development and testing.

mod embedding_stub;
mod graph_index;
mod layers;
mod multi_array_stub;
mod teleological_store_stub;
mod utl_stub;

pub use embedding_stub::StubEmbeddingProvider;
pub use graph_index::InMemoryGraphIndex;
pub use layers::{
    StubCoherenceLayer, StubLearningLayer, StubMemoryLayer, StubReflexLayer, StubSensingLayer,
};
pub use multi_array_stub::StubMultiArrayProvider;
pub use teleological_store_stub::InMemoryTeleologicalStore;
pub use utl_stub::StubUtlProcessor;
```

---

## Files Summary

| Action | File |
|--------|------|
| CREATE | `crates/context-graph-core/src/traits/teleological_memory_store.rs` |
| CREATE | `crates/context-graph-core/src/stubs/teleological_store_stub.rs` |
| REPLACE | `crates/context-graph-core/src/traits/mod.rs` |
| REPLACE | `crates/context-graph-core/src/stubs/mod.rs` |
| DELETE | `crates/context-graph-core/src/traits/memory_store.rs` |

---

## Test Requirements

**File**: `crates/context-graph-core/src/traits/teleological_memory_store_tests.rs` (CREATE NEW)

### Required Tests (ALL MUST PASS)

```rust
#[tokio::test] async fn test_store_and_retrieve()
#[tokio::test] async fn test_retrieve_nonexistent()
#[tokio::test] async fn test_update()
#[tokio::test] async fn test_update_nonexistent_returns_false()
#[tokio::test] async fn test_soft_delete()
#[tokio::test] async fn test_hard_delete()
#[tokio::test] async fn test_search_semantic()
#[tokio::test] async fn test_search_purpose()
#[tokio::test] async fn test_batch_store_and_retrieve()
#[tokio::test] async fn test_empty_store_count()
#[tokio::test] async fn test_search_empty_store()
#[tokio::test] async fn test_checkpoint_and_restore()
#[tokio::test] async fn test_backend_type()
#[tokio::test] async fn test_min_similarity_filter()
```

### Test Requirements

1. **Print State**: Every test must print `[BEFORE]` and `[AFTER]` state
2. **Verify Output**: Print `[VERIFIED]` when assertion passes
3. **Real Data**: Create `TeleologicalFingerprint` with actual embeddings
4. **No Mocks**: Use `InMemoryTeleologicalStore::new()` directly

### Test Helper Function

```rust
fn create_test_fingerprint(content_seed: u8) -> TeleologicalFingerprint {
    let id = Uuid::new_v4();
    let mut semantic = SemanticFingerprint::zeroed();
    let base = content_seed as f32 / 255.0;
    semantic.e1_semantic = vec![base; 1024];
    // ... fill other embeddings

    let alignments: [f32; NUM_EMBEDDERS] = core::array::from_fn(|i| {
        (base + i as f32 * 0.05).min(1.0)
    });
    let purpose_vector = PurposeVector::new(alignments);
    let johari = JohariFingerprint::zeroed();

    TeleologicalFingerprint {
        id,
        semantic,
        purpose_vector,
        johari,
        purpose_evolution: Vec::new(),
        theta_to_north_star: base,
        content_hash: [content_seed; 32],
        created_at: Utc::now(),
        last_updated: Utc::now(),
        access_count: 0,
    }
}
```

---

## Full State Verification Protocol

### Source of Truth Locations

| Data | Storage Location | How to Inspect |
|------|------------------|----------------|
| Fingerprints | `InMemoryTeleologicalStore.fingerprints` HashMap | `store.count().await` + `store.retrieve(id).await` |
| Soft-deleted | `InMemoryTeleologicalStore.deleted` HashSet | `store.retrieve(id).await` returns `None` |
| Checkpoints | `/tmp/teleological_checkpoint_*.bin` | `std::fs::read()` + JSON deserialize |

### Verification Commands (Run After Implementation)

```bash
# 1. Build and test with output
cd /home/cabdru/contextgraph
cargo test -p context-graph-core teleological_memory_store -- --nocapture 2>&1 | tee /tmp/f008.log

# 2. Count verification markers
grep -c "\[VERIFIED\]" /tmp/f008.log
# Expected: >= 14 (one per test)

# 3. Check for failures
grep -E "FAILED|panicked|error\[E" /tmp/f008.log
# Expected: 0 matches

# 4. Verify compilation
cargo check -p context-graph-core 2>&1 | grep "^error"
# Expected: 0 matches

# 5. Verify old file deleted
test -f crates/context-graph-core/src/traits/memory_store.rs && echo "ERROR: File exists" || echo "OK: File deleted"
# Expected: OK: File deleted

# 6. Verify dependent crate compiles
cargo check -p context-graph-storage 2>&1 | grep "^error"
# Expected: 0 matches
```

### Edge Case Boundary Tests

| Edge Case | Input | Expected | Test Function |
|-----------|-------|----------|---------------|
| Empty store | `count()` | `0` | `test_empty_store_count` |
| Non-existent ID | `retrieve(random_uuid)` | `None` | `test_retrieve_nonexistent` |
| Update non-existent | `update(new_fp)` | `false` | `test_update_nonexistent_returns_false` |
| Soft-deleted retrieve | `retrieve(soft_deleted_id)` | `None` | `test_soft_delete` |
| High similarity filter | `min_similarity=0.99` | 0-1 results | `test_min_similarity_filter` |
| Empty search | `search_semantic` on empty store | `[]` | `test_search_empty_store` |

---

## Sherlock-Holmes Final Audit (MANDATORY)

After completing implementation, spawn this agent:

```
Task("TASK-F008 Verification", "
Perform forensic audit of TASK-F008 TeleologicalMemoryStore implementation:

EVIDENCE COLLECTION:

1. FILE EXISTENCE CHECK:
   - MUST EXIST: crates/context-graph-core/src/traits/teleological_memory_store.rs
   - MUST EXIST: crates/context-graph-core/src/stubs/teleological_store_stub.rs
   - MUST NOT EXIST: crates/context-graph-core/src/traits/memory_store.rs
   Run: ls -la crates/context-graph-core/src/traits/*.rs
   Run: ls -la crates/context-graph-core/src/stubs/*.rs

2. COMPILATION TEST:
   Run: cargo check -p context-graph-core 2>&1
   Expected: No errors

3. UNIT TEST EXECUTION:
   Run: cargo test -p context-graph-core teleological_memory_store -- --nocapture
   Expected: All tests pass, [VERIFIED] markers present

4. EXPORT VERIFICATION:
   Run: grep 'TeleologicalMemoryStore' crates/context-graph-core/src/traits/mod.rs
   Expected: Export line present
   Run: grep 'InMemoryTeleologicalStore' crates/context-graph-core/src/stubs/mod.rs
   Expected: Export line present

5. NO LEGACY REFERENCES:
   Run: grep -r 'MemoryStore' crates/context-graph-core/src/traits/mod.rs
   Expected: Only TeleologicalMemoryStore, NOT old MemoryStore

6. DEPENDENT CRATE CHECK:
   Run: cargo check -p context-graph-storage
   Expected: No errors

Report findings with PASS/FAIL for each check. If any FAIL, list exact remediation steps.
", "sherlock-holmes")
```

---

## Success Criteria Checklist

- [ ] `teleological_memory_store.rs` created with full trait definition
- [ ] `teleological_store_stub.rs` created with `InMemoryTeleologicalStore`
- [ ] `memory_store.rs` DELETED
- [ ] `traits/mod.rs` exports new trait (not old MemoryStore)
- [ ] `stubs/mod.rs` exports `InMemoryTeleologicalStore`
- [ ] All 14+ tests pass with `--nocapture` showing `[VERIFIED]`
- [ ] `cargo check -p context-graph-core` succeeds
- [ ] `cargo check -p context-graph-storage` succeeds
- [ ] Sherlock-holmes audit reports all PASS

---

## References

| Task | Status | Provides |
|------|--------|----------|
| TASK-F001 | COMPLETE | `SemanticFingerprint` with 13 embeddings |
| TASK-F002 | COMPLETE | `TeleologicalFingerprint` struct |
| TASK-F003 | COMPLETE | `JohariFingerprint` with 13 embedders |
| TASK-F004 | COMPLETE | RocksDB column families |
| TASK-F005 | COMPLETE | HNSW index configuration |
| TASK-F007 | COMPLETE | `MultiArrayEmbeddingProvider` trait |

---

## Appendix: Key Type Structures

### TeleologicalFingerprint (from types.rs)

```rust
pub struct TeleologicalFingerprint {
    pub id: Uuid,
    pub semantic: SemanticFingerprint,           // 13 embeddings ~60KB
    pub purpose_vector: PurposeVector,           // 13D alignment 52 bytes
    pub johari: JohariFingerprint,               // 13x4 weights ~1KB
    pub purpose_evolution: Vec<PurposeSnapshot>, // Up to 100 snapshots
    pub theta_to_north_star: f32,
    pub content_hash: [u8; 32],
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub access_count: u64,
}
```

### SemanticFingerprint Embedding Dimensions

| Embedder | Field | Dimension |
|----------|-------|-----------|
| E1 | `e1_semantic` | 1024 |
| E2 | `e2_temporal_recent` | 512 |
| E3 | `e3_temporal_periodic` | 512 |
| E4 | `e4_temporal_positional` | 512 |
| E5 | `e5_causal` | 768 |
| E6 | `e6_sparse` | SparseVector (30522 vocab) |
| E7 | `e7_code` | 256 |
| E8 | `e8_graph` | 384 |
| E9 | `e9_hdc` | 10000 |
| E10 | `e10_multimodal` | 768 |
| E11 | `e11_entity` | 384 |
| E12 | `e12_late_interaction` | Vec<Vec<f32>> (128D per token) |
| E13 | `e13_splade` | SparseVector (30522 vocab) |
