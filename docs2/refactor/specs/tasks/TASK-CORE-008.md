# TASK-CORE-008: RocksDB Per-Embedder Index Integration

**STATUS: ✅ COMPLETED (2026-01-09)**

```xml
<task_spec id="TASK-CORE-008" version="4.0">
<metadata>
  <title>Integrate EmbedderIndexRegistry into RocksDbTeleologicalStore</title>
  <status>COMPLETED</status>
  <layer>foundation</layer>
  <sequence>8</sequence>
  <implements>
    <requirement_ref>ARCH-01: TeleologicalArray is atomic storage unit</requirement_ref>
    <requirement_ref>ARCH-02: Compare only compatible embedding types (apples-to-apples)</requirement_ref>
    <requirement_ref>ARCH-04: Entry-point discovery for retrieval</requirement_ref>
  </implements>
  <depends_on>
    <task_ref status="COMPLETED">TASK-CORE-007: Per-Embedder Index Structure</task_ref>
    <task_ref status="COMPLETED">TASK-CORE-004: ComparisonValidationError</task_ref>
    <task_ref status="COMPLETED">TASK-CORE-002: Embedder enum</task_ref>
  </depends_on>
  <completed_date>2026-01-09</completed_date>
</metadata>
</task_spec>
```

## Completion Summary

This task has been **fully implemented**. The `HnswMultiSpaceIndex` has been replaced with `EmbedderIndexRegistry` throughout `RocksDbTeleologicalStore`.

### Changes Made

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store.rs`

| Change | Description |
|--------|-------------|
| Struct field | `hnsw_index` → `index_registry: Arc<EmbedderIndexRegistry>` (line 234) |
| Imports | Added `EmbedderIndex, EmbedderIndexOps, EmbedderIndexRegistry, IndexError` (lines 42-44) |
| Constructor | Creates `EmbedderIndexRegistry::new()` with 12 HNSW indexes (line 313) |
| `add_to_indexes()` | Inserts vectors into all 12 HNSW-capable indexes |
| `remove_from_indexes()` | Removes from all indexes |
| `get_embedder_vector()` | Extracts vector for specific embedder from `SemanticFingerprint` |
| `compute_embedder_scores()` | Computes cosine similarity for all 13 embedders |
| `search_semantic()` | Uses entry-point pattern (E1Semantic first) |
| `search_purpose()` | Uses PurposeVector index directly |
| Removed | `initialize_hnsw()`, `is_hnsw_initialized()` |

### Verification Commands

```bash
# Verify compilation
cargo check -p context-graph-storage

# Run all storage tests
cargo test -p context-graph-storage

# Verify old code removed (expect no matches)
grep -r "HnswMultiSpaceIndex" crates/context-graph-storage/src/teleological/

# Verify new code present
grep -n "index_registry" crates/context-graph-storage/src/teleological/rocksdb_store.rs
```

## Architecture After Completion

```
RocksDbTeleologicalStore
├── db: Arc<DB>                          # RocksDB instance
├── cache: Cache                         # Block cache
├── path: PathBuf                        # DB path
├── fingerprint_count: RwLock            # Cached count
├── soft_deleted: RwLock                 # Soft delete tracking
└── index_registry: Arc<EmbedderIndexRegistry>
    └── indexes: HashMap<EmbedderIndex, Arc<HnswEmbedderIndex>>
        ├── E1Semantic (1024D)
        ├── E1Matryoshka128 (128D)
        ├── E2TemporalRecent (512D)
        ├── E3TemporalPeriodic (512D)
        ├── E4TemporalPositional (512D)
        ├── E5Causal (768D)
        ├── E7Code (1536D)
        ├── E8Graph (384D)
        ├── E9HDC (1024D)
        ├── E10Multimodal (768D)
        ├── E11Entity (384D)
        └── PurposeVector (13D)

NOT in EmbedderIndexRegistry (use different index types):
- E6Sparse → InvertedIndex (existing CF)
- E12LateInteraction → MaxSim (future)
- E13Splade → InvertedIndex (existing CF)
```

## Key File Locations

| Item | Path |
|------|------|
| Main store | `crates/context-graph-storage/src/teleological/rocksdb_store.rs` |
| Registry | `crates/context-graph-storage/src/teleological/indexes/registry.rs` |
| Index trait | `crates/context-graph-storage/src/teleological/indexes/embedder_index.rs` |
| HNSW impl | `crates/context-graph-storage/src/teleological/indexes/hnsw_impl.rs` |
| Config constants | `crates/context-graph-storage/src/teleological/indexes/hnsw_config/constants.rs` |

## FAIL FAST Guarantees

| Scenario | Response |
|----------|----------|
| Store with missing embeddings | `CoreError::ValidationError` |
| Search with wrong dimension | `IndexError::DimensionMismatch` |
| E6/E12/E13 passed to HNSW | `panic!` with clear message |
| RocksDB write failure | `TeleologicalStoreError::RocksDbOperation` |
| Index not found | `CoreError::IndexError` |
| NaN/Inf in vector | `IndexError::InvalidVector` |

## Remaining Work (Future Tasks)

| Item | Future Task |
|------|-------------|
| Index Persistence | HNSW indexes are in-memory only; future task should persist to RocksDB |
| InvertedIndex for E6/E13 | TASK-LOGIC-002 (Sparse Similarity Functions) |
| MaxSim for E12 | TASK-LOGIC-003 (Token-Level Similarity) |
| 5-Stage Pipeline | TASK-LOGIC-008 (5-Stage Pipeline) |
| RRF Fusion | TASK-LOGIC-011 (RRF Fusion) |

## NO BACKWARDS COMPATIBILITY

- ❌ Do NOT keep `HnswMultiSpaceIndex` as fallback
- ❌ Do NOT add deprecated method aliases
- ❌ Do NOT silently convert between old/new structures
- ❌ Old `hnsw_index` field references MUST fail to compile
- ❌ Tests MUST use real vectors, NOT mocks
