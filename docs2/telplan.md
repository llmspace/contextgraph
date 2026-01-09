# Teleological Implementation Plan

## Executive Summary

The teleological system implements a **13-embedder multi-array architecture** for rich semantic search. Each memory is represented as a `TeleologicalVector` containing outputs from all 13 specialized embedding models, enabling "super search" across multiple semantic dimensions simultaneously.

**Current Status**: ~70% Complete
- Core types and data structures: **100%**
- Services layer (TELEO-007 to TELEO-015): **100%**
- Matrix search algorithm: **100%**
- MCP tool definitions: **100%**
- MCP tool handlers: **0%** (definitions exist, handlers not wired)
- Storage layer: **80%** (TeleologicalFingerprint store exists, TeleologicalVector store missing)
- Embeddings pipeline integration: **60%** (multi_array provider exists, not connected to MCP)
- GWT/Kuramoto integration: **0%**

---

## Architecture Overview

### 13-Embedder Array Structure

| Index | Embedder | Dimension | Purpose | Status |
|-------|----------|-----------|---------|--------|
| E1 | Semantic (E5-Large) | 1024 | General meaning | Implemented |
| E2 | Temporal Recent | 256 | Time decay | Implemented |
| E3 | Temporal Periodic | 256 | Cyclical patterns | Implemented |
| E4 | Temporal Positional | 256 | Sequence position | Implemented |
| E5 | Causal (E5-Mistral) | 1024 | Cause-effect | Implemented |
| E6 | Sparse (SPLADE) | 30522 | Term expansion | Implemented |
| E7 | Code (CodeBERT) | 768 | Programming | Implemented |
| E8 | Graph (GraphSAGE) | 256 | Structural | Implemented |
| E9 | HDC | 512 | Hyperdimensional | Implemented |
| E10 | Multimodal (CLIP) | 768 | Cross-modal | Implemented |
| E11 | Entity (TransE) | 256 | Knowledge graph | Implemented |
| E12 | Late Interaction (ColBERT) | 128×N | Token-level | Implemented |
| E13 | SPLADE Inverted | Variable | Inverted index | Implemented |

### Key Data Structures

```
TeleologicalVector (crates/context-graph-core/src/teleological/vector.rs)
├── purpose_vector: [f32; 13]           // Per-embedder alignment scores
├── cross_correlations: [f32; 78]       // 13×12/2 unique embedder pairs
├── group_alignments: GroupAlignments   // 6D hierarchical aggregation
├── tucker_core: Option<TuckerCore>     // Compressed representation
└── confidence: f32                     // Quality score

TeleologicalFingerprint (crates/context-graph-core/src/types/fingerprint.rs)
├── semantic: SemanticFingerprint       // Primary semantic embedding
├── purpose: PurposeVector              // 13D alignment vector
├── johari: JohariFingerprint           // Quadrant classification
└── hash: ContentHash                   // Deduplication
```

### Group Hierarchy (6 Groups)

| Group | Embedders | Semantic Focus |
|-------|-----------|----------------|
| FACTUAL | E1, E12, E13 | What IS |
| TEMPORAL | E2, E3 | When/Sequence |
| CAUSAL | E4, E7 | Why/How |
| RELATIONAL | E5, E8, E9 | Like/Where/Who |
| QUALITATIVE | E10, E11 | Feel/Principle |
| IMPLEMENTATION | E6 | Code |

---

## Implementation Tasks

### Phase 1: MCP Tool Handlers (Priority: HIGH)

**Task TELEO-H1: Wire search_teleological handler**
- File: `crates/context-graph-mcp/src/handlers/tools.rs`
- Status: **NOT STARTED**
- Depends on: TeleologicalMatrixSearch (completed)

```rust
// Add match arm in handle_tools_call():
"search_teleological" => {
    // 1. Parse params: query_vector, strategy, scope, weights, top_k, threshold
    // 2. Create MatrixSearchConfig from params
    // 3. Instantiate TeleologicalMatrixSearch
    // 4. Call search() or search_with_breakdown()
    // 5. Return JSON results with similarity scores, breakdowns
}
```

Parameters:
- `query_vector`: TeleologicalVector (JSON) - required
- `candidates`: Vec<TeleologicalVector> (JSON) - required
- `strategy`: "cosine" | "euclidean" | "synergy_weighted" | "group_hierarchical" | "cross_correlation_dominant" | "tucker_compressed" | "adaptive"
- `scope`: "full" | "purpose_vector_only" | "cross_correlations_only" | "group_alignments_only" | object with specific_pairs/groups
- `weights`: ComponentWeights object (optional)
- `top_k`: number (default 10)
- `threshold`: number (default 0.0)
- `include_breakdown`: boolean (default false)

**Task TELEO-H2: Wire compute_teleological_vector handler**
- File: `crates/context-graph-mcp/src/handlers/tools.rs`
- Status: **NOT STARTED**
- Depends on: MultiArrayProvider (exists), FusionEngine (exists)

```rust
// Add match arm:
"compute_teleological_vector" => {
    // 1. Parse params: content, compute_tucker, profile_id
    // 2. Get/create MultiArrayProvider
    // 3. Call embed_all() to get all 13 embeddings
    // 4. Use CorrelationExtractor to compute cross-correlations
    // 5. Use GroupAggregator for group alignments
    // 6. Optionally apply TuckerDecomposer
    // 7. Return TeleologicalVector JSON
}
```

Parameters:
- `content`: string - required (text to embed)
- `compute_tucker`: boolean (default false)
- `profile_id`: string (optional, for weighted computation)

**Task TELEO-H3: Wire fuse_embeddings handler**
- File: `crates/context-graph-mcp/src/handlers/tools.rs`
- Status: **NOT STARTED**
- Depends on: FusionEngine (exists), SynergyService (exists)

```rust
"fuse_embeddings" => {
    // 1. Parse embeddings array (13 vectors)
    // 2. Get/create FusionEngine
    // 3. Optionally load profile
    // 4. Call fuse() or fuse_with_profile()
    // 5. Return FusionResult JSON
}
```

Parameters:
- `embeddings`: array of 13 embedding vectors - required
- `profile_id`: string (optional)
- `synergy_matrix`: SynergyMatrix JSON (optional, uses default if omitted)
- `fusion_method`: "weighted_average" | "attention" | "tucker"

**Task TELEO-H4: Wire update_synergy_matrix handler**
- File: `crates/context-graph-mcp/src/handlers/tools.rs`
- Status: **NOT STARTED**
- Depends on: SynergyService, FeedbackLearner (both exist)

```rust
"update_synergy_matrix" => {
    // 1. Parse feedback event (vector_id, feedback_type, contributions)
    // 2. Get SynergyService from state
    // 3. Create FeedbackEvent
    // 4. Call FeedbackLearner.record_feedback()
    // 5. If should_learn(), call learn() and apply_gradient()
    // 6. Return updated synergy weights
}
```

Parameters:
- `vector_id`: UUID string - required
- `feedback_type`: "positive_retrieval" | "negative_retrieval" | "user_accept" | "user_reject" | "time_spent"
- `contributions`: array of 13 floats (optional)
- `context`: string (optional)

**Task TELEO-H5: Wire manage_teleological_profile handler**
- File: `crates/context-graph-mcp/src/handlers/tools.rs`
- Status: **NOT STARTED**
- Depends on: ProfileManager (exists)

```rust
"manage_teleological_profile" => {
    // Parse action and route to ProfileManager
    match action {
        "create" => profile_manager.create_profile(id, weights),
        "get" => profile_manager.get_profile(id),
        "update" => profile_manager.update_profile(id, weights),
        "delete" => profile_manager.delete_profile(id),
        "list" => profile_manager.list_profiles(),
        "find_best" => profile_manager.find_best_match(context),
    }
}
```

Parameters:
- `action`: "create" | "get" | "update" | "delete" | "list" | "find_best" - required
- `profile_id`: string (required for create/get/update/delete)
- `weights`: array of 13 floats (required for create/update)
- `context`: string (required for find_best)

---

### Phase 2: Storage Layer Completion (Priority: HIGH)

**Task TELEO-S1: Implement TeleologicalVector storage**
- Files:
  - `crates/context-graph-storage/src/teleological/rocksdb_store.rs`
  - `crates/context-graph-storage/src/teleological/serialization.rs`
- Status: **PARTIALLY COMPLETE** (CF exists, methods missing)

Column family `CF_TELEOLOGICAL_VECTORS` exists but needs:
```rust
impl RocksDbTeleologicalStore {
    pub async fn store_teleological_vector(&self, memory_id: Uuid, vector: &TeleologicalVector) -> Result<()>;
    pub async fn retrieve_teleological_vector(&self, memory_id: Uuid) -> Result<Option<TeleologicalVector>>;
    pub async fn search_by_teleological_vector(&self, query: &TeleologicalVector, k: usize) -> Result<Vec<(Uuid, f32)>>;
}
```

Serialization functions needed:
```rust
pub fn serialize_teleological_vector(vector: &TeleologicalVector) -> Vec<u8>;
pub fn deserialize_teleological_vector(data: &[u8]) -> TeleologicalVector;
```

**Task TELEO-S2: Implement SynergyMatrix persistence**
- File: `crates/context-graph-storage/src/teleological/rocksdb_store.rs`
- Status: **NOT STARTED**

Column family `CF_SYNERGY_MATRIX` exists but needs:
```rust
impl RocksDbTeleologicalStore {
    pub fn store_synergy_matrix(&self, matrix: &SynergyMatrix) -> Result<()>;
    pub fn load_synergy_matrix(&self) -> Result<Option<SynergyMatrix>>;
}
```

Uses singleton key `SYNERGY_MATRIX_KEY = b"synergy"`.

**Task TELEO-S3: Implement TeleologicalProfile persistence**
- File: `crates/context-graph-storage/src/teleological/rocksdb_store.rs`
- Status: **NOT STARTED**

Column family `CF_TELEOLOGICAL_PROFILES` exists but needs:
```rust
impl RocksDbTeleologicalStore {
    pub fn store_profile(&self, profile: &TeleologicalProfile) -> Result<()>;
    pub fn load_profile(&self, id: &ProfileId) -> Result<Option<TeleologicalProfile>>;
    pub fn list_profiles(&self) -> Result<Vec<ProfileId>>;
    pub fn delete_profile(&self, id: &ProfileId) -> Result<bool>;
}
```

---

### Phase 3: Embeddings Pipeline Integration (Priority: MEDIUM)

**Task TELEO-E1: Connect MultiArrayProvider to MCP handlers**
- File: `crates/context-graph-mcp/src/handlers/mod.rs`
- Status: **NOT STARTED**

The `MultiArrayProvider` in `crates/context-graph-embeddings/src/provider/multi_array.rs` computes all 13 embeddings. Need to:
1. Add `multi_array_provider: Arc<MultiArrayProvider>` to Handlers state
2. Initialize in `Handlers::new()` or lazy-load on first use
3. Use in `compute_teleological_vector` handler

**Task TELEO-E2: Implement TeleologicalVectorBuilder**
- File: `crates/context-graph-core/src/teleological/vector.rs` (new builder)
- Status: **NOT STARTED**

Create builder that takes MultiArrayProvider output and constructs TeleologicalVector:
```rust
pub struct TeleologicalVectorBuilder {
    correlation_extractor: CorrelationExtractor,
    group_aggregator: GroupAggregator,
    tucker_decomposer: Option<TuckerDecomposer>,
}

impl TeleologicalVectorBuilder {
    pub fn from_embeddings(embeddings: &[Vec<f32>; 13]) -> Result<TeleologicalVector>;
    pub fn from_multi_array_result(result: MultiArrayResult) -> Result<TeleologicalVector>;
}
```

**Task TELEO-E3: GPU acceleration for batch embedding**
- File: `crates/context-graph-embeddings/src/provider/multi_array.rs`
- Status: **PARTIALLY COMPLETE** (individual models support GPU, batch orchestration needed)

Add batch processing:
```rust
impl MultiArrayProvider {
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<MultiArrayResult>>;
    pub async fn embed_parallel(&self, texts: &[&str], max_concurrent: usize) -> Result<Vec<MultiArrayResult>>;
}
```

---

### Phase 4: GWT/Kuramoto Integration (Priority: MEDIUM)

**Task TELEO-K1: Connect Kuramoto phases to embedding coherence**
- Files:
  - `crates/context-graph-utl/src/phase/oscillator/kuramoto.rs`
  - `crates/context-graph-mcp/src/handlers/gwt.rs`
- Status: **NOT STARTED**

The Kuramoto oscillator network should derive phase relationships from embedding correlations:
```rust
// In SynergyService or new KuramotoEmbeddingBridge:
pub fn compute_phase_from_correlations(cross_correlations: &[f32; 78]) -> [f64; 13];
pub fn update_kuramoto_from_teleological(vector: &TeleologicalVector);
```

**Task TELEO-K2: Feed GWT workspace with TeleologicalVector data**
- File: `crates/context-graph-mcp/src/handlers/gwt.rs`
- Status: **NOT STARTED**

When memories enter Global Workspace:
1. Compute TeleologicalVector for competing memories
2. Use group alignments to determine broadcast priority
3. Update Kuramoto coupling based on active embedding coherence

---

### Phase 5: Advanced Features (Priority: LOW)

**Task TELEO-A1: HNSW indexing for TeleologicalVector search**
- File: `crates/context-graph-storage/src/teleological/indexes/`
- Status: **INFRASTRUCTURE EXISTS** (config files present)

Implement HNSW index for fast approximate nearest neighbor search on:
- Full purpose_vector (13D)
- Group alignments (6D)
- Tucker-compressed vectors

**Task TELEO-A2: Adaptive synergy matrix evolution**
- File: `crates/context-graph-core/src/teleological/services/feedback_learner.rs`
- Status: **IMPLEMENTED** but not connected to persistence

Add:
1. Periodic checkpoint of synergy matrix to storage
2. Rollback mechanism for bad updates
3. A/B testing framework for synergy configurations

**Task TELEO-A3: Multi-resolution search cascade**
- File: `crates/context-graph-core/src/teleological/matrix_search.rs`
- Status: **PARTIALLY IMPLEMENTED** (resolution hierarchy exists)

Implement search cascade:
1. Quick filter using 1D purpose alignment score
2. Candidate expansion using 6D group alignments
3. Full scoring using 13D purpose vector + correlations

**Task TELEO-A4: Cross-modal teleological search**
- Status: **NOT STARTED**

Enable search across:
- Text → Image (via E10 multimodal)
- Code → Documentation (via E7 code)
- Entity → Entity (via E11 entity)

---

## File Locations Summary

### Core Types (`crates/context-graph-core/src/teleological/`)
| File | Contents | Status |
|------|----------|--------|
| `mod.rs` | Module exports | Complete |
| `types.rs` | ProfileId, TuckerCore, constants | Complete |
| `vector.rs` | TeleologicalVector | Complete |
| `synergy_matrix.rs` | SynergyMatrix | Complete |
| `groups.rs` | GroupAlignments, GroupType | Complete |
| `resolution.rs` | MultiResolutionHierarchy | Complete |
| `profile.rs` | TeleologicalProfile, TaskType | Complete |
| `meaning.rs` | MeaningExtractionConfig | Complete |
| `matrix_search.rs` | TeleologicalMatrixSearch | Complete |

### Services (`crates/context-graph-core/src/teleological/services/`)
| File | Service | Status |
|------|---------|--------|
| `synergy_service.rs` | SynergyService (007) | Complete |
| `correlation_extractor.rs` | CorrelationExtractor (008) | Complete |
| `meaning_pipeline.rs` | MeaningPipeline (009) | Complete |
| `tucker_decomposer.rs` | TuckerDecomposer (010) | Complete |
| `group_aggregator.rs` | GroupAggregator (011) | Complete |
| `fusion_engine.rs` | FusionEngine (012) | Complete |
| `multi_space_retriever.rs` | MultiSpaceRetriever (013) | Complete |
| `feedback_learner.rs` | FeedbackLearner (014) | Complete |
| `profile_manager.rs` | ProfileManager (015) | Complete |

### Storage (`crates/context-graph-storage/src/teleological/`)
| File | Contents | Status |
|------|----------|--------|
| `column_families.rs` | CF definitions | Complete |
| `schema.rs` | Key format functions | Complete |
| `rocksdb_store.rs` | RocksDbTeleologicalStore | 80% (missing TV, SM, Profile methods) |
| `serialization.rs` | Fingerprint serialization | Complete |
| `indexes/` | HNSW config | Infrastructure only |

### MCP (`crates/context-graph-mcp/src/`)
| File | Contents | Status |
|------|----------|--------|
| `tools.rs` | Tool definitions (34 total) | Complete |
| `handlers/tools.rs` | Tool call dispatch | 0% for teleological |

### Embeddings (`crates/context-graph-embeddings/src/`)
| File | Contents | Status |
|------|----------|--------|
| `provider/multi_array.rs` | 13-embedder orchestration | Complete |
| `models/pretrained/` | Individual embedding models | Complete |

---

## Test Coverage

### Existing Tests
- `crates/context-graph-core/src/teleological/matrix_search.rs`: 18 tests (all passing)
- `crates/context-graph-storage/src/teleological/tests.rs`: Integration tests
- `crates/context-graph-storage/tests/teleological_integration.rs`: E2E tests
- `crates/context-graph-mcp/src/handlers/tests/tools_list.rs`: Tool count verification (34)

### Tests Needed
1. **TELEO-T1**: Handler tests for each teleological tool
2. **TELEO-T2**: Storage round-trip for TeleologicalVector
3. **TELEO-T3**: Storage round-trip for SynergyMatrix
4. **TELEO-T4**: Storage round-trip for TeleologicalProfile
5. **TELEO-T5**: End-to-end: content → embeddings → vector → storage → search
6. **TELEO-T6**: Feedback loop: search → feedback → synergy update → improved search

---

## Compiler Warnings to Fix

```
crates/context-graph-core/src/teleological/services/meaning_pipeline.rs:21
  unused import: `CROSS_CORRELATION_COUNT`

crates/context-graph-core/src/teleological/services/group_aggregator.rs:17
  unused import: `groups::group_indices`

crates/context-graph-core/src/teleological/services/fusion_engine.rs:12,15
  unused imports: `CROSS_CORRELATION_COUNT`, `GroupAlignments`, `SynergyMatrix`

crates/context-graph-core/src/teleological/services/multi_space_retriever.rs:12-14
  unused imports: `CROSS_CORRELATION_COUNT`, `GroupAlignments`, `types::NUM_EMBEDDERS`

crates/context-graph-mcp/src/tools.rs:1080-1088
  unused constants: SEARCH_TELEOLOGICAL, COMPUTE_TELEOLOGICAL_VECTOR,
                    FUSE_EMBEDDINGS, UPDATE_SYNERGY_MATRIX, MANAGE_TELEOLOGICAL_PROFILE
  (Will be used when handlers are wired)
```

---

## Implementation Order

### Sprint 1: Core MCP Integration (Est. 2-3 days)
1. TELEO-H2: compute_teleological_vector handler
2. TELEO-E1: Connect MultiArrayProvider
3. TELEO-H1: search_teleological handler
4. TELEO-T5: E2E test

### Sprint 2: Storage & Persistence (Est. 2 days)
1. TELEO-S1: TeleologicalVector storage
2. TELEO-S2: SynergyMatrix persistence
3. TELEO-S3: TeleologicalProfile persistence
4. TELEO-T2, T3, T4: Storage tests

### Sprint 3: Feedback Loop (Est. 1-2 days)
1. TELEO-H4: update_synergy_matrix handler
2. TELEO-H5: manage_teleological_profile handler
3. TELEO-H3: fuse_embeddings handler
4. TELEO-T6: Feedback loop test

### Sprint 4: GWT Integration (Est. 2-3 days)
1. TELEO-K1: Kuramoto-embedding bridge
2. TELEO-K2: GWT workspace integration
3. Integration tests

### Sprint 5: Advanced Features (Est. 3-5 days)
1. TELEO-A1: HNSW indexing
2. TELEO-A3: Multi-resolution cascade
3. Performance optimization

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Tool definitions | 5/5 | 5/5 |
| Tool handlers implemented | 5/5 | 0/5 |
| Storage methods | 9/9 | 3/9 |
| Unit test coverage | >80% | ~60% |
| E2E test coverage | 3 scenarios | 0 |
| Search latency (p50) | <50ms | N/A |
| Search latency (p99) | <200ms | N/A |
| Synergy update latency | <10ms | N/A |

---

## Dependencies

### Internal
- `context-graph-core`: Types, services
- `context-graph-embeddings`: MultiArrayProvider, embedding models
- `context-graph-storage`: RocksDB persistence
- `context-graph-utl`: Kuramoto oscillators

### External (already in Cargo.toml)
- `rocksdb`: Storage backend
- `serde_json`: Serialization
- `uuid`: ID generation
- `tracing`: Logging
- `tokio`: Async runtime

---

## Notes

1. **No new crate dependencies required** - all infrastructure exists
2. **Column families already created** - storage schema is defined
3. **Services are stateless** - can be instantiated per-request or shared
4. **GPU models warm-loaded** - embedding computation is fast once initialized
5. **Matrix search is CPU-only** - consider SIMD optimization for hot path
