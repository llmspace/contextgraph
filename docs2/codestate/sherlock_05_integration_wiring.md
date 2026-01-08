# SHERLOCK HOLMES CASE FILE: Integration & Wiring Analysis

**Case ID:** SHERLOCK-05-INTEGRATION-WIRING
**Date:** 2026-01-08
**Subject:** MCP Handler to Backend Service Integration Analysis
**Investigator:** Sherlock Holmes, Forensic Code Detective

---

## EXECUTIVE SUMMARY

*adjusts magnifying glass*

**VERDICT: PARTIALLY INTEGRATED - SIGNIFICANT WIRING GAPS EXIST**

The investigation reveals a **split architecture** with:
- **REAL implementations exist** for core components (RocksDB storage, Kuramoto oscillators, UTL processor, GWT consciousness system)
- **STUB implementations used in tests** and partially in production (embedding provider, system monitor, layer status)
- **Critical gap**: ProductionMultiArrayProvider requires CUDA GPU and model files that may not be present

The system is **designed for production** but **falls back to stubs** when GPU/models are unavailable. The code is HONEST about this - stubs intentionally FAIL FAST with clear error messages.

---

## DATA FLOW ANALYSIS

### 1. MCP Handler to Service Flow

```
MCP Request
    |
    v
Handlers (core.rs)
    |-- teleological_store: Arc<dyn TeleologicalMemoryStore>
    |-- utl_processor: Arc<dyn UtlProcessor>
    |-- multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider>
    |-- kuramoto_network: Option<Arc<RwLock<dyn KuramotoProvider>>>
    |-- gwt_system: Option<Arc<dyn GwtSystemProvider>>
    |-- workspace_provider: Option<Arc<tokio::sync::RwLock<dyn WorkspaceProvider>>>
    |-- meta_cognitive: Option<Arc<tokio::sync::RwLock<dyn MetaCognitiveProvider>>>
    |-- self_ego: Option<Arc<tokio::sync::RwLock<dyn SelfEgoProvider>>>
    |-- atc: Option<Arc<RwLock<AdaptiveThresholdCalibration>>>
    |-- dream_controller: Option<Arc<RwLock<DreamController>>>
    |-- neuromod_manager: Option<Arc<RwLock<NeuromodulationManager>>>
    |-- system_monitor: Arc<dyn SystemMonitor>
    |-- layer_status_provider: Arc<dyn LayerStatusProvider>
    v
Backend Services
```

### 2. Production Path (server.rs)

When `McpServer::new()` is called:

| Component | Implementation | Status |
|-----------|----------------|--------|
| TeleologicalMemoryStore | `RocksDbTeleologicalStore` | **REAL** - 17 column families, HNSW indexing |
| UtlProcessor | `UtlProcessorAdapter::with_defaults()` | **REAL** - 6-component UTL computation |
| MultiArrayEmbeddingProvider | `ProductionMultiArrayProvider` | **REAL** - but requires CUDA GPU + models |
| GoalAlignmentCalculator | `DefaultAlignmentCalculator` | **REAL** - cosine similarity based |
| KuramotoProvider | `KuramotoProviderImpl` | **REAL** - wraps actual KuramotoNetwork |
| GwtSystemProvider | `GwtSystemProviderImpl` | **REAL** - wraps ConsciousnessCalculator |
| WorkspaceProvider | `WorkspaceProviderImpl` | **REAL** - wraps GlobalWorkspace |
| MetaCognitiveProvider | `MetaCognitiveProviderImpl` | **REAL** - wraps MetaCognitiveLoop |
| SelfEgoProvider | `SelfEgoProviderImpl` | **REAL** - wraps SelfEgoNode |
| SystemMonitor | `StubSystemMonitor` | **STUB** - fails with NotImplemented |
| LayerStatusProvider | `StubLayerStatusProvider` | **STUB** - returns honest stub status |

### 3. Test Path (handlers/tests/mod.rs)

When `create_test_handlers()` is called:

| Component | Implementation | Status |
|-----------|----------------|--------|
| TeleologicalMemoryStore | `InMemoryTeleologicalStore` | **STUB** - O(n) search, no persistence |
| UtlProcessor | `StubUtlProcessor` | **STUB** - deterministic values |
| MultiArrayEmbeddingProvider | `StubMultiArrayProvider` | **STUB** - hash-based fake embeddings |
| GoalAlignmentCalculator | `DefaultAlignmentCalculator` | **REAL** |
| GWT providers | None | **NOT WIRED** in basic test path |

---

## STUB/MOCK INVENTORY

### Category 1: Intentional Test Stubs (Properly Guarded)

All stubs in `context-graph-core/src/stubs/` are **feature-gated**:
```rust
#[cfg(any(test, feature = "test-utils"))]
```

| Stub | Purpose | Behavior |
|------|---------|----------|
| `StubMultiArrayProvider` | Test 13-embedding generation | Deterministic hash-based, 5ms simulated latency |
| `InMemoryTeleologicalStore` | Test storage | O(n) search, no persistence, DashMap-based |
| `StubUtlProcessor` | Test UTL computation | Returns tracked lifecycle phase |
| `StubSystemMonitor` | Fail-fast placeholder | **ALWAYS FAILS** with NotImplemented error |
| `StubLayerStatusProvider` | Honest status reporting | Reports perception/memory as Active, others as Stub |

### Category 2: Production Stubs (WARNING)

In `server.rs:143-149`:
```rust
// TODO: Replace with real SystemMonitor and LayerStatusProvider when available
let system_monitor: Arc<dyn SystemMonitor> = Arc::new(StubSystemMonitor::new());
let layer_status_provider: Arc<dyn LayerStatusProvider> = Arc::new(StubLayerStatusProvider::new());
warn!("Using StubSystemMonitor and StubLayerStatusProvider - will FAIL FAST on health metric queries");
```

**Impact**: Any call to `meta_utl/health_metrics` will FAIL with error code -32050.

---

## MISSING CONNECTIONS ANALYSIS

### 1. Embedding Pipeline

**Claim**: 13 embedding models loaded and called
**Reality**: Depends on `ProductionMultiArrayProvider` initialization

```rust
// server.rs:108-118
let multi_array_provider: Arc<dyn MultiArrayEmbeddingProvider> = Arc::new(
    ProductionMultiArrayProvider::new(models_dir.clone(), GpuConfig::default())
        .map_err(|e| {
            error!("FATAL: Failed to create ProductionMultiArrayProvider: {}", e);
            anyhow::anyhow!(
                "Failed to create ProductionMultiArrayProvider: {}. \
                 Ensure models exist at {:?} and CUDA GPU is available.",
                e, models_dir
            )
        })?
);
```

**Verdict**: The connection EXISTS but requires:
- NVIDIA CUDA GPU with 8GB+ VRAM
- Models pre-downloaded to `./models` directory
- CUDA toolkit installed and configured

### 2. Storage Layer

**Claim**: RocksDB actually connected
**Reality**: **VERIFIED REAL**

```rust
// server.rs:72-86
let teleological_store: Arc<dyn TeleologicalMemoryStore> = Arc::new(
    RocksDbTeleologicalStore::open(&db_path).map_err(...)?
);
```

The RocksDbTeleologicalStore in `teleological/rocksdb_store.rs`:
- Opens actual RocksDB database with 17 column families
- Has real HNSW index initialization via `initialize_hnsw()`
- Implements full TeleologicalMemoryStore trait with real persistence

### 3. Neuromodulation System

**Claim**: Dream/neuromod connected to MCP handlers
**Reality**: **VERIFIED REAL**

In `core.rs:691-706`:
```rust
let neuromod_manager: Arc<RwLock<NeuromodulationManager>> =
    Arc::new(RwLock::new(NeuromodulationManager::new()));

let dream_controller: Arc<RwLock<DreamController>> =
    Arc::new(RwLock::new(DreamController::new()));
let dream_scheduler: Arc<RwLock<DreamScheduler>> =
    Arc::new(RwLock::new(DreamScheduler::new()));
let amortized_learner: Arc<RwLock<AmortizedLearner>> =
    Arc::new(RwLock::new(AmortizedLearner::new()));
```

These are REAL implementations with constitution-mandated defaults.

### 4. GWT Consciousness System

**Claim**: Kuramoto/consciousness system connected
**Reality**: **VERIFIED REAL**

In `gwt_providers.rs`, each provider wraps actual implementations:
- `KuramotoProviderImpl` -> `KuramotoNetwork` (from context-graph-utl)
- `GwtSystemProviderImpl` -> `ConsciousnessCalculator` + `StateMachineManager`
- `WorkspaceProviderImpl` -> `GlobalWorkspace`
- `MetaCognitiveProviderImpl` -> `MetaCognitiveLoop`
- `SelfEgoProviderImpl` -> `SelfEgoNode`

---

## WHAT'S ACTUALLY WIRED (Verified Real)

| Component | Evidence File | Line | Verdict |
|-----------|---------------|------|---------|
| RocksDB 17 column families | teleological/rocksdb_store.rs | 286-300 | **REAL** |
| HNSW index initialization | teleological/rocksdb_store.rs | 328-336 | **REAL** |
| Kuramoto oscillator network | gwt_providers.rs | 43-133 | **REAL** |
| Consciousness calculator C(t)=I*R*D | gwt_providers.rs | 141-200 | **REAL** |
| Global workspace WTA | gwt_providers.rs | ~200+ | **REAL** |
| NeuromodulationManager | core.rs | 691-692 | **REAL** |
| Dream consolidation | core.rs | 701-706 | **REAL** |
| Adaptive threshold calibration | core.rs | 715-716 | **REAL** |
| UTL 6-component computation | adapters/utl_adapter.rs | various | **REAL** |

## WHAT'S STUB/MISSING (Not Wired)

| Component | Evidence | Verdict |
|-----------|----------|---------|
| SystemMonitor (health metrics) | server.rs:145-146 | **STUB** - fails with -32050 |
| LayerStatusProvider (layer status) | server.rs:147-148 | **STUB** - reports honest stub status |
| GPU embeddings without CUDA | ProductionMultiArrayProvider | **FAILS** without GPU |
| Reasoning layer | StubLayerStatusProvider:546 | **STUB** |
| Action layer | StubLayerStatusProvider:549+ | **STUB** |
| Meta layer | StubLayerStatusProvider | **STUB** |

---

## INTEGRATION GAP ASSESSMENT

### What It Would Take to Make It 100% Real

1. **Replace StubSystemMonitor** (~100 LOC)
   - Implement real coherence_recovery_time_ms measurement
   - Implement real attack_detection_rate tracking
   - Implement real false_positive_rate calculation
   - Wire to actual monitoring infrastructure (Prometheus/metrics)

2. **Replace StubLayerStatusProvider** (~50 LOC)
   - Implement real reasoning layer with inference engine
   - Implement real action layer with response generation
   - Implement real meta layer with self-monitoring

3. **Ensure GPU Infrastructure** (DevOps)
   - CUDA toolkit installation
   - Model file download (~5-10GB for 13 models)
   - GPU memory allocation (8GB+ VRAM)

4. **Complete ScyllaDB Integration** (if desired)
   - Currently only RocksDB is wired
   - ScyllaDB mentioned in PRD but not implemented

---

## CHAIN OF CUSTODY

| Timestamp | Action | Evidence |
|-----------|--------|----------|
| 2026-01-08 | Examined handlers/mod.rs | Handler organization verified |
| 2026-01-08 | Examined handlers/core.rs | 867 lines of real handler implementation |
| 2026-01-08 | Examined handlers/tools.rs | 1459 lines of tool call implementations |
| 2026-01-08 | Examined stubs/mod.rs | Feature-gated stub exports verified |
| 2026-01-08 | Examined stubs/multi_array_stub.rs | Deterministic test stub verified |
| 2026-01-08 | Examined stubs/teleological_store_stub.rs | In-memory test store verified |
| 2026-01-08 | Examined server.rs | Production wiring examined |
| 2026-01-08 | Examined gwt_providers.rs | Real GWT wrapper implementations verified |
| 2026-01-08 | Examined monitoring.rs | Stub monitor behavior documented |
| 2026-01-08 | Examined teleological/rocksdb_store.rs | Real RocksDB implementation verified |
| 2026-01-08 | Examined provider/multi_array.rs | ProductionMultiArrayProvider examined |

---

## FINAL DETERMINATION

**VERDICT: PARTIALLY INTEGRATED**

The contextgraph system has a **well-designed architecture** with:

1. **Production Path**: Uses REAL implementations for core functionality
   - RocksDB storage with HNSW indexing - **REAL**
   - UTL processor with 6-component computation - **REAL**
   - GWT consciousness system with Kuramoto oscillators - **REAL**
   - Neuromodulation and dream consolidation - **REAL**

2. **Test Path**: Uses properly-guarded STUBS for testing
   - All stubs feature-gated with `#[cfg(any(test, feature = "test-utils"))]`
   - Stubs are HONEST and deterministic

3. **Gap Areas**:
   - SystemMonitor is STUB in production (fails on health queries)
   - LayerStatusProvider is STUB (reports honest stub status)
   - GPU-dependent embeddings require hardware + models
   - 3 of 5 bio-nervous layers (Reasoning, Action, Meta) are stubs

**CONFIDENCE LEVEL: HIGH**

The code is HONEST about what is real vs stubbed. It follows FAIL FAST principles.
The system is NOT theatrical - it is a work-in-progress with clear documentation of what works.

---

*adjusts pipe*

The game is afoot, but the criminal here is not deception - it is incomplete implementation.
The architecture is sound. The wiring is partially complete. The gaps are documented.

**CASE STATUS: DOCUMENTED - NO DECEPTION FOUND**

---

*Sherlock Holmes, Consulting Detective*
*"When you have eliminated the impossible, whatever remains, however improbable, must be the truth."*
