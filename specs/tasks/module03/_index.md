# Module 3: Embedding Pipeline - Task Index

```yaml
metadata:
  module_id: M03
  module_name: 12-Model Embedding Pipeline
  version: 2.4.0
  total_tasks: 68
  foundation_tasks: 16
  logic_tasks: 33
  surface_tasks: 19
  generated: 2026-01-01
  updated: 2026-01-01
  approach: inside-out-bottom-up
  changelog:
    - "v2.4.0: Added M03-L32 (HDC Vector Persistence), M03-L33 (SPLADE Projection)"
    - "v2.4.0: Added M03-S18 (Tracing & Instrumentation), M03-S19 (Alignment Verification)"
    - "v2.3.0: Added M03-S17 (Fail-Safe Async Orchestrator), M03-L29 (Tokenizer Cache)"
    - "v2.3.0: Added M03-L30 (Grouped GEMM for MoE), M03-L31 (FuseMoE Weight Registry)"
    - "v2.2.0: Added M03-L28 (Blackwell Quantization), M03-S15 (GDS), M03-S16 (Warm-up)"
    - "v2.2.0: Updated M03-F05 with aux_data for ColBERT token storage"
```

---

## Quick Reference

| Layer | Task Range | Count | Focus |
|-------|------------|-------|-------|
| Foundation | M03-F01 → M03-F16 | 16 | Types, traits, configuration |
| Logic | M03-L01 → M03-L33 | 33 | Models, batch, cache, fusion, security, neuromod, quantization, tokenization, GEMM, persistence |
| Surface | M03-S01 → M03-S19 | 19 | Pipeline, CUDA, GDS, warm-up, async orchestration, tracing, alignment, artifacts, tests |

---

## Execution Order (Topological Sort)

Execute tasks in this order to satisfy all dependencies:

### Phase 1: Foundation Layer (Week 1)

| Order | Task ID | Title | Dependencies | Est. Hours |
|-------|---------|-------|--------------|------------|
| 1 | M03-F01 | ModelId Enum Definition | - | 2 |
| 2 | M03-F06 | ModelInput Enum | - | 1.5 |
| 3 | M03-F07 | InputType Enum | - | 0.5 |
| 4 | M03-F08 | EmbeddingError Enum (Extended) | - | 2 |
| 5 | M03-F13 | BatchConfig Struct | - | 1 |
| 6 | M03-F14 | FusionConfig Struct | - | 1 |
| 7 | M03-F15 | CacheConfig and GpuConfig | - | 1.5 |
| 8 | M03-F02 | ModelDimensions Constants | M03-F01 | 1 |
| 9 | M03-F03 | ModelEmbedding Struct | M03-F01 | 1.5 |
| 10 | M03-F05 | FusedEmbedding Struct | M03-F01, M03-F02 | 2 |
| 11 | M03-F04 | ConcatenatedEmbedding Struct | M03-F01, M03-F03 | 1.5 |
| 12 | M03-F09 | EmbeddingModel Trait | M03-F01, M03-F03, M03-F06, M03-F08 | 2 |
| 13 | M03-F10 | ModelFactory Trait | M03-F01, M03-F09 | 1 |
| 14 | M03-F11 | EmbeddingConfig Root | M03-F01 | 2 |
| 15 | M03-F12 | ModelRegistryConfig | M03-F01 | 1.5 |
| 16 | M03-F16 | Module Structure/Exports | M03-F01→M03-F15 | 2 |

**Foundation Subtotal: 24 hours**

### Phase 2: Logic Layer - Models (Week 2)

| Order | Task ID | Title | Dependencies | Est. Hours |
|-------|---------|-------|--------------|------------|
| 17 | M03-L02 | MemoryTracker | M03-F15 | 2 |
| 18 | M03-L01 | ModelRegistry Core | M03-F09, M03-F11, M03-F12 | 4 |
| 19 | M03-L04 | Temporal-Recent (E2) | M03-F09 | 3 |
| 20 | M03-L05 | Temporal-Periodic (E3) | M03-F09 | 3 |
| 21 | M03-L06 | Temporal-Positional (E4) | M03-F09 | 2 |
| 22 | M03-L11 | HDC Custom (E9) | M03-F09 | 4 |
| 23 | M03-L03 | Semantic/E5-large (E1) | M03-F09, M03-L01 | 4 |
| 24 | M03-L07 | Causal/Longformer (E5) | M03-F09, M03-L01 | 4 |
| 25 | M03-L08 | Sparse/SPLADE (E6) | M03-F09, M03-L01 | 4 |
| 26 | M03-L09 | Code/CodeBERT (E7) | M03-F09, M03-L01 | 4 |
| 27 | M03-L10 | Graph/MiniLM (E8) | M03-F09, M03-L01 | 3 |
| 28 | M03-L12 | Multimodal/CLIP (E10) | M03-F09, M03-L01 | 4 |

**Phase 2 Subtotal: 41 hours**

### Phase 3: Logic Layer - Infrastructure (Week 3)

| Order | Task ID | Title | Dependencies | Est. Hours |
|-------|---------|-------|--------------|------------|
| 29 | M03-L13 | Entity/MiniLM (E11) | M03-F09, M03-L01 | 3 |
| 30 | M03-L14 | LateInteraction/ColBERT (E12) | M03-F09, M03-L01 | 4 |
| 31 | M03-L15 | ModelFactory Implementation | M03-L03→M03-L14 | 3 |
| 32 | M03-L16 | BatchQueue and Request Types | M03-F06, M03-F13 | 2 |
| 33 | M03-L18 | CacheEntry and CacheKey Types | M03-F05, M03-F15 | 1.5 |
| 34 | M03-L17 | BatchProcessor | M03-L01, M03-L16, M03-F13 | 5 |
| 35 | M03-L19 | CacheManager | M03-L18, M03-F15 | 4 |
| 36 | M03-L20 | GatingNetwork | M03-F02, M03-F14 | 3 |
| 37 | M03-L21 | Expert Networks | M03-F02, M03-F14 | 4 |
| 38 | M03-L22 | FuseMoE Router | M03-L20, M03-L21 | 4 |
| 39 | M03-L23 | FuseMoE Main Module | M03-L22, M03-F05 | 4 |
| 40 | M03-L24 | CAME-AB Bridge (Optional) | M03-L23 | 3 |
| 41 | M03-L25 | PII Scrubber Implementation | M03-F06, M03-F08 | 4 |
| 42 | M03-L26 | Neuromodulation Integration | M03-F14, M03-L20, M03-L23 | 5 |
| 43 | M03-L27 | EmbeddingPriors/Context Priming | M03-F01, M03-F14, M03-L20, M03-L23 | 4 |
| 44 | M03-L28 | Blackwell Quantization Kernels (FP8/FP4) | M03-L23, M03-S04 | 5 |
| 45 | M03-L29 | Shared Tokenizer Cache (TokenizationManager) | M03-F06, M03-F01, M03-L01 | 3 |
| 46 | M03-L30 | Grouped GEMM for MoE Expert Execution | M03-L21, M03-L22, M03-S04 | 5 |
| 47 | M03-L31 | FuseMoE Weight Registry and Initialization | M03-L23, M03-L21, M03-S13 | 4 |
| 48 | M03-L32 | HDC Base-Vector Persistence | M03-L11, M03-F01, M03-S13 | 3 |
| 49 | M03-L33 | Sparse Projection Matrix Management (SPLADE) | M03-L08, M03-L31, M03-S13 | 3 |

**Phase 3 Subtotal: 76.5 hours**

### Phase 4: Surface Layer (Week 4)

| Order | Task ID | Title | Dependencies | Est. Hours |
|-------|---------|-------|--------------|------------|
| 50 | M03-S08 | Configuration File Loading | M03-F11 | 2 |
| 51 | M03-S04 | CUDA Device Trait | M03-F16 | 3 |
| 52 | M03-S05 | GPU Memory Pool | M03-S04 | 3 |
| 53 | M03-S15 | GPU Direct Storage (GDS) Integration | M03-S04, M03-S05 | 6 |
| 54 | M03-S06 | CUDA Kernel Stubs | M03-S04 | 2 |
| 55 | M03-S13 | Model Artifact Manager | M03-F01, M03-F11, M03-F12, M03-L01 | 5 |
| 56 | M03-S14 | Green Context SM Partitioning | M03-S04, M03-S05 | 6 |
| 57 | M03-S01 | EmbeddingPipeline Core | M03-L01, M03-L17, M03-L19, M03-L23, M03-L25, M03-S13, M03-S15, M03-L29 | 5 |
| 58 | M03-S17 | Fail-Safe Async Orchestrator | M03-S01, M03-L17, M03-L23 | 4 |
| 59 | M03-S18 | Tracing & Async Instrumentation | M03-S01, M03-S17, M03-L17 | 4 |
| 60 | M03-S16 | Pipeline Warm-up & JIT Trigger | M03-S01, M03-S05, M03-L23 | 3 |
| 61 | M03-S02 | PipelineMetrics/HealthStatus | M03-S01 | 1.5 |
| 62 | M03-S03 | EmbeddingProvider Bridge | M03-S01, M03-S17 | 2 |
| 63 | M03-S07 | HotSwap Model Loading | M03-L01 | 3 |
| 64 | M03-S09 | Unit Tests Suite | M03-F01→M03-F16 | 4 |
| 65 | M03-S10 | Integration Tests | M03-S01, M03-S03, M03-S16, M03-S17 | 6 |
| 66 | M03-S19 | Semantic Alignment Verification | M03-S01, M03-L23, M03-S10 | 4 |
| 67 | M03-S11 | Benchmarks | M03-S01, M03-S16, M03-S17, M03-S18 | 4 |
| 68 | M03-S12 | Documentation/Examples | M03-S01 | 3 |

**Phase 4 Subtotal: 70.5 hours**

---

## Complete Dependency Graph (DAG)

```mermaid
graph TB
    subgraph Foundation["Foundation Layer (Week 1)"]
        F01[M03-F01: ModelId]
        F02[M03-F02: Dimensions]
        F03[M03-F03: ModelEmbedding]
        F04[M03-F04: ConcatenatedEmb]
        F05[M03-F05: FusedEmbedding]
        F06[M03-F06: ModelInput]
        F07[M03-F07: InputType]
        F08[M03-F08: EmbeddingError]
        F09[M03-F09: EmbeddingModel Trait]
        F10[M03-F10: ModelFactory Trait]
        F11[M03-F11: EmbeddingConfig]
        F12[M03-F12: RegistryConfig]
        F13[M03-F13: BatchConfig]
        F14[M03-F14: FusionConfig]
        F15[M03-F15: Cache/GpuConfig]
        F16[M03-F16: Module Exports]
    end

    subgraph Logic_Models["Logic Layer - Models (Week 2)"]
        L01[M03-L01: ModelRegistry]
        L02[M03-L02: MemoryTracker]
        L03[M03-L03: Semantic E5]
        L04[M03-L04: Temporal-Recent]
        L05[M03-L05: Temporal-Periodic]
        L06[M03-L06: Temporal-Position]
        L07[M03-L07: Causal Longformer]
        L08[M03-L08: Sparse SPLADE]
        L09[M03-L09: Code CodeBERT]
        L10[M03-L10: Graph MiniLM]
        L11[M03-L11: HDC Custom]
        L12[M03-L12: Multimodal CLIP]
    end

    subgraph Logic_Infra["Logic Layer - Infrastructure (Week 3)"]
        L13[M03-L13: Entity MiniLM]
        L14[M03-L14: LateInt ColBERT]
        L15[M03-L15: ModelFactory Impl]
        L16[M03-L16: BatchTypes]
        L17[M03-L17: BatchProcessor]
        L18[M03-L18: CacheTypes]
        L19[M03-L19: CacheManager]
        L20[M03-L20: GatingNetwork]
        L21[M03-L21: Expert Networks]
        L22[M03-L22: Router]
        L23[M03-L23: FuseMoE]
        L24[M03-L24: CAME-AB]
        L28[M03-L28: Blackwell Quant]
        L29[M03-L29: Tokenizer Cache]
        L30[M03-L30: Grouped GEMM]
        L31[M03-L31: Weight Registry]
        L32[M03-L32: HDC Persistence]
        L33[M03-L33: SPLADE Projection]
    end

    subgraph Surface["Surface Layer (Week 4)"]
        S01[M03-S01: EmbeddingPipeline]
        S02[M03-S02: Metrics/Health]
        S03[M03-S03: EmbeddingProvider]
        S04[M03-S04: CUDA Device]
        S05[M03-S05: GPU Memory Pool]
        S06[M03-S06: CUDA Kernels]
        S07[M03-S07: HotSwap]
        S08[M03-S08: Config Loader]
        S09[M03-S09: Unit Tests]
        S10[M03-S10: Integration Tests]
        S11[M03-S11: Benchmarks]
        S12[M03-S12: Documentation]
        S15[M03-S15: GPU Direct Storage]
        S16[M03-S16: Pipeline Warm-up]
        S17[M03-S17: Async Orchestrator]
        S18[M03-S18: Tracing]
        S19[M03-S19: Alignment Verification]
    end

    %% Foundation dependencies
    F01 --> F02
    F01 --> F03
    F01 --> F05
    F01 --> F09
    F01 --> F11
    F01 --> F12
    F02 --> F05
    F03 --> F04
    F01 --> F04
    F06 --> F09
    F08 --> F09
    F09 --> F10
    F01 --> F10

    %% Logic - Models dependencies
    F15 --> L02
    F09 --> L01
    F11 --> L01
    F12 --> L01
    F09 --> L04
    F09 --> L05
    F09 --> L06
    F09 --> L11
    L01 --> L03
    L01 --> L07
    L01 --> L08
    L01 --> L09
    L01 --> L10
    L01 --> L12
    L01 --> L13
    L01 --> L14

    %% Logic - Infrastructure dependencies
    L03 --> L15
    L04 --> L15
    L05 --> L15
    L06 --> L15
    L07 --> L15
    L08 --> L15
    L09 --> L15
    L10 --> L15
    L11 --> L15
    L12 --> L15
    L13 --> L15
    L14 --> L15
    F06 --> L16
    F13 --> L16
    L16 --> L17
    L01 --> L17
    F05 --> L18
    F15 --> L18
    L18 --> L19
    F02 --> L20
    F14 --> L20
    F02 --> L21
    F14 --> L21
    L20 --> L22
    L21 --> L22
    L22 --> L23
    F05 --> L23
    L23 --> L24
    L23 --> L28
    S04 --> L28

    %% New task dependencies (v2.3.0)
    F06 --> L29
    F01 --> L29
    L01 --> L29
    L21 --> L30
    L22 --> L30
    S04 --> L30
    L23 --> L31
    L21 --> L31
    S13 --> L31
    L11 --> L32
    F01 --> L32
    S13 --> L32
    L08 --> L33
    L31 --> L33
    S13 --> L33

    %% Surface dependencies
    F11 --> S08
    F16 --> S04
    S04 --> S05
    S04 --> S06
    S05 --> S15
    S04 --> S15
    L01 --> S01
    L17 --> S01
    L19 --> S01
    L23 --> S01
    S15 --> S01
    L29 --> S01
    S01 --> S17
    L17 --> S17
    L23 --> S17
    S01 --> S16
    S05 --> S16
    L23 --> S16
    S01 --> S02
    S01 --> S03
    S17 --> S03
    L01 --> S07
    S01 --> S09
    S01 --> S10
    S03 --> S10
    S16 --> S10
    S17 --> S10
    S10 --> S19
    S01 --> S19
    L23 --> S19
    S01 --> S18
    S17 --> S18
    L17 --> S18
    S01 --> S11
    S16 --> S11
    S17 --> S11
    S18 --> S11
    S01 --> S12
```

---

## Critical Path Analysis

The longest dependency chain (critical path) determines minimum completion time:

```
M03-F01 (ModelId) ─────────────────────────────────────────────────────────────┐
    │                                                                           │
    └─► M03-F09 (EmbeddingModel trait) ────────────────────────────────────────┐│
        │                                                                       ││
        └─► M03-L01 (ModelRegistry) ───────────────────────────────────────────┐││
            │                                                                   │││
            └─► M03-L03→L14 (12 Model Implementations) [PARALLELIZABLE] ───────┐│││
                │                                                               ││││
                └─► M03-L15 (ModelFactory Implementation) ─────────────────────┐│││││
                    │                                                           ││││││
                    └─► M03-L17 (BatchProcessor) ──────────────────────────────┐│││││││
                        │                                                       ││││││││
                        ├─► M03-L20→L22 (Gating, Experts, Router) [PARALLEL] ──┤│││││││││
                        │                                                       ││││││││││
                        └─► M03-L23 (FuseMoE) ─────────────────────────────────┐││││││││││
                            │                                                   │││││││││││
                            └─► M03-S01 (EmbeddingPipeline) ───────────────────┐││││││││││││
                                │                                               ││││││││││││││
                                └─► M03-S03 (EmbeddingProvider) ───────────────┘││││││││││││││
                                └─► M03-S10 (Integration Tests) ───────────────┘│││││││││││││
```

**Critical Path Duration: ~85 hours (sequential)**
**Optimized with Parallelization: ~55 hours (3.5 weeks)**

---

## Parallelization Opportunities

### Independent Foundation Tasks (Run in Parallel)
- M03-F01, M03-F06, M03-F07, M03-F08, M03-F13, M03-F14, M03-F15

### Independent Model Implementations (Run in Parallel)
- Custom models: M03-L04, M03-L05, M03-L06, M03-L11 (no registry dependency)
- Pretrained models: M03-L03, M03-L07, M03-L08, M03-L09, M03-L10, M03-L12, M03-L13, M03-L14 (after M03-L01)

### Independent Fusion Components (Run in Parallel)
- M03-L20 (GatingNetwork) and M03-L21 (Expert Networks)

### Independent Surface Tasks (Run in Parallel)
- M03-S04, M03-S05, M03-S06 (CUDA components)
- M03-S09, M03-S11, M03-S12 (after M03-S01)

---

## Quality Gates

| Gate | Required Tasks | Blocks |
|------|---------------|--------|
| **Foundation Complete** | M03-F01 → M03-F16 | Logic layer |
| **Types Validated** | M03-F01 → M03-F10 compile | Model implementations |
| **Custom Models Ready** | M03-L04, M03-L05, M03-L06, M03-L11 | Integration |
| **Pretrained Models Ready** | M03-L03, M03-L07→L14 | ModelFactory |
| **ModelFactory Ready** | M03-L15 | BatchProcessor |
| **Batch Ready** | M03-L17 | Pipeline |
| **Cache Ready** | M03-L19 | Pipeline |
| **Fusion Ready** | M03-L23 | Pipeline |
| **Pipeline Operational** | M03-S01 | Integration tests |
| **GDS Ready** | M03-S15 | High-speed model loading |
| **Warm-up Complete** | M03-S16 | First request latency |
| **Quantization Ready** | M03-L28 | FP8/FP4 inference |
| **Tokenization Ready** | M03-L29 | Shared tokenizer efficiency |
| **Grouped GEMM Ready** | M03-L30 | MoE 4x speedup |
| **Weights Loaded** | M03-L31 | FuseMoE initialization |
| **Orchestration Ready** | M03-S17 | Fail-safe async collection |
| **HDC Stable** | M03-L32 | Long-term embedding stability |
| **SPLADE Stable** | M03-L33 | Projection matrix persistence |
| **Tracing Ready** | M03-S18 | P95 latency debugging |
| **Alignment Verified** | M03-S19 | Unified space coherence |
| **Provider Compatible** | M03-S03 | Module 4 |
| **All Tests Pass** | M03-S09, M03-S10 | Release |
| **Performance Met** | M03-S11 benchmarks | Release |

---

## Task File References

### Master Files
| File | Description |
|------|-------------|
| `module-03-embedding-pipeline-tasks.md` | Consolidated YAML task definitions |
| `_traceability.md` | PRD requirement coverage matrix |
| `_index.md` | This file - execution order and dependencies |

### Foundation Layer (16 files)
| Task ID | File | Title |
|---------|------|-------|
| M03-F01 | `M03-F01.md` | ModelId Enum Definition |
| M03-F02 | `M03-F02.md` | ModelDimensions Constants |
| M03-F03 | `M03-F03.md` | ModelEmbedding Struct |
| M03-F04 | `M03-F04.md` | ConcatenatedEmbedding Struct |
| M03-F05 | `M03-F05.md` | FusedEmbedding Struct (1536D) |
| M03-F06 | `M03-F06.md` | ModelInput Enum |
| M03-F07 | `M03-F07.md` | InputType Enum |
| M03-F08 | `M03-F08.md` | EmbeddingError Enum |
| M03-F09 | `M03-F09.md` | EmbeddingModel Async Trait |
| M03-F10 | `M03-F10.md` | ModelFactory Trait |
| M03-F11 | `M03-F11.md` | EmbeddingConfig Root |
| M03-F12 | `M03-F12.md` | ModelRegistryConfig |
| M03-F13 | `M03-F13.md` | BatchConfig Struct |
| M03-F14 | `M03-F14.md` | FusionConfig Struct |
| M03-F15 | `M03-F15.md` | CacheConfig and GpuConfig |
| M03-F16 | `M03-F16.md` | Module Structure/Exports |

### Logic Layer (31 files)
| Task ID | File | Title |
|---------|------|-------|
| M03-L01 | `M03-L01.md` | ModelRegistry Core |
| M03-L02 | `M03-L02.md` | MemoryTracker |
| M03-L03 | `M03-L03.md` | Semantic Model (E1 - e5-large-v2) |
| M03-L04 | `M03-L04.md` | Temporal-Recent Model (E2) |
| M03-L05 | `M03-L05.md` | Temporal-Periodic Model (E3) |
| M03-L06 | `M03-L06.md` | Temporal-Positional Model (E4) |
| M03-L07 | `M03-L07.md` | Causal Model (E5 - Longformer) |
| M03-L08 | `M03-L08.md` | Sparse Model (E6 - SPLADE) |
| M03-L09 | `M03-L09.md` | Code Model (E7 - CodeBERT) |
| M03-L10 | `M03-L10.md` | Graph Model (E8 - MiniLM) |
| M03-L11 | `M03-L11.md` | HDC Model (E9 - Custom) |
| M03-L12 | `M03-L12.md` | Multimodal Model (E10 - CLIP) |
| M03-L13 | `M03-L13.md` | Entity Model (E11 - MiniLM) |
| M03-L14 | `M03-L14.md` | Late-Interaction (E12 - ColBERT) |
| M03-L15 | `M03-L15.md` | ModelFactory Implementation |
| M03-L16 | `M03-L16.md` | BatchQueue and Request Types |
| M03-L17 | `M03-L17.md` | BatchProcessor Implementation |
| M03-L18 | `M03-L18.md` | CacheEntry and CacheKey Types |
| M03-L19 | `M03-L19.md` | CacheManager Implementation |
| M03-L20 | `M03-L20.md` | GatingNetwork for FuseMoE |
| M03-L21 | `M03-L21.md` | Expert Networks (8 Experts) |
| M03-L22 | `M03-L22.md` | FuseMoE Router |
| M03-L23 | `M03-L23.md` | FuseMoE Main Module |
| M03-L24 | `M03-L24.md` | CAME-AB Bridge Layer |
| M03-L25 | `M03-L25.md` | PII Scrubber Implementation |
| M03-L26 | `M03-L26.md` | Neuromodulation Integration |
| M03-L27 | `M03-L27.md` | EmbeddingPriors/Context Priming |
| M03-L28 | `M03-L28.md` | Blackwell Quantization Kernels (FP8/FP4) |
| M03-L29 | `M03-L29.md` | Shared Tokenizer Cache (TokenizationManager) |
| M03-L30 | `M03-L30.md` | Grouped GEMM for MoE Expert Execution |
| M03-L31 | `M03-L31.md` | FuseMoE Weight Registry and Initialization |
| M03-L32 | `M03-L32.md` | HDC Base-Vector Persistence |
| M03-L33 | `M03-L33.md` | Sparse Projection Matrix Management (SPLADE) |

### Surface Layer (19 files)
| Task ID | File | Title |
|---------|------|-------|
| M03-S01 | `M03-S01.md` | EmbeddingPipeline Core |
| M03-S02 | `M03-S02.md` | PipelineMetrics/HealthStatus |
| M03-S03 | `M03-S03.md` | EmbeddingProvider Bridge |
| M03-S04 | `M03-S04.md` | CUDA Device Trait |
| M03-S05 | `M03-S05.md` | GPU Memory Pool |
| M03-S06 | `M03-S06.md` | CUDA Kernel Stubs |
| M03-S07 | `M03-S07.md` | HotSwap Model Loading |
| M03-S08 | `M03-S08.md` | Configuration File Loading |
| M03-S09 | `M03-S09.md` | Unit Tests Suite |
| M03-S10 | `M03-S10.md` | Integration Tests |
| M03-S11 | `M03-S11.md` | Benchmarks |
| M03-S12 | `M03-S12.md` | Documentation/Examples |
| M03-S13 | `M03-S13.md` | Model Artifact Manager |
| M03-S14 | `M03-S14.md` | Green Context SM Partitioning |
| M03-S15 | `M03-S15.md` | GPU Direct Storage (GDS) Integration |
| M03-S16 | `M03-S16.md` | Pipeline Warm-up & JIT Trigger |
| M03-S17 | `M03-S17.md` | Fail-Safe Async Orchestrator (FusionInputAssembler) |
| M03-S18 | `M03-S18.md` | Tracing & Async Instrumentation |
| M03-S19 | `M03-S19.md` | Semantic Alignment Verification |

---

## Model Mapping (Corrected)

| PRD ID | Task ID | Model Type | HuggingFace Repo | Dimension |
|--------|---------|------------|------------------|-----------|
| E1 | M03-L03 | Semantic | intfloat/e5-large-v2 | 1024D |
| E2 | M03-L04 | Temporal-Recent | CUSTOM | 512D |
| E3 | M03-L05 | Temporal-Periodic | CUSTOM | 512D |
| E4 | M03-L06 | Temporal-Positional | CUSTOM | 512D |
| E5 | M03-L07 | Causal | allenai/longformer-base-4096 | 768D |
| E6 | M03-L08 | Sparse | naver/splade-cocondenser-ensembledistil | 1536D* |
| E7 | M03-L09 | Code | microsoft/codebert-base | 768D |
| E8 | M03-L10 | Graph | sentence-transformers/paraphrase-MiniLM-L6-v2 | 384D |
| E9 | M03-L11 | HDC | CUSTOM | 1024D* |
| E10 | M03-L12 | Multimodal | openai/clip-vit-large-patch14 | 768D |
| E11 | M03-L13 | Entity | sentence-transformers/all-MiniLM-L6-v2 | 384D |
| E12 | M03-L14 | Late-Interaction | colbert-ir/colbertv2.0 | 128D |

*Projected dimensions after transformation

**Total Concatenated: 8320D → FuseMoE → 1536D**

---

## Performance Targets Summary

| Metric | Target | Task Validation |
|--------|--------|-----------------|
| Single embed E2E | <200ms P95 | M03-S11 |
| Batch throughput | >100 items/sec | M03-S11 |
| FuseMoE fusion | <3ms | M03-S11 |
| Cache hit latency | <100μs | M03-S11 |
| Cache hit rate | >80% | M03-S10 |
| GPU memory | <24GB | M03-S10 |

---

---

## New Tasks Summary (v2.4.0)

### M03-L32: HDC Base-Vector Persistence
- **Purpose**: Serialize and reload HDC base/position vectors for long-term embedding stability
- **Dependencies**: M03-L11 (HDC Model), M03-F01 (ModelId), M03-S13 (ArtifactManager)
- **Effort**: 3 hours
- **Key Feature**: Checksum verification, deterministic serialization, cross-restart stability

### M03-L33: Sparse Projection Matrix Management (SPLADE)
- **Purpose**: Persist random/learned projection matrix for SPLADE 30k→1536D projection
- **Dependencies**: M03-L08 (SPLADE), M03-L31 (WeightRegistry), M03-S13 (ArtifactManager)
- **Effort**: 3 hours
- **Key Feature**: Johnson-Lindenstrauss determinism, WeightRegistry integration

### M03-S18: Tracing & Async Instrumentation
- **Purpose**: Hierarchical spans for debugging parallel model execution bottlenecks
- **Dependencies**: M03-S01 (Pipeline), M03-S17 (Orchestrator), M03-L17 (BatchProcessor)
- **Effort**: 4 hours
- **Key Feature**: tokio-console, OpenTelemetry export, per-model timing visibility

### M03-S19: Semantic Alignment Verification
- **Purpose**: Integration tests ensuring FuseMoE produces coherent unified latent space
- **Dependencies**: M03-S01 (Pipeline), M03-L23 (FuseMoE), M03-S10 (Integration Tests)
- **Effort**: 4 hours
- **Key Feature**: Text↔Code equivalence tests, 90% pass rate threshold, similarity assertions

---

## Previous Tasks Summary (v2.3.0)

### M03-S17: Fail-Safe Async Orchestrator (FusionInputAssembler)
- **Purpose**: Handle partial results when models timeout or crash during parallel execution
- **Dependencies**: M03-S01 (Pipeline), M03-L17 (BatchProcessor), M03-L23 (FuseMoE)
- **Effort**: 4 hours
- **Key Feature**: 150ms hard timeout with tokio::JoinSet, graceful degradation

### M03-L29: Shared Tokenizer Cache (TokenizationManager)
- **Purpose**: Eliminate redundant tokenization across models sharing tokenizer families
- **Dependencies**: M03-F06 (ModelInput), M03-F01 (ModelId), M03-L01 (ModelRegistry)
- **Effort**: 3 hours
- **Key Feature**: 12 models → 5-6 unique tokenizations, <100μs cache hits

### M03-L30: Grouped GEMM for MoE Expert Execution
- **Purpose**: 4x speedup for MoE via cuBLAS/CUTLASS grouped GEMM (Blackwell optimized)
- **Dependencies**: M03-L21 (Expert Networks), M03-L22 (Router), M03-S04 (CUDA Device)
- **Effort**: 5 hours
- **Key Feature**: Single kernel launch for all active experts vs sequential

### M03-L31: FuseMoE Weight Registry and Initialization
- **Purpose**: Manage FuseMoE weights with checkpoint loading and deterministic initialization
- **Dependencies**: M03-L23 (FuseMoE), M03-L21 (Expert Networks), M03-S13 (Artifact Manager)
- **Effort**: 4 hours
- **Key Feature**: Golden checkpoint, Kaiming/Xavier init, hot-swap, version tracking

---

## Previous Tasks Summary (v2.2.0)

### M03-L28: Blackwell-Specific Quantization Kernels (FP8/FP4)
- **Purpose**: Implement FP8/FP4 quantization for maximum throughput on RTX 5090
- **Dependencies**: M03-L23 (FuseMoE), M03-S04 (CUDA Device)
- **Effort**: 5 hours
- **Key Feature**: Block-scaling logic for E4M3/E5M2/NVFP4 formats

### M03-S15: GPU Direct Storage (GDS) Integration
- **Purpose**: High-bandwidth NVMe→GPU DMA bypass for model loading
- **Dependencies**: M03-S04 (CUDA Device), M03-S05 (GPU Memory Pool)
- **Effort**: 6 hours
- **Key Feature**: 25+ GB/s model loading vs ~6 GB/s via CPU

### M03-S16: Pipeline Warm-up & JIT Trigger
- **Purpose**: Prime GPU and CUDA kernels during initialization
- **Dependencies**: M03-S01 (Pipeline), M03-S05 (Memory Pool), M03-L23 (FuseMoE)
- **Effort**: 3 hours
- **Key Feature**: Ensures <200ms P95 latency from first interaction

### M03-F05 Update: ColBERT Token Storage
- **Purpose**: Add `aux_data` field for per-token embeddings (Late-Interaction)
- **Key Feature**: `AuxiliaryEmbeddingData` struct for Module 4 graph storage

---

*Index Generated: 2026-01-01*
*Module: 03 - 12-Model Embedding Pipeline*
*Version: 2.4.0*
