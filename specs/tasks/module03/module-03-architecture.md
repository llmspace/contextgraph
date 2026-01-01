# Module 3: Embedding Pipeline Architecture Design

```yaml
metadata:
  document_type: architecture_design
  module_id: module-03
  version: 1.0.0
  created: 2026-01-01
  author: System Architecture Designer (Agent #2/7)
  layer_assignment: foundation/logic/surface
  total_files: 32
  total_tasks: 52
```

---

## 1. Executive Summary

This architecture document defines the complete file-by-file breakdown for Module 3 (Embedding Pipeline) with dependencies and layer assignments. The design follows the inside-out, bottom-up approach from the SPARC methodology.

### Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| 12 HuggingFace models via Candle | Rust-native inference eliminates Python FFI overhead |
| FuseMoE with 8 experts, top-k=2 | Balances quality vs. inference cost |
| 12954D concatenated -> 1536D output | Matches existing MemoryNode.embedding dimension |
| Lazy model loading | Reduces startup time, enables partial GPU memory usage |
| LRU cache with 100K entries | >80% hit rate target, disk persistence for restarts |

---

## 2. File Hierarchy Design

### 2.1 Complete File Tree

```
crates/context-graph-embeddings/
├── Cargo.toml                          # [F01] Crate dependencies
├── src/
│   ├── lib.rs                          # [F02] Module exports
│   ├── error.rs                        # [F03] EmbeddingError enum (existing, expand)
│   ├── config.rs                       # [F04] All configuration types
│   ├── pipeline.rs                     # [F05] EmbeddingPipeline main entry
│   │
│   ├── types/
│   │   ├── mod.rs                      # [F06] Types module exports
│   │   ├── model_id.rs                 # [F07] ModelId enum (12 variants)
│   │   ├── embedding.rs                # [F08] ModelEmbedding, ConcatenatedEmbedding
│   │   ├── fused.rs                    # [F09] FusedEmbedding struct
│   │   └── input.rs                    # [F10] ModelInput, InputType enums
│   │
│   ├── traits/
│   │   ├── mod.rs                      # [F11] Traits module exports
│   │   └── embedding_model.rs          # [F12] EmbeddingModel trait
│   │
│   ├── models/
│   │   ├── mod.rs                      # [F13] Models module exports
│   │   ├── registry.rs                 # [F14] ModelRegistry with lazy loading
│   │   ├── loader.rs                   # [F15] Model factory functions
│   │   ├── base.rs                     # [F16] BaseEmbeddingModel abstraction
│   │   │
│   │   ├── text/
│   │   │   ├── mod.rs                  # [F17] Text models exports
│   │   │   ├── minilm.rs               # [F18] E1: all-MiniLM-L6 (384D)
│   │   │   ├── bge.rs                  # [F19] E2: bge-large-en (1024D)
│   │   │   ├── instructor.rs           # [F20] E3: instructor-xl (768D)
│   │   │   ├── sentence_t5.rs          # [F21] E7: sentence-t5-xxl (768D)
│   │   │   ├── mpnet.rs                # [F22] E8: all-mpnet-base (768D)
│   │   │   ├── contriever.rs           # [F23] E9: contriever-msmarco (768D)
│   │   │   ├── dragon.rs               # [F24] E10: dragon-plus (768D)
│   │   │   ├── gte.rs                  # [F25] E11: gte-large-en (1024D)
│   │   │   └── e5_mistral.rs           # [F26] E12: e5-mistral-7b (4096D)
│   │   │
│   │   ├── multimodal/
│   │   │   ├── mod.rs                  # [F27] Multimodal exports
│   │   │   ├── clip.rs                 # [F28] E4: clip-vit-large (768D)
│   │   │   └── whisper.rs              # [F29] E5: whisper-large (1280D)
│   │   │
│   │   └── code/
│   │       ├── mod.rs                  # [F30] Code models exports
│   │       └── codebert.rs             # [F31] E6: codebert-base (768D)
│   │
│   ├── fusion/
│   │   ├── mod.rs                      # [F32] Fusion module exports
│   │   ├── gating.rs                   # [F33] GatingNetwork
│   │   ├── experts.rs                  # [F34] Expert MLP networks
│   │   ├── router.rs                   # [F35] Top-k routing
│   │   └── fusemoe.rs                  # [F36] FuseMoE layer
│   │
│   ├── batch/
│   │   ├── mod.rs                      # [F37] Batch module exports
│   │   ├── processor.rs                # [F38] BatchProcessor
│   │   ├── queue.rs                    # [F39] BatchQueue
│   │   └── padding.rs                  # [F40] Padding strategies
│   │
│   ├── cache/
│   │   ├── mod.rs                      # [F41] Cache module exports
│   │   ├── manager.rs                  # [F42] CacheManager
│   │   ├── lru.rs                      # [F43] LRU eviction policy
│   │   └── disk.rs                     # [F44] DiskCache persistence
│   │
│   └── provider_impl.rs                # [F45] EmbeddingProvider trait impl
│
├── tests/
│   ├── model_id_tests.rs               # [T01]
│   ├── embedding_types_tests.rs        # [T02]
│   ├── fused_embedding_tests.rs        # [T03]
│   ├── error_tests.rs                  # [T04]
│   ├── config_tests.rs                 # [T05]
│   ├── trait_tests.rs                  # [T06]
│   ├── registry_tests.rs               # [T07]
│   ├── batch_tests.rs                  # [T08]
│   ├── cache_tests.rs                  # [T09]
│   ├── fusemoe_tests.rs                # [T10]
│   ├── pipeline_tests.rs               # [T11]
│   └── integration_tests.rs            # [T12]
│
└── benches/
    ├── embedding_bench.rs              # [B01]
    └── fusion_bench.rs                 # [B02]

crates/context-graph-cuda/              # (Existing, extend)
├── src/
│   ├── kernels/
│   │   ├── mod.rs                      # [F46] Kernel exports
│   │   ├── embedding.rs                # [F47] Embedding CUDA ops
│   │   ├── fusion.rs                   # [F48] FuseMoE CUDA ops
│   │   └── similarity.rs               # [F49] Similarity CUDA ops
│   └── memory.rs                       # [F50] GPU memory manager
│
└── kernels/                            # CUDA source files
    ├── embedding.cu                    # [C01]
    ├── fusemoe.cu                      # [C02]
    └── similarity.cu                   # [C03]
```

---

## 3. Dependency Graph (DAG)

### 3.1 Core Type Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FOUNDATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   F07:ModelId ──────────────────┬──────────────────────────────────────►│
│        │                        │                                        │
│        ▼                        ▼                                        │
│   F08:ModelEmbedding    F09:FusedEmbedding                              │
│        │                        │                                        │
│        └────────┬───────────────┘                                        │
│                 ▼                                                        │
│   F03:EmbeddingError ◄──────────────────────────────────────────────────│
│        │                                                                 │
│        ▼                                                                 │
│   F10:ModelInput ──────────────────────────────────────────────────────►│
│        │                                                                 │
│        └──────────────────┐                                              │
│                           ▼                                              │
│                    F12:EmbeddingModel trait ◄───────────────────────────│
│                           │                                              │
│   F04:Configuration ──────┤                                              │
│        │                  │                                              │
│        └──────────────────┤                                              │
│                           ▼                                              │
│                    F06/F11:Type Modules                                  │
│                           │                                              │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            LOGIC LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   F14:ModelRegistry ◄────────────────┬──────────────────────────────────│
│        │                             │                                   │
│        ├───────────────┐             │                                   │
│        ▼               ▼             │                                   │
│   F15:Loader    F16:BaseModel        │                                   │
│        │               │             │                                   │
│        └───────┬───────┘             │                                   │
│                │                     │                                   │
│                ▼                     │                                   │
│   ┌────────────────────────────┐     │                                   │
│   │    Model Implementations   │     │                                   │
│   │ F18-F31 (12 HF models)     │     │                                   │
│   └────────────────────────────┘     │                                   │
│                │                     │                                   │
│                ▼                     │                                   │
│   F38:BatchProcessor ◄───────────────┤                                   │
│        │                             │                                   │
│        ├───────────────┐             │                                   │
│        ▼               ▼             │                                   │
│   F39:BatchQueue  F40:Padding        │                                   │
│                                      │                                   │
│   F42:CacheManager ◄─────────────────┤                                   │
│        │                             │                                   │
│        ├───────────────┐             │                                   │
│        ▼               ▼             │                                   │
│   F43:LRU        F44:DiskCache       │                                   │
│                                      │                                   │
│   F33:GatingNetwork ◄────────────────┤                                   │
│        │                             │                                   │
│        ▼                             │                                   │
│   F34:Experts ◄──────────────────────┤                                   │
│        │                             │                                   │
│        ▼                             │                                   │
│   F35:Router ◄───────────────────────┤                                   │
│        │                             │                                   │
│        ▼                             │                                   │
│   F36:FuseMoE ◄──────────────────────┘                                   │
│        │                                                                 │
└────────┼─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SURFACE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   F05:EmbeddingPipeline ◄────────────────────────────────────────────────│
│        │                                                                 │
│        │  Composes: Registry + BatchProcessor + CacheManager + FuseMoE   │
│        │                                                                 │
│        ▼                                                                 │
│   F45:ProviderImpl (EmbeddingProvider trait from Module 1)              │
│        │                                                                 │
│        └──────────► Replaces StubEmbedder                                │
│                                                                          │
│   F47-F50:CUDA Integration (GPU acceleration path)                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Explicit File Dependencies

| File ID | File Path | Depends On | Depended By |
|---------|-----------|------------|-------------|
| F01 | Cargo.toml | - | All |
| F02 | lib.rs | F03-F45 | External crates |
| F03 | error.rs | - | F07-F45 |
| F04 | config.rs | F07 | F14, F38, F42, F36 |
| F05 | pipeline.rs | F14, F38, F42, F36 | F45 |
| F06 | types/mod.rs | F07-F10 | F02 |
| F07 | types/model_id.rs | - | F08, F09, F12, F14-F31 |
| F08 | types/embedding.rs | F07 | F12, F14, F38, F36 |
| F09 | types/fused.rs | F07 | F36, F42, F05 |
| F10 | types/input.rs | - | F12, F18-F31 |
| F11 | traits/mod.rs | F12 | F02 |
| F12 | traits/embedding_model.rs | F07, F08, F10, F03 | F14-F31 |
| F13 | models/mod.rs | F14-F31 | F02 |
| F14 | models/registry.rs | F04, F12 | F05, F15 |
| F15 | models/loader.rs | F14, F16 | F14 |
| F16 | models/base.rs | F12 | F18-F31 |
| F17 | models/text/mod.rs | F18-F26 | F13 |
| F18-F31 | (Individual models) | F12, F16 | F14 |
| F32 | fusion/mod.rs | F33-F36 | F02 |
| F33 | fusion/gating.rs | F04, F08 | F35 |
| F34 | fusion/experts.rs | F04 | F35 |
| F35 | fusion/router.rs | F33, F34, F04 | F36 |
| F36 | fusion/fusemoe.rs | F35, F08, F09, F04 | F05 |
| F37 | batch/mod.rs | F38-F40 | F02 |
| F38 | batch/processor.rs | F04, F14, F39, F40 | F05 |
| F39 | batch/queue.rs | F10 | F38 |
| F40 | batch/padding.rs | F04 | F38 |
| F41 | cache/mod.rs | F42-F44 | F02 |
| F42 | cache/manager.rs | F04, F09, F43, F44 | F05 |
| F43 | cache/lru.rs | F04 | F42 |
| F44 | cache/disk.rs | F09, F04 | F42 |
| F45 | provider_impl.rs | F05 | F02 |

---

## 4. Layer Assignment

### 4.1 Foundation Layer (16 files)

Types, traits, errors, and configuration - no business logic.

| File ID | Path | Purpose | LOC Estimate |
|---------|------|---------|--------------|
| F01 | Cargo.toml | Crate manifest | 80 |
| F02 | lib.rs | Module exports | 100 |
| F03 | error.rs | EmbeddingError enum | 150 |
| F04 | config.rs | All configuration structs | 350 |
| F06 | types/mod.rs | Types module | 20 |
| F07 | types/model_id.rs | ModelId enum | 200 |
| F08 | types/embedding.rs | ModelEmbedding, Concatenated | 180 |
| F09 | types/fused.rs | FusedEmbedding struct | 150 |
| F10 | types/input.rs | ModelInput, InputType | 120 |
| F11 | traits/mod.rs | Traits module | 15 |
| F12 | traits/embedding_model.rs | EmbeddingModel trait | 120 |
| F13 | models/mod.rs | Models module | 30 |
| F17 | models/text/mod.rs | Text models | 25 |
| F27 | models/multimodal/mod.rs | Multimodal models | 20 |
| F30 | models/code/mod.rs | Code models | 15 |
| F32 | fusion/mod.rs | Fusion module | 20 |

**Total Foundation Files: 16**
**Total Foundation LOC: ~1,595**

### 4.2 Logic Layer (22 files)

Core business logic - models, batching, caching, fusion.

| File ID | Path | Purpose | LOC Estimate |
|---------|------|---------|--------------|
| F14 | models/registry.rs | ModelRegistry | 400 |
| F15 | models/loader.rs | Model factories | 350 |
| F16 | models/base.rs | BaseEmbeddingModel | 250 |
| F18 | models/text/minilm.rs | E1: MiniLM | 280 |
| F19 | models/text/bge.rs | E2: BGE | 280 |
| F20 | models/text/instructor.rs | E3: Instructor | 300 |
| F21 | models/text/sentence_t5.rs | E7: SentenceT5 | 300 |
| F22 | models/text/mpnet.rs | E8: MPNet | 280 |
| F23 | models/text/contriever.rs | E9: Contriever | 280 |
| F24 | models/text/dragon.rs | E10: DRAGON | 280 |
| F25 | models/text/gte.rs | E11: GTE | 280 |
| F26 | models/text/e5_mistral.rs | E12: E5-Mistral | 400 |
| F28 | models/multimodal/clip.rs | E4: CLIP | 350 |
| F29 | models/multimodal/whisper.rs | E5: Whisper | 350 |
| F31 | models/code/codebert.rs | E6: CodeBERT | 300 |
| F33 | fusion/gating.rs | GatingNetwork | 250 |
| F34 | fusion/experts.rs | Expert networks | 300 |
| F35 | fusion/router.rs | Top-k router | 280 |
| F36 | fusion/fusemoe.rs | FuseMoE layer | 350 |
| F38 | batch/processor.rs | BatchProcessor | 400 |
| F42 | cache/manager.rs | CacheManager | 380 |
| F43 | cache/lru.rs | LRU eviction | 200 |

**Total Logic Files: 22**
**Total Logic LOC: ~6,840**

### 4.3 Surface Layer (12 files)

Pipeline integration, CUDA, provider implementation.

| File ID | Path | Purpose | LOC Estimate |
|---------|------|---------|--------------|
| F05 | pipeline.rs | EmbeddingPipeline | 450 |
| F37 | batch/mod.rs | Batch module | 25 |
| F39 | batch/queue.rs | BatchQueue | 250 |
| F40 | batch/padding.rs | Padding strategies | 180 |
| F41 | cache/mod.rs | Cache module | 20 |
| F44 | cache/disk.rs | DiskCache | 300 |
| F45 | provider_impl.rs | EmbeddingProvider | 200 |
| F46 | cuda/kernels/mod.rs | Kernel exports | 30 |
| F47 | cuda/kernels/embedding.rs | Embedding CUDA | 300 |
| F48 | cuda/kernels/fusion.rs | Fusion CUDA | 280 |
| F49 | cuda/kernels/similarity.rs | Similarity CUDA | 200 |
| F50 | cuda/memory.rs | GPU memory mgr | 350 |

**Total Surface Files: 12**
**Total Surface LOC: ~2,585**

---

## 5. Task Breakdown (52 Atomic Tasks)

### 5.1 Foundation Layer Tasks (16 tasks)

| Task ID | Title | File(s) | Hours | Dependencies |
|---------|-------|---------|-------|--------------|
| M03-F01 | Update Cargo.toml with dependencies | F01 | 1 | - |
| M03-F02 | Define ModelId enum (12 variants) | F07 | 2 | - |
| M03-F03 | Expand EmbeddingError enum | F03 | 1.5 | - |
| M03-F04 | Define ModelInput and InputType | F10 | 1.5 | - |
| M03-F05 | Define ModelEmbedding struct | F08 | 2 | M03-F02 |
| M03-F06 | Define ConcatenatedEmbedding struct | F08 | 1 | M03-F02, M03-F05 |
| M03-F07 | Define FusedEmbedding struct | F09 | 2 | M03-F02 |
| M03-F08 | Define EmbeddingModel trait | F12 | 2 | M03-F02, M03-F03, M03-F04, M03-F05 |
| M03-F09 | Define EmbeddingConfig struct | F04 | 1.5 | M03-F02 |
| M03-F10 | Define ModelRegistryConfig | F04 | 1.5 | M03-F02 |
| M03-F11 | Define BatchConfig | F04 | 1 | - |
| M03-F12 | Define FusionConfig | F04 | 1 | - |
| M03-F13 | Define CacheConfig | F04 | 1 | - |
| M03-F14 | Define GpuConfig | F04 | 1 | - |
| M03-F15 | Create types module structure | F06, F11 | 1 | M03-F02 to M03-F08 |
| M03-F16 | Create lib.rs with exports | F02 | 1 | M03-F15 |

**Foundation Total: 21.0 hours**

### 5.2 Logic Layer Tasks (24 tasks)

| Task ID | Title | File(s) | Hours | Dependencies |
|---------|-------|---------|-------|--------------|
| M03-L01 | Implement BaseEmbeddingModel | F16 | 3 | M03-F08 |
| M03-L02 | Implement ModelRegistry core | F14 | 4 | M03-L01, M03-F10 |
| M03-L03 | Implement lazy loading in Registry | F14 | 3 | M03-L02 |
| M03-L04 | Implement model loader factory | F15 | 2 | M03-L02 |
| M03-L05 | Implement MiniLM (E1) | F18 | 3 | M03-L01 |
| M03-L06 | Implement BGE (E2) | F19 | 3 | M03-L01 |
| M03-L07 | Implement Instructor (E3) | F20 | 3 | M03-L01 |
| M03-L08 | Implement CLIP (E4) | F28 | 4 | M03-L01 |
| M03-L09 | Implement Whisper (E5) | F29 | 4 | M03-L01 |
| M03-L10 | Implement CodeBERT (E6) | F31 | 3 | M03-L01 |
| M03-L11 | Implement SentenceT5 (E7) | F21 | 3 | M03-L01 |
| M03-L12 | Implement MPNet (E8) | F22 | 3 | M03-L01 |
| M03-L13 | Implement Contriever (E9) | F23 | 3 | M03-L01 |
| M03-L14 | Implement DRAGON (E10) | F24 | 3 | M03-L01 |
| M03-L15 | Implement GTE (E11) | F25 | 3 | M03-L01 |
| M03-L16 | Implement E5-Mistral (E12) | F26 | 4 | M03-L01 |
| M03-L17 | Implement GatingNetwork | F33 | 3 | M03-F06, M03-F12 |
| M03-L18 | Implement Expert networks | F34 | 4 | M03-F12 |
| M03-L19 | Implement Router (top-k) | F35 | 3 | M03-L17, M03-L18 |
| M03-L20 | Implement FuseMoE layer | F36 | 4 | M03-L19, M03-F07 |
| M03-L21 | Implement BatchQueue | F39 | 2 | M03-F04 |
| M03-L22 | Implement BatchProcessor | F38 | 4 | M03-L02, M03-L21, M03-F11 |
| M03-L23 | Implement LRU cache | F43 | 2 | M03-F13 |
| M03-L24 | Implement CacheManager | F42 | 4 | M03-L23, M03-F07, M03-F13 |

**Logic Total: 77.0 hours**

### 5.3 Surface Layer Tasks (12 tasks)

| Task ID | Title | File(s) | Hours | Dependencies |
|---------|-------|---------|-------|--------------|
| M03-S01 | Implement padding strategies | F40 | 2 | M03-F11 |
| M03-S02 | Implement DiskCache | F44 | 3 | M03-F07, M03-F13 |
| M03-S03 | Implement EmbeddingPipeline | F05 | 5 | M03-L02, M03-L20, M03-L22, M03-L24 |
| M03-S04 | Implement EmbeddingProvider | F45 | 2 | M03-S03 |
| M03-S05 | Implement hot-swap capability | F14 | 3 | M03-S03 |
| M03-S06 | Define CUDA kernel interfaces | F46-F49 | 4 | M03-F08, M03-F07 |
| M03-S07 | Implement GPU memory manager | F50 | 3 | M03-S06 |
| M03-S08 | Create batch module structure | F37 | 0.5 | M03-L21, M03-L22, M03-S01 |
| M03-S09 | Create cache module structure | F41 | 0.5 | M03-L23, M03-L24, M03-S02 |
| M03-S10 | Create model submodule structures | F17, F27, F30 | 1 | M03-L05 to M03-L16 |
| M03-S11 | Integration tests | T01-T12 | 6 | M03-S03 |
| M03-S12 | Performance benchmarks | B01-B02 | 4 | M03-S03 |

**Surface Total: 34.0 hours**

---

## 6. Task Estimates Summary

| Layer | Files | Tasks | Hours | Percentage |
|-------|-------|-------|-------|------------|
| Foundation | 16 | 16 | 21.0 | 16% |
| Logic | 22 | 24 | 77.0 | 58% |
| Surface | 12 | 12 | 34.0 | 26% |
| **Total** | **50** | **52** | **132.0** | 100% |

### Week-by-Week Allocation

| Week | Tasks | Focus |
|------|-------|-------|
| Week 1 | M03-F01 to M03-F16 (Foundation) | Types, traits, configuration |
| Week 2 | M03-L01 to M03-L12 | ModelRegistry, first 8 models |
| Week 3 | M03-L13 to M03-L24 | Remaining models, fusion, cache |
| Week 4 | M03-S01 to M03-S12 | Pipeline, CUDA, integration, benchmarks |

---

## 7. Critical Path Analysis

The following is the longest dependency chain (critical path):

```
M03-F02 (ModelId)
    └─► M03-F08 (EmbeddingModel trait)
        └─► M03-L01 (BaseEmbeddingModel)
            └─► M03-L02 (ModelRegistry)
                └─► M03-L05-L16 (12 model implementations) [parallel]
                    └─► M03-L22 (BatchProcessor)
                        └─► M03-L20 (FuseMoE)
                            └─► M03-L24 (CacheManager)
                                └─► M03-S03 (EmbeddingPipeline)
                                    └─► M03-S04 (EmbeddingProvider)
```

**Critical Path Duration: ~85 hours (3.5 weeks with parallelization)**

---

## 8. Memory/Storage Locations

The architecture data should be stored in the following memory locations for downstream agents:

```bash
# File hierarchy
npx claude-flow memory store "file-hierarchy" '{"files": ["F01:Cargo.toml", "F02:lib.rs", "F03:error.rs", "F04:config.rs", "F05:pipeline.rs", "F06:types/mod.rs", "F07:types/model_id.rs", "F08:types/embedding.rs", "F09:types/fused.rs", "F10:types/input.rs", "F11:traits/mod.rs", "F12:traits/embedding_model.rs", "F13:models/mod.rs", "F14:models/registry.rs", "F15:models/loader.rs", "F16:models/base.rs", "F17:models/text/mod.rs", "F18:models/text/minilm.rs", "F19:models/text/bge.rs", "F20:models/text/instructor.rs", "F21:models/text/sentence_t5.rs", "F22:models/text/mpnet.rs", "F23:models/text/contriever.rs", "F24:models/text/dragon.rs", "F25:models/text/gte.rs", "F26:models/text/e5_mistral.rs", "F27:models/multimodal/mod.rs", "F28:models/multimodal/clip.rs", "F29:models/multimodal/whisper.rs", "F30:models/code/mod.rs", "F31:models/code/codebert.rs", "F32:fusion/mod.rs", "F33:fusion/gating.rs", "F34:fusion/experts.rs", "F35:fusion/router.rs", "F36:fusion/fusemoe.rs", "F37:batch/mod.rs", "F38:batch/processor.rs", "F39:batch/queue.rs", "F40:batch/padding.rs", "F41:cache/mod.rs", "F42:cache/manager.rs", "F43:cache/lru.rs", "F44:cache/disk.rs", "F45:provider_impl.rs"]}' --namespace "contextgraph/module3/architecture"

# Layer assignments
npx claude-flow memory store "layer-assignment" '{"foundation": ["F01", "F02", "F03", "F04", "F06", "F07", "F08", "F09", "F10", "F11", "F12", "F13", "F17", "F27", "F30", "F32"], "logic": ["F14", "F15", "F16", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F28", "F29", "F31", "F33", "F34", "F35", "F36", "F38", "F42", "F43"], "surface": ["F05", "F37", "F39", "F40", "F41", "F44", "F45", "F46", "F47", "F48", "F49", "F50"]}' --namespace "contextgraph/module3/architecture"

# Task estimates
npx claude-flow memory store "task-estimates" '{"total": 52, "foundation": 16, "logic": 24, "surface": 12, "total_hours": 132, "foundation_hours": 21, "logic_hours": 77, "surface_hours": 34}' --namespace "contextgraph/module3/architecture"

# Dependency DAG (simplified)
npx claude-flow memory store "dependency-dag" '{"F07": [], "F08": ["F07"], "F09": ["F07"], "F12": ["F07", "F08", "F10", "F03"], "F14": ["F04", "F12"], "F16": ["F12"], "F18-F31": ["F12", "F16"], "F33": ["F08", "F04"], "F34": ["F04"], "F35": ["F33", "F34"], "F36": ["F35", "F09"], "F38": ["F14", "F39", "F40"], "F42": ["F43", "F09"], "F05": ["F14", "F38", "F42", "F36"], "F45": ["F05"]}' --namespace "contextgraph/module3/architecture"
```

---

## 9. Next Agent Guidance

### For Specification Agents (Foundation/Logic/Surface):

1. **Foundation Agent**: Start with M03-F01 through M03-F16. Focus on type correctness and trait design. All types must be `Send + Sync`.

2. **Logic Agent**: Begin M03-L01 after Foundation is complete. Implement models in parallel batches (E1-E6 first, then E7-E12). FuseMoE depends on all models being complete.

3. **Surface Agent**: EmbeddingPipeline (M03-S03) is the integration point. Ensure graceful degradation when models fail.

### Key Constraints to Enforce:

- Total concatenated dimension: 12,954D (sum of all 12 models)
- FuseMoE output: 1,536D (matches existing MemoryNode.embedding)
- GPU memory budget: <16GB for all models
- Latency targets: <200ms single, >100 items/sec batch
- Cache hit rate target: >80%

### Critical Integration Points:

1. `EmbeddingProvider` trait (F45) must match existing interface from Module 1
2. `FusedEmbedding.vector` must be 1536D for FAISS compatibility
3. `CacheManager` must use xxhash64 for deterministic content hashing

---

*Document generated: 2026-01-01*
*Architecture Design Version: 1.0.0*
*Module: 03 - Embedding Pipeline*
