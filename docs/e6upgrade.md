# E6 Sparse Embedder Upgrade Proposal

**Version:** 1.0
**Date:** 2026-01-22
**Status:** Draft
**Author:** Context Graph Development Team

---

## Executive Summary

E6 (V_selectivity) sparse embedder is currently **underutilized** in the context-graph system. Despite being designed for keyword-level precision, it's stored as dense vectors and used with minimal weight (5%) in fusion scoring. This proposal outlines enhancements to properly integrate E6 as a **supporting embedder** that enhances E1 semantic retrieval rather than competing with it.

**Key Insight:** The 13-embedder system creates fingerprints where each embedder captures a different aspect of memory. E6's role is to capture **exact keyword/term matches** that E1's semantic understanding might miss - especially for technical terms, API paths, version strings, and proper nouns.

---

## Current State Analysis

### How E6 Currently Works

| Aspect | Current Implementation | Issue |
|--------|----------------------|-------|
| **Storage** | Projected to dense 1536D | Loses sparsity advantage |
| **Similarity** | Cosine on projected vectors | Jaccard code is dead |
| **Index Type** | HNSW (dense) | Should use inverted index |
| **Pipeline Role** | None (only E13 for Stage 1) | Underutilized |
| **Weight in Fusion** | 5% (backup) | Too low for its strength |
| **Vocabulary** | 30,522 (BERT) | Good |
| **Sparsity** | ~235 active terms (0.77%) | Excellent |

### Benchmark Results (Wikipedia Data)

| Metric | E6 Sparse | E1 Semantic | E13 SPLADE |
|--------|-----------|-------------|------------|
| MRR@10 | 0.8106 | 0.8406 | 0.8841 |
| Best For | Technical terms | General semantic | Learned expansion |

### Per-Topic Performance (E6)

| Topic Type | MRR@10 | Analysis |
|------------|--------|----------|
| Technical/Proper nouns | 1.000 | `anarchism`, `aristotle`, `albedo` |
| Common words | 0.250-0.774 | `alabama`, `academy` |

**Key Finding:** E6 excels at distinctive terms but struggles with common words - exactly where E1 should take over.

---

## Problem Statement

### 1. E6 is Not Stored as Sparse

```rust
// Current: projects sparse → dense (loses information)
pub fn embed(&self, ...) -> EmbeddingResult<Vec<f32>> {
    let sparse = self.embed_sparse(text)?;
    self.projection.project(&sparse)  // Returns 1536D dense
}
```

The original 30,522-dimensional sparse vector with ~235 active terms is projected to a dense 1536D vector, **losing the sparsity advantage**.

### 2. Inverted Index Declared But Not Implemented

```rust
// In registry.rs - claims E6 uses inverted index
// NOT INCLUDED (use other index types):
// - E6Sparse: InvertedIndex
// - E13Splade: InvertedIndex
```

But **no inverted index implementation exists** for E6. The `EmbedderIndexRegistry.get(E6Sparse)` returns `None`.

### 3. E6 Not Used in Retrieval Pipeline

The 5-stage pipeline uses:
- **Stage 1:** E13 SPLADE (sparse recall) - but NOT E6
- **Stage 2:** Matryoshka (dense filter)
- **Stage 3:** Full HNSW (13-space)
- **Stage 4:** Teleological alignment
- **Stage 5:** E12 ColBERT (late interaction)

E6 participates only in Stage 3 fusion with minimal weight.

### 4. Jaccard Similarity is Dead Code

```rust
// distance.rs - implemented but never called
pub fn jaccard_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
    a.jaccard_similarity(b)
}
```

All E6 comparisons use **cosine on projected dense**, not Jaccard on original sparse.

---

## Proposed Enhancements

### Enhancement 1: Dual Storage (Sparse + Projected)

Store E6 embeddings in **both** formats:

```rust
pub struct E6DualEmbedding {
    /// Original sparse vector for Stage 1 recall
    pub sparse: SparseVector,           // ~235 active terms
    /// Projected dense vector for Stage 3 fusion
    pub dense_projected: Vec<f32>,      // 1536D
}
```

**Benefits:**
- Stage 1: Use sparse for fast inverted index recall
- Stage 3: Use projected dense for weighted fusion with E1
- Storage overhead: ~2KB per memory (acceptable)

### Enhancement 2: E6 Inverted Index Implementation

Create a proper inverted index for E6 sparse vectors:

```rust
pub struct E6InvertedIndex {
    /// Term ID → Vec<(memory_id, weight)>
    postings: HashMap<u32, Vec<(Uuid, f32)>>,
    /// Memory ID → sparse vector for scoring
    vectors: HashMap<Uuid, SparseVector>,
}

impl E6InvertedIndex {
    /// Fast term-based candidate generation
    pub fn recall(&self, query_sparse: &SparseVector, k: usize) -> Vec<Uuid> {
        // Union of posting lists for query terms
        // Rank by BM25 or dot product
    }
}
```

**Benefits:**
- O(active_terms) query complexity vs O(n) dense scan
- Perfect for exact keyword matching
- Complements E13 SPLADE learned expansion

### Enhancement 3: E6 as Stage 1 Co-Pilot

Modify the retrieval pipeline to use **both** E6 and E13 for Stage 1:

```
Stage 1: DUAL SPARSE RECALL
├── E6 Inverted Index (exact keywords)    → 500 candidates
├── E13 SPLADE Index (learned expansion)  → 500 candidates
└── Union + Dedup                         → ~800 candidates

Stage 2: Matryoshka 128D filter           → 200 candidates
Stage 3: Full 13-space HNSW (including E6 projected dense)
Stage 4: Teleological alignment
Stage 5: E12 ColBERT re-ranking
```

**Benefits:**
- E6 catches exact technical terms E13 might expand away from
- E13 catches semantic variations E6 misses
- Better recall for mixed technical/general queries

### Enhancement 4: Query-Aware E6 Weight Boosting

Detect query type and dynamically boost E6 weight:

```rust
pub fn detect_e6_boost(query: &str) -> f32 {
    let mut boost = 1.0;

    // Technical indicators boost E6
    if contains_api_path(query) { boost += 0.5; }      // tokio::spawn
    if contains_version_string(query) { boost += 0.3; } // TLS 1.3
    if contains_acronym(query) { boost += 0.3; }        // HNSW, UUID
    if contains_proper_noun(query) { boost += 0.2; }    // PostgreSQL

    // General language reduces E6 (let E1 handle)
    if high_common_word_ratio(query) { boost -= 0.3; }

    boost.clamp(0.5, 2.0)
}
```

**Weight Profile with Dynamic Boost:**

| Query Type | Base E6 | Boost | Final E6 |
|------------|---------|-------|----------|
| `tokio::spawn async` | 0.10 | 1.5x | 0.15 |
| `TLS 1.3 handshake` | 0.10 | 1.3x | 0.13 |
| `how to use HNSW` | 0.10 | 1.3x | 0.13 |
| `what is democracy` | 0.10 | 0.7x | 0.07 |

### Enhancement 5: E6-Specific Similarity for Code Queries

For code-related queries, use **E6 term overlap** as a scoring signal:

```rust
pub fn e6_term_overlap_score(query_sparse: &SparseVector, doc_sparse: &SparseVector) -> f32 {
    let shared_terms = query_sparse.intersection_count(doc_sparse);
    let query_terms = query_sparse.active_count();

    // Precision-focused: what fraction of query terms appear in doc?
    shared_terms as f32 / query_terms as f32
}
```

**Use Cases:**
- API documentation search: `HashMap::get` must match exactly
- Error message search: specific tokens are diagnostic
- Configuration search: exact key names matter

### Enhancement 6: E6 as Tie-Breaker for E1

When E1 returns multiple memories with similar scores, use E6 to break ties:

```rust
pub fn apply_e6_tiebreaker(
    candidates: &mut Vec<ScoredMemory>,
    query_sparse: &SparseVector,
    store: &E6InvertedIndex,
) {
    for memory in candidates {
        if let Some(doc_sparse) = store.get_sparse(memory.id) {
            // Small boost for exact term overlap
            let overlap = e6_term_overlap_score(query_sparse, doc_sparse);
            memory.score += overlap * 0.05;  // 5% max tie-breaker
        }
    }
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
}
```

**Benefits:**
- Doesn't change ranking when E1 is confident
- Helps disambiguate when scores are close
- Rewards exact keyword matches without overriding semantic

---

## Implementation Plan

### Phase 1: Dual Storage Infrastructure (Week 1-2)

1. **Modify TeleologicalFingerprint**
   - Add `e6_sparse: Option<SparseVector>` field
   - Backward compatible (None for existing memories)

2. **Update SparseModel.embed()**
   - Return both sparse and projected dense
   - Store both in fingerprint

3. **Storage Layer**
   - Serialize/deserialize sparse vectors
   - ~2KB overhead per memory

### Phase 2: E6 Inverted Index (Week 2-3)

1. **Implement E6InvertedIndex**
   - Posting lists with term→memories mapping
   - BM25 or sparse dot product scoring

2. **Index Builder**
   - Batch indexing from existing memories
   - Incremental updates on new memories

3. **Recall API**
   - `recall(query_sparse, k) -> Vec<Uuid>`
   - Integrate with pipeline Stage 1

### Phase 3: Pipeline Integration (Week 3-4)

1. **Dual Stage 1 Recall**
   - E6 inverted index recall
   - E13 SPLADE recall
   - Union with deduplication

2. **Query-Aware Boosting**
   - Technical term detection
   - Dynamic E6 weight adjustment

3. **Tie-Breaker Integration**
   - Post-Stage-3 E6 term overlap scoring
   - Small boost for exact matches

### Phase 4: Benchmark & Tune (Week 4)

1. **Run sparse_bench with real data**
   - Measure E6 Stage 1 recall quality
   - Compare dual vs single sparse recall

2. **A/B Testing**
   - Old pipeline vs new dual-recall
   - Measure MRR improvement

3. **Threshold Tuning**
   - Optimal E6 boost factors
   - Tie-breaker weight

---

## Success Metrics

| Metric | Current | Target | Rationale |
|--------|---------|--------|-----------|
| E6 Stage 1 recall rate | 0% | 50% | Half of candidates from E6 |
| Technical query MRR | 0.81 | 0.90+ | E6 should improve exact matches |
| General query MRR | 0.84 | 0.84+ | E1 maintains semantic quality |
| Storage overhead | 0 | <5% | Dual storage cost |
| Stage 1 latency | <5ms | <10ms | Dual index overhead |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY INPUT                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │  E6 SPARSE EMBED  │           │ E13 SPLADE EMBED  │
        │  (30,522D sparse) │           │ (30,522D sparse)  │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DUAL SPARSE RECALL                          │
│  ┌─────────────────────┐       ┌─────────────────────┐                  │
│  │ E6 Inverted Index   │       │ E13 Inverted Index  │                  │
│  │ (exact keywords)    │       │ (learned expansion) │                  │
│  │     500 cands       │       │     500 cands       │                  │
│  └──────────┬──────────┘       └──────────┬──────────┘                  │
│             └──────────────┬──────────────┘                             │
│                            ▼                                            │
│                   Union + Dedup → ~800 candidates                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: MATRYOSHKA FILTER                            │
│                      128D dense → 200 candidates                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: 13-SPACE HNSW FUSION                         │
│                                                                         │
│   E1 (35%)   E5 (15%)   E6 (10%)*  E7 (20%)   E10 (15%)  Others...     │
│   semantic   causal     sparse↑    code       multimodal                │
│                         projected                                       │
│                                                                         │
│   * E6 weight dynamically boosted for technical queries                 │
│                                                                         │
│                    RRF Aggregation → 100 candidates                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  STAGE 3.5: E6 TIE-BREAKER (NEW)                         │
│                                                                         │
│   For candidates with similar E1 scores:                                │
│   - Compute E6 term overlap with query                                  │
│   - Add small boost (max 5%) for exact keyword matches                  │
│   - Re-rank close candidates                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: TELEOLOGICAL ALIGNMENT                       │
│                         Topic filtering → 50 candidates                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: E12 COLBERT RE-RANKING                       │
│                      Late interaction → Final 20 results                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Code Changes Summary

| File | Change |
|------|--------|
| `context-graph-core/src/types/fingerprint.rs` | Add `e6_sparse: Option<SparseVector>` |
| `context-graph-embeddings/src/models/pretrained/sparse/model.rs` | Return dual (sparse, dense) |
| `context-graph-storage/src/teleological/indexes/` | Add `E6InvertedIndex` |
| `context-graph-core/src/retrieval/pipeline/default.rs` | Dual Stage 1 recall |
| `context-graph-mcp/src/weights.rs` | Dynamic E6 boost detection |
| `context-graph-core/src/retrieval/distance.rs` | Activate jaccard similarity |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Storage overhead | Low | Low | ~2KB/memory is acceptable |
| Stage 1 latency increase | Medium | Medium | Parallel E6/E13 recall |
| E6 over-boosting degrades general queries | Medium | High | Cap boost at 2x, monitor MRR |
| Backward compatibility | Low | Medium | `e6_sparse: Option<>` for existing |

---

## Conclusion

E6's unique strength is **exact keyword matching** - a capability that complements E1's semantic understanding rather than competing with it. By:

1. **Storing E6 as sparse** (not just projected dense)
2. **Adding E6 to Stage 1 recall** (alongside E13)
3. **Dynamically boosting E6** for technical queries
4. **Using E6 as a tie-breaker** when E1 is uncertain

We can create a retrieval system where:
- **E1 handles "what does this mean?"** (semantic understanding)
- **E6 handles "does this contain X?"** (exact term matching)

This is the intended design of the 13-embedder fingerprint system - each embedder contributing its unique perspective to build a more complete memory representation.

---

## References

1. Constitution v6.0: 13-Embedder Architecture
2. AP-71: Temporal embedders in post-retrieval boost only
3. AP-73: E12 for Stage 3 rerank only
4. AP-74: E13 for Stage 1 recall only
5. ARCH-10: Semantic embedders for divergence detection
6. Pinecone Cascading Retrieval Research
7. ACM TOIS Multi-Stage Fusion Study
