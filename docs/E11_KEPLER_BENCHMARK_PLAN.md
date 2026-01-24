# E11 KEPLER Integration & Benchmark Testing Plan

**Version**: 1.0
**Date**: 2026-01-24
**Status**: Active Development
**Branch**: `benchmark`

---

## Executive Summary

This document provides the optimal integration and benchmark testing plan for E11 (V_factuality) using the KEPLER model (RoBERTa-base + TransE on Wikidata5M, 768D). E11 is a RELATIONAL_ENHANCER per ARCH-12 that provides entity knowledge retrieval capabilities that E1 (semantic foundation) misses.

**Key Value Proposition**: E11 finds entities like "Diesel" knowing it IS a database ORM, while E1 only matches semantically on "database" or "Rust".

---

## 1. Current Implementation State

### 1.1 Git History (Recent E11 Changes)

| Commit | Description |
|--------|-------------|
| `41a7f76` | Simplified E11 to use KEPLER's native entity embeddings instead of keyword lookup |
| `b99adab` | Added retrieval pipeline stages, KEPLER tensor fixes, search_code/search_by_keywords tools |
| `808b6d6` | Initial E11 entity embedder with KePLER model and benchmark suite |

### 1.2 File Structure

```
crates/context-graph-benchmark/
├── src/
│   ├── datasets/
│   │   ├── e11_entity.rs          # E11 dataset structures & loaders
│   │   ├── e1_semantic.rs         # E1 benchmark dataset (reference)
│   │   └── mod.rs
│   ├── metrics/
│   │   ├── e11_entity.rs          # E11 metrics & thresholds
│   │   ├── e1_semantic.rs         # E1 metrics (reference)
│   │   └── mod.rs
│   ├── runners/
│   │   ├── e11_entity.rs          # E11 benchmark runner
│   │   ├── e1_semantic.rs         # E1 runner (reference)
│   │   └── mod.rs
│   └── bin/
│       ├── e11_entity_bench.rs    # E11 CLI entry point
│       ├── e1_semantic_bench.rs   # E1 CLI (reference)
│       └── e10_intent_bench.rs    # E10 CLI (reference pattern)
```

### 1.3 Available Datasets

| Directory | Description | Format |
|-----------|-------------|--------|
| `data/beir_scifact/` | BEIR SciFact scientific claims (5K docs, 300 queries) | chunks.jsonl, qrels.json |
| `data/hf_benchmark_diverse/` | HuggingFace diverse benchmark | chunks.jsonl |
| `data/e10_benchmark/` | E10 intent benchmark synthetic data | jsonl |
| `data/semantic_benchmark/` | Semantic similarity benchmark | jsonl |

---

## 2. Benchmark Architecture

### 2.1 E11 Benchmark Suite Structure

Following the established pattern from E1 and E10 benchmarks:

```
┌─────────────────────────────────────────────────────────────┐
│                    E11 ENTITY BENCHMARK                      │
├─────────────────────────────────────────────────────────────┤
│  Benchmark A: Entity Extraction                              │
│  ├── Tests: extract_entities MCP tool                       │
│  ├── Metrics: Precision, Recall, F1, Canonicalization       │
│  └── Target: F1 >= 0.85, Canon >= 0.90                      │
├─────────────────────────────────────────────────────────────┤
│  Benchmark B: Entity Retrieval                               │
│  ├── Tests: E1-only vs E11-only vs E1+E11 hybrid            │
│  ├── Metrics: MRR, P@K, E11 Contribution %                  │
│  └── Target: E11 Contribution >= +10%                       │
├─────────────────────────────────────────────────────────────┤
│  Benchmark C: TransE Relationship Inference                  │
│  ├── Tests: infer_relationship (h + r ≈ t)                  │
│  ├── Metrics: Valid/Invalid scores, Separation, MRR         │
│  └── Target: Valid > -5.0, Invalid < -10.0, Sep > 5.0       │
├─────────────────────────────────────────────────────────────┤
│  Benchmark D: Knowledge Validation                           │
│  ├── Tests: validate_knowledge triple scoring               │
│  ├── Metrics: Accuracy, Confusion Matrix                    │
│  └── Target: Accuracy >= 75%                                │
├─────────────────────────────────────────────────────────────┤
│  Benchmark E: Entity Graph                                   │
│  ├── Tests: get_entity_graph construction                   │
│  ├── Metrics: Nodes, Edges, Density                         │
│  └── Target: Informational                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Target Thresholds (KEPLER-Calibrated)

From `metrics/e11_entity.rs`:

| Metric | Target | Priority | Notes |
|--------|--------|----------|-------|
| Entity Extraction F1 | >= 0.85 | Critical | NER + canonicalization |
| Canonicalization Accuracy | >= 0.90 | High | postgres → postgresql |
| TransE Valid Score | > -5.0 | Critical | KEPLER: valid triples |
| TransE Invalid Score | < -10.0 | Critical | KEPLER: invalid triples |
| Score Separation | > 5.0 | Critical | valid_avg - invalid_avg |
| Relation Inference MRR | >= 0.40 | High | Rank of correct relation |
| Knowledge Validation Acc | >= 75% | High | Binary classification |
| E11 Contribution | >= +10% | Medium | vs E1-only baseline |
| Entity Search Latency | < 100ms | High | P95 latency |

---

## 3. Optimal Testing Strategy

### 3.1 Phase 1: Dataset Preparation

**Goal**: Ensure high-quality entity-annotated data for benchmarking.

#### 3.1.1 Use BEIR SciFact (Primary)

The BEIR SciFact dataset is ideal for E11 testing because:
- Contains scientific entities (proteins, genes, diseases)
- Has relevance judgments (qrels.json) for retrieval evaluation
- Ground truth available for claim verification

```bash
# SciFact structure
data/beir_scifact/
├── chunks.jsonl     # 5K+ document chunks
├── queries.jsonl    # Query set
├── qrels.json       # Relevance judgments
└── doc_id_map.json  # ID mappings
```

#### 3.1.2 Create Entity-Annotated Ground Truth

**Action Items**:
1. Extract entities from SciFact using KEPLER
2. Build ground truth entity→document mappings
3. Create valid/invalid knowledge triples from scientific facts
4. Generate entity pairs with known relationships

```python
# Example ground truth structure
{
    "entities": {
        "BRCA1": {"type": "Gene", "synonyms": ["BRCA-1", "breast cancer 1"]},
        "p53": {"type": "Protein", "synonyms": ["TP53", "tumor protein p53"]}
    },
    "triples": {
        "valid": [
            {"h": "BRCA1", "r": "associated_with", "t": "breast_cancer"},
            {"h": "p53", "r": "regulates", "t": "apoptosis"}
        ],
        "invalid": [
            {"h": "BRCA1", "r": "causes", "t": "COVID-19"},
            {"h": "p53", "r": "is_a", "t": "database"}
        ]
    }
}
```

### 3.2 Phase 2: Benchmark Execution

#### 3.2.1 Run Order

```bash
# 1. Extraction benchmark (no GPU warmup needed)
cargo run -p context-graph-benchmark --bin e11-entity-bench \
    --release --features real-embeddings -- \
    --data-dir data/beir_scifact \
    --benchmark extraction \
    --max-chunks 1000

# 2. Full benchmark suite (requires warm GPU)
cargo run -p context-graph-benchmark --bin e11-entity-bench \
    --release --features real-embeddings -- \
    --data-dir data/beir_scifact \
    --benchmark all \
    --max-chunks 1000 \
    --num-queries 100 \
    --output benchmark_results/e11_scifact.json
```

#### 3.2.2 Benchmark Comparison Matrix

Run benchmarks across multiple datasets and configurations:

| Dataset | Chunks | Queries | Extraction | Retrieval | TransE | Validation |
|---------|--------|---------|------------|-----------|--------|------------|
| SciFact | 1000 | 100 | ✅ | ✅ | ✅ | ✅ |
| SciFact | 5000 | 300 | ✅ | ✅ | ✅ | ✅ |
| HF Diverse | 1000 | 100 | ✅ | ✅ | ✅ | ✅ |
| Semantic | 500 | 50 | ✅ | ✅ | ❌ | ❌ |

### 3.3 Phase 3: Comparative Analysis

#### 3.3.1 E11 vs Other Embedders

Compare E11 contribution against baseline and other enhancers:

```
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVAL MRR COMPARISON                        │
├─────────────────────────────────────────────────────────────┤
│  E1-only (baseline)          │  0.XX                        │
│  E1 + E11 (entity)           │  0.XX (+YY%)                 │
│  E1 + E10 (intent)           │  0.XX (+YY%)                 │
│  E1 + E5 (causal)            │  0.XX (+YY%)                 │
│  E1 + E11 + E10 (combined)   │  0.XX (+YY%)                 │
│  Full MultiSpace             │  0.XX (+YY%)                 │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3.2 Query Type Analysis

Analyze E11 contribution by query type:

| Query Type | E1 MRR | E11 MRR | Hybrid MRR | E11 Contribution |
|------------|--------|---------|------------|------------------|
| Entity-focused | ? | ? | ? | Expected: High |
| Semantic-only | ? | ? | ? | Expected: Low |
| Mixed | ? | ? | ? | Expected: Medium |

---

## 4. Integration with MCP Tools

### 4.1 E11 MCP Tools

From `constitution.yaml`:

```yaml
entity:
  - extract_entities        # Extract entities from text
  - search_by_entities      # Search memories by entities
  - infer_relationship      # TransE: h + r ≈ t
  - find_related_entities   # Find entities with relationship
  - validate_knowledge      # Validate triples
  - get_entity_graph        # Build entity graph
```

### 4.2 Tool Testing Matrix

| Tool | Benchmark | Test Cases | Coverage |
|------|-----------|------------|----------|
| `extract_entities` | A | Entity detection, canonicalization | ✅ |
| `search_by_entities` | B | Retrieval with entity filter | ✅ |
| `infer_relationship` | C | TransE scoring | ✅ |
| `find_related_entities` | C | Relation traversal | ⚠️ Partial |
| `validate_knowledge` | D | Triple validation | ✅ |
| `get_entity_graph` | E | Graph construction | ✅ |

### 4.3 Hook Integration

The E11 is integrated into the user_prompt_submit hook:

```rust
// From user_prompt_submit.rs
// 1. Extract entities from user prompt
// 2. Search with E11 to find what E1 misses
// 3. Separate budget (3 memories) ensures E11 contributions visible
```

---

## 5. Known Issues & Improvements

### 5.1 Current Limitations

| Issue | Impact | Priority | Resolution |
|-------|--------|----------|------------|
| Simple entity extraction (capitalized words) | Low recall | High | Implement NER model |
| Synthetic knowledge triples | May not match real KB | Medium | Use Wikidata triples |
| No entity linking disambiguation | Ambiguity | Medium | Implement ARCH-20 |
| KEPLER thresholds not validated | Incorrect targets | High | Run with real model |

### 5.2 Improvement Roadmap

1. **Entity Extraction Enhancement**
   - Replace simple capitalization with spaCy/BERT NER
   - Add domain-specific entity recognition (programming terms)
   - Improve canonicalization with fuzzy matching

2. **Knowledge Base Integration**
   - Load real Wikidata5M triples for validation
   - Create technical domain KB (programming, databases)
   - Add entity type hierarchy

3. **TransE Calibration**
   - Run actual KEPLER embeddings to establish real thresholds
   - Compare MiniLM vs KEPLER score distributions
   - Tune optimal validation threshold

---

## 6. Execution Commands

### 6.1 Quick Start

```bash
# Run extraction benchmark only (fast, no GPU)
cargo run -p context-graph-benchmark --bin e11-entity-bench \
    --release --features real-embeddings -- \
    --data-dir data/beir_scifact \
    --benchmark extraction \
    --max-chunks 500

# Run full benchmark (requires GPU)
cargo run -p context-graph-benchmark --bin e11-entity-bench \
    --release --features real-embeddings -- \
    --data-dir data/beir_scifact \
    --benchmark all \
    --max-chunks 1000 \
    --num-queries 100 \
    --output benchmark_results/e11_full.json
```

### 6.2 Comparison Benchmarks

```bash
# E1 semantic baseline
cargo run -p context-graph-benchmark --bin e1-semantic-bench \
    --release --features real-embeddings -- \
    --data-dir data/beir_scifact

# E10 intent baseline
cargo run -p context-graph-benchmark --bin e10-intent-bench \
    --release --features real-embeddings -- \
    --data-dir data/e10_benchmark
```

### 6.3 MCP Tool Testing

```bash
# Test extract_entities
context-graph-cli mcp call extract_entities \
    --text "Rust uses Tokio for async and Diesel for PostgreSQL"

# Test infer_relationship
context-graph-cli mcp call infer_relationship \
    --head-entity "Tokio" \
    --tail-entity "Rust"

# Test validate_knowledge
context-graph-cli mcp call validate_knowledge \
    --subject "Django" \
    --predicate "uses" \
    --object "Python"
```

---

## 7. Success Criteria

### 7.1 Benchmark Targets

| Metric | Target | Status |
|--------|--------|--------|
| Entity Extraction F1 | >= 0.85 | ⏳ Pending |
| Canonicalization | >= 0.90 | ⏳ Pending |
| TransE Valid Score | > -5.0 | ⏳ Pending |
| TransE Invalid Score | < -10.0 | ⏳ Pending |
| E11 Contribution | >= +10% | ⏳ Pending |
| Entity Search P95 | < 100ms | ⏳ Pending |

### 7.2 Integration Targets

| Integration | Target | Status |
|-------------|--------|--------|
| Hook integration working | E11 in user_prompt_submit | ✅ Complete |
| MCP tools functional | All 6 entity tools | ⏳ Testing |
| Benchmark reproducible | Deterministic with seed | ✅ Complete |
| Results persisted | JSON output | ✅ Complete |

---

## 8. Next Steps

### Immediate (This Sprint)

1. [ ] Run extraction benchmark on SciFact to establish baseline
2. [ ] Validate KEPLER thresholds with real model
3. [ ] Fix any runtime errors in e11_entity_bench
4. [ ] Compare E11 contribution vs E1-only

### Short Term (Next 2 Sprints)

1. [ ] Improve entity extraction with NER
2. [ ] Add real Wikidata triples for validation
3. [ ] Run full comparison matrix
4. [ ] Document optimal blend weights

### Medium Term

1. [ ] Implement ARCH-20 entity linking
2. [ ] Add technical domain KB
3. [ ] Integrate with topic detection
4. [ ] Performance optimization

---

## Appendix A: E11 Constitution Reference

From `CLAUDE.md`:

```yaml
E11:
  name: V_factuality
  dim: 768
  finds: "Entity knowledge (Diesel=database ORM)"
  model: "KEPLER (RoBERTa-base + TransE)"
  category: RELATIONAL_ENHANCER
  topic_weight: 0.5
```

## Appendix B: Related Files

| File | Purpose |
|------|---------|
| `crates/context-graph-benchmark/src/datasets/e11_entity.rs` | Dataset structures |
| `crates/context-graph-benchmark/src/metrics/e11_entity.rs` | Metrics & thresholds |
| `crates/context-graph-benchmark/src/runners/e11_entity.rs` | Benchmark runner |
| `crates/context-graph-benchmark/src/bin/e11_entity_bench.rs` | CLI entry point |
| `crates/context-graph-core/src/entity/mod.rs` | Entity module |
| `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs` | MCP handlers |
| `crates/context-graph-cli/src/commands/hooks/user_prompt_submit.rs` | Hook integration |
| `docs2/constitution.yaml` | Architecture rules |
| `docs2/contextprd.md` | PRD specification |
