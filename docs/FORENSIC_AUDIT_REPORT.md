# Context Graph Forensic Audit Report

**Date:** 2026-02-15
**Branch:** casetrack
**Scope:** Full system integrity — build, tests, MCP handlers, embedders, search, storage, persistence
**Method:** 4 parallel forensic investigations covering all layers

---

## Executive Summary

The Context Graph codebase is **structurally sound** — no phantom embedders, no fake computations, no stub handlers. All 55 MCP tools have real implementations, all 13 embedders compute genuine vectors, all 4 search strategies perform real work, and all 51 column families are registered and opened.

However, the audit uncovered **31 findings** across 4 severity levels that create a pattern of **silent degradation** — things that appear to work but fail invisibly.

| Severity | Count | Theme |
|----------|-------|-------|
| CRITICAL | 3 | Failing tests, incomplete flush, silent search degradation |
| HIGH | 3 | Error swallowing patterns (29 instances + 2 specific tools) |
| MEDIUM | 10 | Validation gaps, incomplete operations, misleading docs |
| LOW | 15 | Stale comments, dead code, cosmetic issues |

---

## CRITICAL Findings

### C-1: 12 Failing Tests (5 Wrong Assertions + 2 Race Conditions + 5 Flaky)

**Impact:** Test suite gives false confidence. `cargo test` fails.

#### 5 MCP Handler Tests Assert the Wrong Thing

| Test | File |
|------|------|
| `test_fsv_boost_importance_not_found` | `crates/context-graph-mcp/src/handlers/tests/curation_tools_fsv.rs:460` |
| `test_fsv_forget_concept_not_found` | `crates/context-graph-mcp/src/handlers/tests/curation_tools_fsv.rs:160` |
| `test_detect_topics_force_parameter` | `crates/context-graph-mcp/src/handlers/tests/topic_tools.rs:316` |
| `test_detect_topics_insufficient_memories` | `crates/context-graph-mcp/src/handlers/tests/topic_tools.rs:284` |
| `test_fsv_detect_topics_insufficient_memories` | `crates/context-graph-mcp/src/handlers/tests/topic_tools_fsv.rs:104` |

**Root Cause (double contradiction):**
1. Tests assert `response.error.is_some()` but `tool_error_typed()` returns `JsonRpcResponse::success(...)` with `isError: true` in the result body. The `.error` field is always `None`.
2. Even if checking the right place, tests assert wrong error codes: curation tests expect `FINGERPRINT_NOT_FOUND` (-32010) but `ToolErrorKind::NotFound` maps to `NODE_NOT_FOUND` (-32002). Topic tests expect `INSUFFICIENT_MEMORIES` (-32021) but `ToolErrorKind::Validation` maps to `INVALID_PARAMS` (-32602).

**Fix:** Rewrite assertions to check `response.result["isError"]` and correct the error codes.

#### 2 Causal-Agent Tests: LLM Singleton Race Condition

| Test | File |
|------|------|
| `test_2_1_causal_content_analysis` | `crates/context-graph-causal-agent/tests/analyze_single_text_integration.rs:50` |
| `test_2_3_effect_direction_content` | `crates/context-graph-causal-agent/tests/analyze_single_text_integration.rs:135` |

**Root Cause:** Multiple tests call `llama_backend_init()` in parallel. The C++ backend is global state — `BackendAlreadyInitialized` error every time.

**Fix:** Add `#[serial]` from `serial_test` crate, or use a `std::sync::Once` guard.

#### 5 CLI E2E Tests: Flaky Under Parallel Execution

All in `crates/context-graph-cli/tests/e2e/error_recovery_test.rs`. Pass in isolation, fail when GPU/VRAM is contended by parallel crate tests.

**Fix:** Mark with `#[serial]` or gate behind a `--test-threads=1` CI flag.

---

### C-2: `flush_async()` Only Flushes 33 of 51 Column Families

**File:** `crates/context-graph-storage/src/teleological/rocksdb_store/persistence.rs:223-266`

Only flushes teleological (20) + quantized (13) CFs. **Missing:**
- 11 base CFs (including `CF_SYSTEM` which stores soft-delete markers)
- 5 code CFs
- 2 causal CFs

**Impact:** On crash, soft-delete markers and causal relationships may not be persisted. This undermines the soft-delete persistence fix (which correctly writes to `CF_SYSTEM` but `CF_SYSTEM` is never explicitly flushed).

The same 33/51 gap affects:
- `health_check()` at `store.rs:1036-1044` — only checks 33 CFs
- `compact_async()` at `persistence.rs:364-373` — only compacts 33 CFs
- `storage_size_bytes_internal()` at `persistence.rs:181-207` — undercounts disk usage

**Fix:** Iterate ALL CFs from `get_all_column_family_descriptors()` in flush/health_check/compact/storage_size.

---

### C-3: `search_graph` Silently Drops Content on Retrieval Failure

**File:** `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs:1122-1156`

When `includeContent=true` and content/metadata batch retrieval fails, the handler logs a `warn!()` and returns `null` for all content fields — wrapped in a **successful** MCP response. The caller cannot distinguish "no content stored" from "content retrieval failed."

**Contrast:** `code_tools.rs:194-215` and `graph_link_tools.rs:205-216` correctly return `tool_error()` on the same failure.

**Impact:** `search_graph` is the most-called tool. Silent degradation means downstream agents silently lose context.

**Fix:** Return `tool_error()` on batch retrieval failure, matching existing patterns in other tools.

---

## HIGH Findings

### H-1: 29 Instances of Silent Serialization Failure

**Pattern:** `serde_json::to_value(response).unwrap_or_else(|_| json!({}))`

Returns `{}` (empty JSON) as a **successful** MCP response when response DTO serialization fails. No error logged, no error reported. Found in:

- `graph_link_tools.rs` (4 instances)
- `graph_tools.rs` (5 instances)
- `embedder_tools.rs` (8 instances)
- `entity_tools.rs` (8 instances)
- `causal_tools.rs` (3 instances)
- `code_tools.rs` (1 instance)
- `robustness_tools.rs` (1 instance)
- `keyword_tools.rs` (1 instance)

**Practical risk:** Low (would require `f64::NAN` in a response field). But 12 other tools already use the correct `match` + `tool_error()` pattern. The inconsistency is the real problem.

**Fix:** Replace all 29 with `match serde_json::to_value(response) { Ok(v) => ..., Err(e) => self.tool_error(...) }`.

### H-2: `validate_knowledge` Returns Empty Evidence on Silent Failure

**File:** `crates/context-graph-mcp/src/handlers/tools/entity_tools.rs:1302-1347`

Triple-nested `if let Ok(...)` pattern:
```
if let Ok(fingerprint) = embed() {
    if let Ok(candidates) = search() {
        if let Ok(contents) = get_content() {
            // populate evidence
        }
    }
}
```

If any step fails, the response has `confidence: 0.5, supporting: [], contradicting: []` — which reads as "no evidence found" rather than "search failed." A caller asking `validate_knowledge("the earth is flat")` gets an ambiguous non-answer.

**Fix:** Replace with explicit error propagation or include a `search_failed: true` field.

### H-3: 3 Graph-Agent Examples Fail to Compile

**Files:**
- `crates/context-graph-graph-agent/examples/benchmark_graph.rs`
- `crates/context-graph-graph-agent/examples/benchmark_graph_code.rs`
- `crates/context-graph-graph-agent/examples/benchmark_graph_multi.rs`

Import `GraphDiscoveryConfig`/`GraphDiscoveryService` which are behind `#[cfg(feature = "llm")]`, but no `required-features = ["llm"]` in Cargo.toml.

**Fix:** Add `[[example]]` sections with `required-features = ["llm"]` to Cargo.toml.

---

## MEDIUM Findings

### M-1: Unknown Strategy/Parameter Values Silently Default

**File:** `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`

| Parameter | Unknown Value Becomes | Line |
|-----------|-----------------------|------|
| `strategy` | `MultiSpace` | 575-580 |
| `decayFunction` | `Linear` | 673-679 |
| `temporalScale` | `Meso` | 738-746 |
| `sessionScope` | `"all"` | 940-947 |

A typo like `"pipline"` silently becomes `MultiSpace`. **Contrast:** `trigger_consolidation` and `search_causal_relationships` correctly validate and return errors.

### M-2: LLM-Gated Tools Appear in `tools/list` But Always Fail

**Files:** `graph_tools.rs:559-576`, `causal_discovery_tools.rs:37-54`

4 tools (`discover_graph_relationships`, `validate_graph_link`, `trigger_causal_discovery`, `get_causal_discovery_status`) appear in the tool list when `llm` feature is disabled, but always return errors. MCP callers waste attempts.

### M-3: Quantized CFs Misleadingly Described as Search Acceleration

13 quantized CFs (`emb_0` through `emb_12`) exist and store compressed embeddings. But `search.rs` **never reads from them**. All search uses full-precision vectors from `CF_FINGERPRINTS` + HNSW indexes.

The quantized path is real infrastructure for storage compression (17KB vs 63KB), but the claim of "search acceleration" is incorrect.

### M-4: Pipeline Search Lacks E12 MaxSim Reranking

The documented pipeline is "E13 -> E1 -> E12 (MaxSim rerank)." The actual RocksDB pipeline does E13+E1+E5+E7+E8+E11 recall -> multi-space scoring. **E12 MaxSim reranking is not implemented** (noted as AP-74 TODO in code comments).

### M-5: E10 Labeled "Multimodal/CLIP" But Is Text-Only

`ModelId` enum maps E10 as `Multimodal` and documentation references "CLIP." The actual implementation at `crates/context-graph-embeddings/src/models/pretrained/contextual/model.rs` loads **e5-base-v2** — a text-only model. Not a functional bug (it works correctly as a text model), but documentation is misleading.

### M-6: No Automatic GC Scheduling for Soft-Deleted Entries

`gc_soft_deleted()` exists at `crates/context-graph-storage/src/teleological/rocksdb_store/crud.rs:274-330` but requires manual invocation via MCP maintenance tools. No background task, no timer. Soft-deleted entries accumulate unboundedly in DashMap if GC is never triggered.

### M-7: No Checkpoint Cleanup

`checkpoint_async()` creates full database copies at `{db_path}/checkpoints/checkpoint_{timestamp}/`. No rotation or cleanup mechanism exists. Disk space leak over time.

### M-8: Silent Relationship Skipping in Causal Tools

**Files:** `causal_tools.rs:242,587`, `causal_relationship_tools.rs:334,539,608-615`

Failed or corrupted causal relationships are silently omitted from results via `if let Ok(Some(rel))` patterns. Result counts silently shrink without warning.

### M-9: Silent Cluster/Consolidation Degradation

- `get_embedder_clusters` at `embedder_tools.rs:314,323,370`: Silent scan, insert, and content failures
- `trigger_consolidation` at `consolidation.rs:280-296`: Content retrieval failure produces empty texts; consolidation proceeds with no text for analysis

### M-10: Stale Doc Comments Claim Wrong CF Counts

| Location | Claims | Actual |
|----------|--------|--------|
| `teleological/column_families.rs:1013` | "19 teleological" | 20 |
| `teleological/column_families.rs:1098` | "32 descriptors" | 33 |
| `teleological/column_families.rs:1435` | "39 descriptors" | 40 |
| `teleological/rocksdb_store/mod.rs:2` | "17 column families" | 51 |
| `teleological/rocksdb_store/store.rs:183` | "39 total" | 51 |
| Doc example assertion `assert_eq!(descriptors.len(), 32)` | Would FAIL if run | Should be 33 |

---

## LOW Findings

### L-1: Phantom `test-mode` Feature Never Enabled

Declared in `context-graph-causal-agent/Cargo.toml:18` and `context-graph-graph-agent/Cargo.toml:21`. Used in 15+ `#[cfg(feature = "test-mode")]` blocks. **Never enabled by any crate, CI config, or build profile.** All gated code is dead.

### L-2: Dead Public Functions Suppressed with `#[allow(dead_code)]`

| Function | File |
|----------|------|
| `Handlers::with_all()` | `crates/context-graph-mcp/src/handlers/core/handlers.rs:145` |
| `Handlers::with_code_pipeline()` | same file:194 |

Both `pub fn`, both `#[allow(dead_code)]`, both called from nowhere.

### L-3: E8 Module Doc Says 384D, Code Enforces 1024D

`crates/context-graph-embeddings/src/models/pretrained/graph/mod.rs:8` says "384D" but `constants.rs:16` correctly defines `GRAPH_DIMENSION: usize = 1024`. Stale docstring.

### L-4: 3 Deprecated CFs Created But Never Written

| CF | Purpose |
|----|---------|
| `CF_ENTITY_PROVENANCE` | Trait methods not wired |
| `CF_TOOL_CALL_INDEX` | Trait methods not wired |
| `CF_CONSOLIDATION_RECOMMENDATIONS` | Trait methods not wired |

All three have comments saying "DEPRECATED: Kept in open list for RocksDB compat."

### L-5: No Audit Log Rotation

`CF_AUDIT_LOG` is append-only by design ("NO update or delete"). No size limit, no pruning, no rotation. Grows unboundedly over time.

### L-6: Topic Portfolio Accumulation

Old session portfolios in `CF_TOPIC_PORTFOLIO` are never deleted. Individual portfolios are small (1-50KB), but they accumulate per-session forever.

### L-7: SyntheticProvider Tests Assert Identical Vectors

`crates/context-graph-benchmark/src/causal_bench/provider.rs:352`: Asserts `cause == effect` for the synthetic stub. The asymmetric E5 behavior has **no automated test coverage** in the benchmark suite.

### L-8: Benchmark `test-utils` Feature Is Redundant

`context-graph-benchmark/Cargo.toml` declares a `test-utils` feature, but `context-graph-core` is already included with `features = ["test-utils"]` in the dependencies. The feature appears optional but is always enabled.

### L-9: Empty `queryClassification` in Provenance Output

`memory_tools.rs:1178-1185`: The `queryClassification` field in search provenance is always `{ detected_type: "", detection_patterns: [] }`. Cosmetic/unfinished.

### L-10: 20+ Copies of `cosine_similarity`

2 canonical implementations in `context-graph-core`. 18+ private copies in benchmark binaries. Acceptable for self-contained benchmarks but increases maintenance burden.

### L-11: `storage_size_bytes_internal()` Undercounts Disk Usage

Only counts 33/51 CFs. Missing base, code, and causal CF sizes.

### L-12: Layer Status Degradation Poorly Structured

`status_tools.rs:36-70`: Layer failures return the string `"error"` with no error code, no structured error, and no indication of which layer or why.

### L-13: `NodeMetadata` Has `skip_serializing_if` — Serialization Path Unknown

`crates/context-graph-core/src/types/memory_node/metadata.rs:50-106` has multiple `skip_serializing_if` annotations. If the MemoryNode system serializes with bincode, this is a latent bug. Requires verification.

### L-14: Causal Gate Is No-Op Without Trained LoRA Weights

Without trained E5 LoRA weights, E5 produces near-uniform scores (0.93-0.98) that all exceed `CAUSAL_THRESHOLD=0.04`, making the gate boost everything equally. Functionally a no-op in base model mode.

### L-15: Unstaged P6/P7/P8 Optimizations

3 files modified but not committed. All compile cleanly and appear correct:
- `causal-agent/src/service/mod.rs` — test refactoring
- `memory_tools.rs` — P6 weight resolution hoisting
- `search.rs` — P7 `HashMap::remove()`, P8 `multi_get_cf` batch read

---

## Recommended Fix Priority

### Immediate (blocks CI / data integrity)

1. **Fix 5 MCP handler test assertions** — check `result["isError"]` not `response.error`, correct error codes
2. **Fix `flush_async`/`health_check`/`compact_async`** — iterate ALL 51 CFs, especially `CF_SYSTEM`
3. **Fix LLM singleton race** — add `#[serial]` or `Once` guard to causal-agent integration tests
4. **Add `required-features = ["llm"]`** to 3 graph-agent examples

### Short-term (silent degradation)

5. **`search_graph` content failure** — return `tool_error()` instead of silent null
6. **`validate_knowledge` evidence failure** — report search failure explicitly
7. **Replace 29 `unwrap_or_else(|_| json!({}))`** with `match` + `tool_error()`
8. **Validate strategy/parameter strings** — return errors for unknown values

### Medium-term (operational hygiene)

9. **Automatic GC scheduling** — background task for soft-delete cleanup
10. **Checkpoint rotation** — limit number/age of checkpoints
11. **Fix stale CF count docs** (19->20, 32->33, 39->40, etc.)
12. **Exclude LLM-gated tools** from `tools/list` when feature disabled

### Low-priority (cleanup)

13. Remove or implement 3 deprecated CFs
14. Fix E10 "Multimodal/CLIP" naming
15. Fix E8 384D stale docstring
16. Audit log size governance
17. Commit or stash P6/P7/P8 optimizations

---

## Positive Findings

These areas passed forensic inspection:

- **All 55 MCP tools** have real, non-stub implementations
- **All 13 embedders** compute genuine vectors via GPU forward passes or mathematical computation
- **All 3 asymmetric embedders** (E5, E8, E10) produce genuinely different dual vectors
- **All 4 search strategies** perform real multi-embedder retrieval
- **15 HNSW indexes** are built, persisted to `CF_HNSW_GRAPHS`, and actively queried
- **RRF fusion** genuinely combines multi-embedder scores
- **Soft-delete persistence** correctly uses `CF_SYSTEM` with restart recovery
- **SourceMetadata** correctly uses JSON (not bincode) — no `skip_serializing_if` conflict
- **All 51 CFs** are registered and opened — no missing CF panics
- **Release build** succeeds with 0 warnings, 0 errors
- **Causal gate** thresholds and boost/demotion are correctly wired into search results
- **No TODO/FIXME/HACK** debt in storage source code
- **No secrets or credentials** found in source files
