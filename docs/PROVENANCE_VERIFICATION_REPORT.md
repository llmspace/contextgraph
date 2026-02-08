# Provenance Verification Report

**Date**: 2026-02-07
**Branch**: casetrack
**Scope**: Full forensic audit of 57 MCP tools, 43 column families, all provenance subsystems
**Method**: 6 parallel forensic agents examining audit system, embedding/search provenance, entity/causal provenance, storage layer integrity, MCP tool completeness, and dead code

---

## Executive Summary

The context-graph codebase has **strong foundational infrastructure** -- 55/57 MCP tools perform real work, serialization is correct, and core search/storage paths are sound. However, the audit reveals **significant provenance gaps** where infrastructure was built but never wired into production paths. The most critical finding is that **main HNSW indexes are not rebuilt on server restart**, causing search degradation after any restart.

### Severity Breakdown

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 2 | HNSW index loss on restart; custom weight profiles write-only |
| HIGH | 6 | Dead audit variants, missing search audit, phantom CFs, silent data corruption |
| MEDIUM | 8 | Dead code, stale docs, synthetic provenance values, lossy API |
| LOW | 5 | Unused imports, TODO markers, benchmark dead code |

---

## CRITICAL Findings

### CRIT-1: HNSW Indexes Not Rebuilt on Server Restart

**Impact**: After any server restart, MultiSpace and Pipeline search strategies return empty results until all fingerprints are re-stored.

**Details**: The 15 main HNSW indexes (E1-E5, E7-E11, plus causal/multimodal variants) are in-memory only. They are populated as fingerprints are stored but are **not rebuilt from CF_FINGERPRINTS on startup**. Only the causal E11 index has a `rebuild_causal_e11_index()` routine.

**Files**:
- `crates/context-graph-storage/src/teleological/indexes/registry.rs` (index creation)
- `crates/context-graph-storage/src/teleological/rocksdb_store/store.rs:197` (only causal E11 rebuilt)

**Remediation**: Implement `rebuild_all_hnsw_indexes_from_fingerprints()` called on startup, analogous to `rebuild_causal_e11_index()`.

### CRIT-2: Custom Weight Profiles Are Write-Only (BUG-001)

**Impact**: Users can create custom weight profiles via `create_weight_profile`, but these profiles **can never be used** in any search. The data goes into a HashMap that nothing reads.

**Details**:
- `create_weight_profile` writes to `self.custom_profiles.write()` (in-memory HashMap)
- `custom_profiles.read()` is **never called anywhere** in the entire codebase (0 matches)
- When `search_graph` receives a `weightProfile` name, it passes it to the storage layer's `get_weight_profile()` which only knows about the **16 built-in profiles**
- Custom profiles are also session-scoped (lost on restart) since they are never persisted

**Files**:
- `crates/context-graph-mcp/src/handlers/tools/embedder_tools.rs:945-949` (write)
- `crates/context-graph-mcp/src/handlers/core/handlers.rs:113` (HashMap definition)
- `crates/context-graph-storage/src/teleological/rocksdb_store/search.rs:661` (built-in only lookup)

**Remediation**: Add `custom_profiles.read()` lookup in the weight profile resolution path before falling back to built-in profiles. Consider persisting to RocksDB.

---

## HIGH Findings

### HIGH-1: 4 of 13 AuditOperation Variants Are Dead Code (31%)

**Impact**: Audit coverage is incomplete -- 4 operation types are defined but never emitted.

| Variant | Location | Why Dead |
|---------|----------|----------|
| `MemoryRestored` | `audit.rs:336` | No restore/undelete MCP tool exists |
| `TopicDetected` | `audit.rs:363` | `detect_topics` handler never emits audit |
| `EmbeddingRecomputed` | `audit.rs:371` | `store_embedding_version()` never called in production |
| `HookExecuted` | `audit.rs:379` | No hook audit emission infrastructure |

Additionally, `HookExecutionRecord` (audit.rs:157-178) is a 10-field struct that is **never constructed, stored, or read** anywhere.

### HIGH-2: SearchPerformed Audit Covers Only 1 of 6 Search Handlers

**Impact**: The majority of search operations leave no provenance trail.

| Search Handler | Emits SearchPerformed? |
|----------------|----------------------|
| `search_by_intent` | YES (intent_tools.rs:338-362) |
| `search_graph` | **NO** |
| `adaptive_search` | **NO** |
| `search_by_embedder` | **NO** |
| `search_cross_embedder_anomalies` | **NO** |
| `compare_embedder_views` | **NO** |

### HIGH-3: 3 Phantom Column Families (Defined, Opened, Never Used)

**Impact**: RocksDB resources (memory, file handles) consumed with zero functional benefit.

| Column Family | Defined | Opened | Write Code | Read Code |
|---------------|---------|--------|------------|-----------|
| `CF_ENTITY_PROVENANCE` | `column_families.rs:181` | YES | **NONE** | **NONE** |
| `CF_TOOL_CALL_INDEX` | `column_families.rs:272` | YES | **NONE** | **NONE** |
| `CF_CONSOLIDATION_RECOMMENDATIONS` | `column_families.rs:289` | YES | **NONE** | **NONE** |

None of these have trait methods defined. No `store_entity_provenance()`, `store_tool_call_index()`, or `store_consolidation_recommendation()` exist.

Additionally, 3 legacy teleological CFs are also dead:
- `CF_SYNERGY_MATRIX` (column_families.rs:66)
- `CF_TELEOLOGICAL_PROFILES` (column_families.rs:73)
- `CF_TELEOLOGICAL_VECTORS` (column_families.rs:80)

**Total**: 6 dead column families (+ 2 known deprecated: CF_SESSION_IDENTITY, CF_EGO_NODE).

### HIGH-4: store_embedding_version() Never Called in Production

**Impact**: CF_EMBEDDING_REGISTRY is always empty. The system cannot track which model versions produced embeddings.

- Trait method: `store.rs:814`
- RocksDB impl: `provenance_storage.rs:276-313` (fully working)
- Tests: `tests.rs:1253,1283` (passing)
- Production callers: **ZERO**

### HIGH-5: trigger_causal_discovery Has No Audit Trail

**Impact**: The primary causal discovery path (both "extract" and "pairs" modes) stores relationships without any `RelationshipDiscovered` audit records.

- `causal_discovery_tools.rs` has **zero** calls to `append_audit_record`
- By contrast, `discover_graph_relationships` (graph_tools.rs:783-803) correctly emits audit for each relationship
- File: `crates/context-graph-mcp/src/handlers/tools/causal_discovery_tools.rs`

### HIGH-6: Silent Data Corruption in Causal Index

**Impact**: Corrupted `causal_by_source` index entries silently return empty results.

```rust
// causal_relationships.rs:1036
Ok(Some(bytes)) => bincode::deserialize(&bytes).unwrap_or_default(),
```

This returns an empty `Vec<Uuid>` on deserialization failure with **no log message**. A corrupted index entry would cause the system to believe there are no causal relationships for a source fingerprint.

---

## MEDIUM Findings

### MED-1: LLMProvenance Missing in Default "extract" Mode

CausalRelationships created through "extract" mode (the DEFAULT and recommended mode) have `llm_provenance: None`. Only "pairs" mode attaches LLMProvenance. Additionally, there is no `with_llm_provenance()` builder method on `CausalRelationship`.

### MED-2: Entity Provenance Values Are Synthetic

`extraction_method` and `confidence_explanation` in entity DTOs return hardcoded strings:
- `"knowledgeBase"` or `"heuristic"` (based on entity type != Unknown)
- `"Matched against built-in knowledge base"` or `"Detected via capitalization/pattern heuristics"`

These are NOT derived from actual extraction provenance (CF_ENTITY_PROVENANCE is empty).

### MED-3: audit_record_to_json() Loses Structured Data

The `audit_record_to_json()` function (provenance_tools.rs:15-26) serializes AuditOperation variants as Display strings, losing structured enum fields. The `parameters` and `previous_state` fields are omitted entirely.

### MED-4: get_topic_stability Returns Hardcoded entropy=0.0

```rust
// topic_tools.rs:286
let entropy = 0.0_f32; // Entropy is no longer tracked via UTL processor
```

The field is silently zero rather than being removed or marked unavailable.

### MED-5: get_embedder_clusters Returns Error (Not Implemented)

The tool accepts and validates parameters, then returns an error: "requires cuML HDBSCAN on GPU, which is planned but not yet available." (embedder_tools.rs:248-306). Honest about it, but a dead tool in the tool list.

### MED-6: 160 Compiler Warnings

Breakdown by category:
- Dead code: ~100+ (many suppressed with `#[allow(dead_code)]`)
- Deprecated usage: ~10
- Unused imports: ~15
- Unreachable code: ~4
- Unused variables: ~15 (mostly benchmarks)

### MED-7: 5 CF Doc Comments Claim "bincode" When Actual Is JSON

| CF | Comment Line | Claims | Actual |
|----|-------------|--------|--------|
| CF_SOURCE_METADATA | line 100 | bincode | JSON |
| CF_AUDIT_LOG | line 196 | bincode | JSON |
| CF_MERGE_HISTORY | line 234 | bincode | JSON |
| CF_IMPORTANCE_HISTORY | line 248 | bincode | JSON |
| CF_EMBEDDING_REGISTRY | line 301 | bincode | JSON |

### MED-8: Dead Functions in Production Code

| Function | File | Lines |
|----------|------|-------|
| `capture_git_metadata` | code_watcher.rs | 542-583 |
| `get_current_branch` | code_watcher.rs | 584-601 |
| `detect_language` | code_watcher.rs | 602+ |
| `group_entities_by_type` | entity_tools.rs | 1673+ |
| `format_entity_type_distribution` | entity_tools.rs | 1694+ |
| `with_all`, `with_code_pipeline`, `with_graph_linking`, `with_defaults` | handlers.rs | 128-305 |

---

## LOW Findings

### LOW-1: Deprecated Functions Still Called

- `CausalDiscoveryService::new()` -- 4 call sites (3 tests + 1 benchmark)
- `GraphDiscoveryService::with_config()` -- 4 call sites (examples + stubs)
- `recency_boost` field -- still checked in production search path (search.rs:922-926)

### LOW-2: 6 Stalled TODO Markers

All reference M04 milestone tasks (CUDA kernels, traversal utilities, error conversions, inverted indexes).

### LOW-3: Benchmark Crate Has 35+ Warnings

Dead functions, unused variables, unreachable code. Not production-impacting.

### LOW-4: No MCP-Level Test Coverage for Provenance Tools

No dedicated tests for `get_audit_trail`, `get_merge_history`, or `get_provenance_chain` at the MCP handler level. Storage-level tests exist but MCP integration is untested (stub backend always returns empty).

### LOW-5: Silent Audit Degradation

All 14 audit emission sites use `warn!`-and-continue. If CF_AUDIT_LOG is corrupted, all audit silently stops with only warn-level log messages.

---

## What Works Well (Verified Innocent)

| Component | Status | Evidence |
|-----------|--------|----------|
| 55/57 MCP tools | Real implementations | Full dispatch + handler verification |
| AuditRecord write path | Sound | Atomic WriteBatch, JSON serialization, dual-index |
| AuditRecord retrieval | Functional | get_audit_trail by target + time range, not stubbed |
| CF_MERGE_HISTORY | Fully wired | Written during merge, queryable via provenance tools |
| CF_AUDIT_LOG + CF_AUDIT_BY_TARGET | Functional | Proper write + read + index |
| CF_IMPORTANCE_HISTORY | Functional | Written on boost_importance, queryable |
| searchTransparency block | Real computed data | Based on actual weights and strategy |
| Embedder contributions | Real data | Derived from actual embedder_scores arrays |
| Parameter validation | Comprehensive | Fail-fast on invalid inputs across all tools |
| bincode/JSON serialization | Correctly handled | SourceMetadata migrated to JSON, proper fallback |
| All 15 HNSW indexes | Built and queried correctly | (When populated -- see CRIT-1 for restart issue) |
| All 13 quantized embedder CFs | Functional | Read + write verified |
| All 5 code CFs | Functional | Full CRUD verified |
| Both causal CFs | Functional | Write + search + index verified |
| CausalRelationshipRepaired audit | Properly wired | maintenance_tools.rs:34-56 |
| 9/13 AuditOperation variants | Actively emitted | Verified emission sites |

---

## Prioritized Remediation Plan

### P0 -- Critical (Production Correctness)

1. **Implement HNSW index rebuild on startup** from CF_FINGERPRINTS (CRIT-1)
2. **Fix custom weight profile lookup** -- add `custom_profiles.read()` in search resolution path (CRIT-2)

### P1 -- High (Provenance Completeness)

3. **Add SearchPerformed audit** to search_graph, adaptive_search, search_by_embedder, compare_embedder_views, search_cross_embedder_anomalies (HIGH-2)
4. **Add RelationshipDiscovered audit** to trigger_causal_discovery extract + pairs modes (HIGH-5)
5. **Wire store_embedding_version()** into embed_all pipeline (HIGH-4)
6. **Fix silent unwrap_or_default** in causal_relationships.rs:1036 (HIGH-6)
7. **Wire TopicDetected audit** into detect_topics handler (HIGH-1)

### P2 -- Medium (Code Health)

8. **Decide on phantom CFs**: Either implement trait methods + storage for CF_ENTITY_PROVENANCE / CF_TOOL_CALL_INDEX / CF_CONSOLIDATION_RECOMMENDATIONS, or mark them deprecated (HIGH-3)
9. **Remove 3 legacy dead CFs**: CF_SYNERGY_MATRIX, CF_TELEOLOGICAL_PROFILES, CF_TELEOLOGICAL_VECTORS (HIGH-3)
10. **Fix audit_record_to_json()** to include parameters and use structured serialization (MED-3)
11. **Remove dead functions**: capture_git_metadata, get_current_branch, dead Handlers constructors, dead entity utilities (MED-8)
12. **Fix CF doc comments** claiming bincode when actual is JSON (MED-7)
13. **Add LLMProvenance builder method** and wire into extract mode (MED-1)
14. **Remove or deprecate entropy field** from get_topic_stability response (MED-4)

### P3 -- Low (Technical Debt)

15. **Migrate deprecated constructor callers** to with_models() (LOW-1)
16. **Remove deprecated recency_boost** and migrate to temporal_options.temporal_weight (LOW-1)
17. **Clean up unused imports** across MCP and benchmark crates (LOW-3)
18. **Add MCP-level provenance tool tests** (LOW-4)
19. **Clean up benchmark crate warnings** (LOW-3)
20. **Remove or implement get_embedder_clusters** (MED-5)

---

## Column Family Health Matrix (43 Total)

| Status | Count | CFs |
|--------|-------|-----|
| Functional | 29 | CF_FINGERPRINTS, CF_TOPIC_PROFILES, CF_E13_SPLADE_INVERTED, CF_E6_SPARSE_INVERTED, CF_E1_MATRYOSHKA_128, CF_CONTENT, CF_SOURCE_METADATA, CF_FILE_INDEX, CF_TOPIC_PORTFOLIO, CF_E12_LATE_INTERACTION, CF_AUDIT_LOG, CF_AUDIT_BY_TARGET, CF_MERGE_HISTORY, CF_IMPORTANCE_HISTORY, CF_EMB_0 through CF_EMB_12, CF_CODE_ENTITIES, CF_CODE_E7_EMBEDDINGS, CF_CODE_FILE_INDEX, CF_CODE_NAME_INDEX, CF_CODE_SIGNATURE_INDEX, CF_CAUSAL_RELATIONSHIPS, CF_CAUSAL_BY_SOURCE |
| Dormant (impl exists, never called) | 1 | CF_EMBEDDING_REGISTRY |
| Phantom (no impl) | 6 | CF_ENTITY_PROVENANCE, CF_TOOL_CALL_INDEX, CF_CONSOLIDATION_RECOMMENDATIONS, CF_SYNERGY_MATRIX, CF_TELEOLOGICAL_PROFILES, CF_TELEOLOGICAL_VECTORS |
| Known Deprecated | 2 | CF_SESSION_IDENTITY, CF_EGO_NODE |
| **Total** | **38+5(default)** | |

---

## Audit Operation Coverage Matrix (13 Variants)

| Variant | Emission Sites | Status |
|---------|---------------|--------|
| MemoryCreated | memory_tools.rs:457 | Active |
| MemoryMerged | merge.rs:392 | Active |
| MemoryDeleted | curation_tools.rs:96 | Active |
| ImportanceBoosted | curation_tools.rs:257 | Active |
| RelationshipDiscovered | graph_tools.rs:787,934,1005; builder.rs:366 | Active (but NOT in causal_discovery_tools) |
| ConsolidationAnalyzed | consolidation.rs:367 | Active |
| SearchPerformed | intent_tools.rs:340 | Active (but only 1/6 search handlers) |
| CausalRelationshipRepaired | maintenance_tools.rs:35 | Active |
| FileWatcherEvent | file_watcher_tools.rs:391,544 | Active |
| MemoryRestored | -- | **DEAD** |
| TopicDetected | -- | **DEAD** |
| EmbeddingRecomputed | -- | **DEAD** |
| HookExecuted | -- | **DEAD** |

---

## Methodology

Six parallel forensic agents conducted independent investigations:

1. **Audit System Agent**: Traced all AuditOperation variants from definition through emission to storage and retrieval
2. **Embedding/Search Agent**: Verified embedding version tracking, search provenance, weight profiles, and transparency blocks
3. **Entity/Causal Agent**: Audited entity provenance CFs, causal discovery audit trails, LLMProvenance, and merge history
4. **Storage Layer Agent**: Verified all 43 column families for definition/open/write/read status, serialization consistency, and error handling
5. **MCP Tool Agent**: Verified all 57 tools have real handlers, validated parameter handling, and checked for stubs/fakes
6. **Dead Code Agent**: Comprehensive sweep for dead functions, deprecated usage, TODO markers, compiler warnings, and test gaps

All findings are backed by specific file paths and line numbers. Confidence: HIGH.
