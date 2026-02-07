# Manual Test Plan: Embedder Navigation Gap Fixes

**Date**: 2026-02-07
**Branch**: casetrack
**Scope**: Gaps 1, 3, 5, 6, 8 from EMBEDDER_NAVIGATION_GAPS.md
**Tool count**: 54 MCP tools (was 53 — added `get_memory_fingerprint`)

---

## Prerequisites

### 1. Start the MCP Server

```bash
# Build release binary
cargo build --release -p context-graph-mcp

# Start server on TCP port 3100 with clean database
rm -rf /tmp/context-graph-test-gaps
CONTEXT_GRAPH_DATA_DIR=/tmp/context-graph-test-gaps \
CONTEXT_GRAPH_TCP_PORT=3100 \
RUST_LOG=info,context_graph_mcp=debug \
  ./target/release/context-graph-mcp --port 3100
```

### 2. TCP Client Helper

All tests use raw TCP JSON-RPC to port 3100. Each command is a single JSON line terminated by `\n`.

```bash
# Helper function: send JSON-RPC request and read response
send_mcp() {
  local payload="$1"
  echo "$payload" | nc -w 5 127.0.0.1 3100
}
```

**IMPORTANT**: After each timeout or error, reconnect (open new `nc` connection) to clear stale buffer data. Do NOT reuse a connection after a timeout.

### 3. Synthetic Test Data

We seed the database with 5 carefully chosen memories that exercise different embedder strengths. Each memory is designed so we KNOW which embedders should score highest.

---

## Phase 0: Seed Test Database

Store 5 memories with known characteristics. After each store, **physically verify** the memory exists by retrieving it.

### Memory 1: Pure Semantic (E1 dominant)

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"The theory of relativity describes how space and time are intertwined. Einstein showed that massive objects cause a distortion in spacetime, which is felt as gravity.","importance":0.8,"tags":["physics","einstein","gravity"]}}}
```

**Expected**: E1 (semantic) scores highest. E5 (causal) moderate (cause-effect present). E7 (code) near zero.

**Verification**: Record returned `fingerprintId` as `$MEM1_ID`. Then retrieve:
```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM1_ID","includeVectorNorms":true}}}
```
**Check**: `embeddersPresent >= 10`, E1 has non-zero `l2Norm`, E7 `l2Norm` should be small or zero.

### Memory 2: Code Pattern (E7 dominant)

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"pub async fn embed_all(&self, content: &str) -> CoreResult<MultiArrayEmbeddingOutput> {\n    let start = Instant::now();\n    let fingerprint = self.provider.embed_all(content).await?;\n    Ok(MultiArrayEmbeddingOutput { fingerprint, total_latency: start.elapsed() })\n}","importance":0.9,"modality":"code","tags":["rust","embedding","async"]}}}
```

**Expected**: E7 (code) scores highest. E1 moderate. E6 (keyword) picks up function names.

**Verification**: Record `$MEM2_ID`. Get fingerprint. Check E7 has largest `l2Norm` of dense embedders.

### Memory 3: Causal Relationship (E5 dominant)

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"The database migration caused a cascading failure across all microservices. The root cause was an incompatible schema change that broke the API contract, which led to timeout errors propagating through the retry logic.","importance":0.85,"tags":["incident","causal","database"]}}}
```

**Expected**: E5 (causal) scores highest — strong cause-effect language ("caused", "root cause", "led to", "broke"). E1 moderate.

**Verification**: Record `$MEM3_ID`. Get fingerprint. Check E5 asymmetric variants:
- `cause` variant present with non-zero `l2Norm`
- `effect` variant present with non-zero `l2Norm`

### Memory 4: Entity-Rich (E11 dominant)

```json
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"Diesel is a safe, extensible ORM and query builder for Rust. It supports PostgreSQL, MySQL, and SQLite backends. The diesel_migrations crate handles schema migrations using embedded SQL files.","importance":0.7,"tags":["diesel","rust","orm","database"]}}}
```

**Expected**: E11 (entity) scores highest for entity queries. E6 (keyword) strong. E7 moderate (mentions code concepts).

**Verification**: Record `$MEM4_ID`.

### Memory 5: Intent/Goal (E10 dominant)

```json
{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"store_memory","arguments":{"content":"I want to build a real-time notification system that alerts users when their documents are modified by collaborators. The goal is to reduce response time to under 500ms from edit to notification delivery.","importance":0.75,"tags":["goal","realtime","notification"]}}}
```

**Expected**: E10 (intent) scores highest — clear goal/purpose language. E1 moderate.

**Verification**: Record `$MEM5_ID`.

### Phase 0 Verification Checklist

After storing all 5 memories:

1. **Count check**: Call `get_memetic_status` and verify `fingerprintCount == 5`
2. **Fingerprint check**: Call `get_memory_fingerprint` for each of the 5 IDs
3. **Physical proof**: For each memory, verify:
   - `embeddersPresent >= 10` (most embedders should produce vectors)
   - `createdAt` is a valid RFC3339 timestamp
   - Asymmetric embedders (E5, E8, E10) show both variants

```json
{"jsonrpc":"2.0","id":20,"method":"tools/call","params":{"name":"get_memetic_status","arguments":{}}}
```

---

## Phase 1: Test Gap 3 — All 16 Weight Profiles Exposed

### Test 1.1: search_graph accepts all 16 profiles (Happy Path)

For each of the 16 profiles, issue a search and verify no error:

```bash
PROFILES=(
  "semantic_search" "causal_reasoning" "code_search" "fact_checking"
  "intent_search" "intent_enhanced" "graph_reasoning"
  "temporal_navigation" "sequence_navigation" "conversation_history"
  "category_weighted" "typo_tolerant"
  "pipeline_stage1_recall" "pipeline_stage2_scoring" "pipeline_full"
  "balanced"
)

for profile in "${PROFILES[@]}"; do
  send_mcp '{"jsonrpc":"2.0","id":100,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database","weightProfile":"'"$profile"'"}}}'
  # VERIFY: Response has "content" array, no "error" field
done
```

**Source of Truth**: Each response must contain `fingerprintId` results, not an error.

### Test 1.2: get_unified_neighbors accepts all 16 profiles

```json
{"jsonrpc":"2.0","id":101,"method":"tools/call","params":{"name":"get_unified_neighbors","arguments":{"memory_id":"$MEM1_ID","weight_profile":"balanced"}}}
{"jsonrpc":"2.0","id":102,"method":"tools/call","params":{"name":"get_unified_neighbors","arguments":{"memory_id":"$MEM1_ID","weight_profile":"code_search"}}}
{"jsonrpc":"2.0","id":103,"method":"tools/call","params":{"name":"get_unified_neighbors","arguments":{"memory_id":"$MEM1_ID","weight_profile":"pipeline_full"}}}
```

**Verify**: No error, results array present.

### Test 1.3: search_by_intent accepts all 16 profiles

```json
{"jsonrpc":"2.0","id":104,"method":"tools/call","params":{"name":"search_by_intent","arguments":{"query":"build notification system","weightProfile":"intent_enhanced"}}}
{"jsonrpc":"2.0","id":105,"method":"tools/call","params":{"name":"search_by_intent","arguments":{"query":"build notification system","weightProfile":"balanced"}}}
```

**Verify**: No error.

### Test 1.4: Invalid profile name (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":106,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test","weightProfile":"nonexistent_profile"}}}
```

**Expected**: Error response. The system MUST NOT silently fall back to a default.

---

## Phase 2: Test Gap 1 — Custom Weight Arrays (customWeights)

### Test 2.1: E1-only custom weights (Happy Path)

```json
{"jsonrpc":"2.0","id":200,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database migration","customWeights":{"E1":1.0,"E2":0,"E3":0,"E4":0,"E5":0,"E6":0,"E7":0,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0}}}}
```

**Expected**: Results ranked purely by E1 semantic similarity. Memory 4 (Diesel/database) should rank high.

### Test 2.2: E7-heavy weights for code

```json
{"jsonrpc":"2.0","id":201,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"async function embedding","customWeights":{"E1":0.2,"E2":0,"E3":0,"E4":0,"E5":0,"E6":0,"E7":0.8,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0}}}}
```

**Expected**: Memory 2 (code) should rank #1 or #2.

### Test 2.3: E5-heavy for causal

```json
{"jsonrpc":"2.0","id":202,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"what caused the failure","customWeights":{"E1":0.1,"E2":0,"E3":0,"E4":0,"E5":0.9,"E6":0,"E7":0,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0}}}}
```

**Expected**: Memory 3 (causal) should rank #1.

### Test 2.4: Weights that don't sum to 1.0 (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":203,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test","customWeights":{"E1":0.5,"E7":0.8}}}}
```

**Expected**: Error — weights sum to 1.3, exceeds tolerance 0.01.

### Test 2.5: Weight out of range (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":204,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test","customWeights":{"E1":1.5,"E7":-0.5}}}}
```

**Expected**: Error — weight values outside [0.0, 1.0].

### Test 2.6: customWeights overrides weightProfile

```json
{"jsonrpc":"2.0","id":205,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database","weightProfile":"code_search","customWeights":{"E1":1.0,"E2":0,"E3":0,"E4":0,"E5":0,"E6":0,"E7":0,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0}}}}
```

**Expected**: customWeights takes precedence — results should be E1-ranked, NOT code_search-ranked.

### Test 2.7: customWeights in get_unified_neighbors

```json
{"jsonrpc":"2.0","id":206,"method":"tools/call","params":{"name":"get_unified_neighbors","arguments":{"memory_id":"$MEM3_ID","custom_weights":{"E1":0.1,"E5":0.9}}}}
```

**Expected**: Neighbors ranked primarily by E5 causal similarity to Memory 3.

---

## Phase 3: Test Gap 8 — excludeEmbedders

### Test 3.1: Exclude temporal embedders (Happy Path)

```json
{"jsonrpc":"2.0","id":300,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database","excludeEmbedders":["E2","E3","E4"]}}}
```

**Expected**: Results returned normally. Temporal embedders excluded from fusion.

### Test 3.2: Exclude all except E1 and E7

```json
{"jsonrpc":"2.0","id":301,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"async function","excludeEmbedders":["E2","E3","E4","E5","E6","E8","E9","E10","E11","E12","E13"]}}}
```

**Expected**: Results ranked by E1+E7 only. Memory 2 (code) should rank high.

### Test 3.3: Exclude ALL embedders (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":302,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test","excludeEmbedders":["E1","E2","E3","E4","E5","E6","E7","E8","E9","E10","E11","E12","E13"]}}}
```

**Expected**: Error — cannot exclude all embedders.

### Test 3.4: Invalid embedder name in exclusion (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":303,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"test","excludeEmbedders":["E14"]}}}
```

**Expected**: Error — E14 is not a valid embedder.

### Test 3.5: excludeEmbedders in get_unified_neighbors

```json
{"jsonrpc":"2.0","id":304,"method":"tools/call","params":{"name":"get_unified_neighbors","arguments":{"memory_id":"$MEM1_ID","exclude_embedders":["E2","E3","E4","E6","E13"]}}}
```

**Expected**: Results returned. Excluded embedders should have zero contribution.

### Test 3.6: excludeEmbedders + customWeights (Edge Case)

```json
{"jsonrpc":"2.0","id":305,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database","customWeights":{"E1":0.5,"E5":0.3,"E7":0.2,"E2":0,"E3":0,"E4":0,"E6":0,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0},"excludeEmbedders":["E7"]}}}
```

**Expected**: E7 excluded → weight zeroed. E1 and E5 renormalized: E1=0.5/0.8=0.625, E5=0.3/0.8=0.375.

---

## Phase 4: Test Gap 6 — includeEmbedderBreakdown + dominantEmbedder + agreementLevel

### Test 4.1: Breakdown with semantic query (Happy Path)

```json
{"jsonrpc":"2.0","id":400,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"gravity and spacetime","includeEmbedderBreakdown":true,"includeContent":true}}}
```

**Expected response per result**:
- `embedderBreakdown`: array with objects `{embedder, score, rank, weight, rrfContribution}`
- `dominantEmbedder`: string (e.g., "E1_Semantic") — the embedder with highest RRF contribution
- `agreementLevel`: one of `"low"`, `"medium"`, `"high"`
- Memory 1 (physics) should be top result

**Verification**:
1. `embedderBreakdown` is an array with > 0 entries
2. Each entry has all 5 fields: `embedder`, `score`, `rank`, `weight`, `rrfContribution`
3. `dominantEmbedder` matches the embedder with the max `rrfContribution` in the breakdown
4. `agreementLevel` is `"low"` (0-2 active), `"medium"` (3-6), or `"high"` (7+)

### Test 4.2: Breakdown with code query

```json
{"jsonrpc":"2.0","id":401,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"pub async fn embed_all","includeEmbedderBreakdown":true,"weightProfile":"code_search"}}}
```

**Expected**: Memory 2 (code) ranks #1. E7 should appear in breakdown with high score. `dominantEmbedder` likely `E7_Code`.

### Test 4.3: Breakdown with causal query

```json
{"jsonrpc":"2.0","id":402,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"what caused the cascading failure","includeEmbedderBreakdown":true,"weightProfile":"causal_reasoning"}}}
```

**Expected**: Memory 3 (causal) ranks #1. E5 should have high contribution. `dominantEmbedder` likely `E5_Causal` or `E1_Semantic`.

### Test 4.4: Breakdown disabled (default false)

```json
{"jsonrpc":"2.0","id":403,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database"}}}
```

**Expected**: No `embedderBreakdown`, no `dominantEmbedder`, no `agreementLevel` in results.

### Test 4.5: Breakdown with custom weights

```json
{"jsonrpc":"2.0","id":404,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database ORM Rust","includeEmbedderBreakdown":true,"customWeights":{"E1":0.3,"E2":0,"E3":0,"E4":0,"E5":0,"E6":0.2,"E7":0.2,"E8":0,"E9":0,"E10":0,"E11":0.3,"E12":0,"E13":0}}}}
```

**Expected**: Breakdown shows weights matching custom values: E1=0.3, E6=0.2, E7=0.2, E11=0.3. Memory 4 (Diesel) should rank high.

---

## Phase 5: Test Gap 5 — get_memory_fingerprint

### Test 5.1: Full fingerprint for semantic memory (Happy Path)

```json
{"jsonrpc":"2.0","id":500,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM1_ID","includeVectorNorms":true,"includeContent":true}}}
```

**Expected response**:
- `memoryId`: matches `$MEM1_ID`
- `embedders`: array of 13 entries (all embedders)
- `embeddersPresent`: >= 10
- `content`: "The theory of relativity..."
- `createdAt`: valid RFC3339

**Verification per embedder**:
- `embedder`: "E1" through "E13"
- `name`: matches EmbedderId names (e.g., "V_meaning (Semantic)")
- `dimension`: matches spec (e.g., "1024" for E1)
- `present`: boolean
- `actualDimension`: integer (actual stored vector length)
- `l2Norm`: float if present and includeVectorNorms=true

**Asymmetric embedders** (E5, E8, E10) must show:
- `variants`: array of 2 entries
- E5: `[{variant: "cause", ...}, {variant: "effect", ...}]`
- E8: `[{variant: "source", ...}, {variant: "target", ...}]`
- E10: `[{variant: "intent", ...}, {variant: "context", ...}]`

### Test 5.2: Code memory fingerprint

```json
{"jsonrpc":"2.0","id":501,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM2_ID","includeVectorNorms":true}}}
```

**Expected**: E7 should have high `l2Norm` (code content activates E7 strongly).

### Test 5.3: Filter to specific embedders

```json
{"jsonrpc":"2.0","id":502,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM3_ID","embedders":["E1","E5","E7"]}}}
```

**Expected**: Only 3 entries in `embedders` array: E1, E5, E7.

### Test 5.4: Without vector norms

```json
{"jsonrpc":"2.0","id":503,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM4_ID","includeVectorNorms":false}}}
```

**Expected**: All embedder entries, but `l2Norm` fields absent (skipped via skip_serializing_if).

### Test 5.5: Invalid UUID (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":504,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"not-a-uuid"}}}
```

**Expected**: Error with message about invalid UUID.

### Test 5.6: Non-existent UUID (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":505,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"00000000-0000-0000-0000-000000000000"}}}
```

**Expected**: Error: "Memory 00000000-... not found"

### Test 5.7: Invalid embedder in filter (Edge Case — MUST error)

```json
{"jsonrpc":"2.0","id":506,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM1_ID","embedders":["E1","E14"]}}}
```

**Expected**: Error: "Invalid embedder 'E14'"

---

## Phase 6: Cross-Feature Integration Tests

### Test 6.1: Full pipeline — customWeights + excludeEmbedders + includeEmbedderBreakdown

```json
{"jsonrpc":"2.0","id":600,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database ORM","customWeights":{"E1":0.4,"E2":0,"E3":0,"E4":0,"E5":0.1,"E6":0.2,"E7":0,"E8":0,"E9":0,"E10":0,"E11":0.3,"E12":0,"E13":0},"excludeEmbedders":["E7"],"includeEmbedderBreakdown":true,"includeContent":true}}}
```

**Expected**:
- E7 has zero in breakdown (excluded)
- Weights renormalized: E1=0.4/1.0=0.4, E5=0.1/1.0=0.1, E6=0.2/1.0=0.2, E11=0.3/1.0=0.3
  (Note: E7 was already 0 in custom weights, so exclusion is a no-op here)
- Memory 4 (Diesel/database/ORM) should rank #1
- Breakdown shows contributions from E1, E5, E6, E11

### Test 6.2: Code search with breakdown and fingerprint verification

1. Search for code:
```json
{"jsonrpc":"2.0","id":601,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"embed_all async CoreResult","weightProfile":"code_search","includeEmbedderBreakdown":true}}}
```

2. Take the top result's `fingerprintId` and introspect:
```json
{"jsonrpc":"2.0","id":602,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$TOP_RESULT_ID","embedders":["E1","E5","E7"],"includeVectorNorms":true}}}
```

**Verify**: The top result should be Memory 2. E7 should show highest l2Norm of the three.

### Test 6.3: Compare weight profiles produce different rankings

Run the SAME query with different profiles and verify results differ:

```json
{"jsonrpc":"2.0","id":610,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database schema migration","weightProfile":"semantic_search","includeEmbedderBreakdown":true}}}
{"jsonrpc":"2.0","id":611,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database schema migration","weightProfile":"causal_reasoning","includeEmbedderBreakdown":true}}}
{"jsonrpc":"2.0","id":612,"method":"tools/call","params":{"name":"search_graph","arguments":{"query":"database schema migration","weightProfile":"code_search","includeEmbedderBreakdown":true}}}
```

**Verify**: `dominantEmbedder` should differ between profiles:
- `semantic_search`: E1 dominant
- `causal_reasoning`: E5 dominant or high contribution
- `code_search`: E7 dominant or high contribution

---

## Phase 7: Physical State Verification

After all tests, verify the database state physically.

### 7.1: Memory count

```json
{"jsonrpc":"2.0","id":700,"method":"tools/call","params":{"name":"get_memetic_status","arguments":{}}}
```

**Verify**: `fingerprintCount == 5` (no phantom memories created by searches).

### 7.2: All memories retrievable

For each of the 5 memory IDs:
```json
{"jsonrpc":"2.0","id":701,"method":"tools/call","params":{"name":"get_memory_fingerprint","arguments":{"memory_id":"$MEM_ID","includeContent":true}}}
```

**Verify**: Content matches what was stored. No corruption.

### 7.3: Audit trail exists

```json
{"jsonrpc":"2.0","id":702,"method":"tools/call","params":{"name":"get_audit_trail","arguments":{"limit":20}}}
```

**Verify**: At least 5 `store_memory` entries in audit trail, one per seeded memory.

---

## Phase 8: Error Logging Verification

All error cases (Tests 1.4, 2.4, 2.5, 3.3, 3.4, 5.5, 5.6, 5.7) should produce:

1. **JSON-RPC error response** with `isError: true` and descriptive message
2. **Server-side log** at ERROR level with:
   - Tool name
   - What failed
   - Input that caused the failure
   - Actionable fix description

Check server logs (`RUST_LOG=debug`) for each error case. The log entry should contain enough information to diagnose the issue without reproducing it.

---

## Test Matrix Summary

| Phase | Gap | Test | Type | Expected |
|-------|-----|------|------|----------|
| 0 | - | Seed 5 memories | Setup | 5 memories stored |
| 0 | - | Verify fingerprints | Verification | All 13 embedders present |
| 1.1 | 3 | 16 weight profiles in search_graph | Happy | All return results |
| 1.2 | 3 | 16 profiles in get_unified_neighbors | Happy | All return results |
| 1.3 | 3 | 16 profiles in search_by_intent | Happy | All return results |
| 1.4 | 3 | Invalid profile name | Edge | Error response |
| 2.1 | 1 | E1-only custom weights | Happy | E1-ranked results |
| 2.2 | 1 | E7-heavy custom weights | Happy | Code memory ranks high |
| 2.3 | 1 | E5-heavy custom weights | Happy | Causal memory ranks high |
| 2.4 | 1 | Weights sum > 1.0 | Edge | Error response |
| 2.5 | 1 | Weight out of range | Edge | Error response |
| 2.6 | 1 | customWeights overrides profile | Happy | Custom takes precedence |
| 2.7 | 1 | customWeights in unified_neighbors | Happy | Weighted fusion |
| 3.1 | 8 | Exclude temporal | Happy | Results without E2-E4 |
| 3.2 | 8 | Exclude all except E1+E7 | Happy | Two-embedder fusion |
| 3.3 | 8 | Exclude ALL | Edge | Error response |
| 3.4 | 8 | Invalid embedder name | Edge | Error response |
| 3.5 | 8 | excludeEmbedders in unified | Happy | Filtered fusion |
| 3.6 | 8 | exclude + customWeights | Edge | Renormalized weights |
| 4.1 | 6 | Semantic breakdown | Happy | All breakdown fields |
| 4.2 | 6 | Code breakdown | Happy | E7 dominant |
| 4.3 | 6 | Causal breakdown | Happy | E5 high contribution |
| 4.4 | 6 | Breakdown disabled | Happy | No breakdown fields |
| 4.5 | 6 | Breakdown + custom weights | Happy | Custom weights in breakdown |
| 5.1 | 5 | Full fingerprint | Happy | 13 embedders, norms, variants |
| 5.2 | 5 | Code fingerprint | Happy | E7 high norm |
| 5.3 | 5 | Filter embedders | Happy | Only requested embedders |
| 5.4 | 5 | Without norms | Happy | No l2Norm fields |
| 5.5 | 5 | Invalid UUID | Edge | Error response |
| 5.6 | 5 | Non-existent UUID | Edge | Error response |
| 5.7 | 5 | Invalid embedder filter | Edge | Error response |
| 6.1 | All | Full pipeline integration | Integration | All features compose |
| 6.2 | 5+6 | Search + fingerprint verify | Integration | Physical data match |
| 6.3 | 3+6 | Compare profiles differ | Integration | Rankings differ |
| 7.1 | - | Memory count | Verification | Count == 5 |
| 7.2 | - | All memories intact | Verification | Content matches |
| 7.3 | - | Audit trail | Verification | 5 store entries |

**Total**: 33 test cases (22 happy path, 8 edge case, 3 integration)

---

## Pass/Fail Criteria

- **PASS**: All 33 tests return expected results. No silent failures. No fallbacks.
- **FAIL**: Any test returns unexpected result, or any edge case does NOT error.
- **CRITICAL FAIL**: Any test causes server crash, data corruption, or silent data loss.

Every error must be logged at ERROR level with enough context to debug without reproduction.
