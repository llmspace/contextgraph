# Causal Hint Integration - Manual Testing Plan

## Overview

This document provides a comprehensive manual testing plan for the Causal Discovery LLM + E5 Embedder Integration. Testing follows Full State Verification (FSV) principles.

## Architecture Under Test

```
store_memory(content)
    ↓
CausalHintProvider.get_hint(content)  ← 100ms timeout
    ↓
EmbeddingMetadata { causal_hint: Some(...) }
    ↓
embed_all_with_metadata()
    ↓
E5.embed_dual_with_hint() applies direction bias
    ↓
RocksDB CF_FINGERPRINTS (Source of Truth)
```

## Source of Truth

| Component | Storage Location | Verification Method |
|-----------|------------------|---------------------|
| Fingerprints | RocksDB `CF_FINGERPRINTS` | `retrieve_async(uuid)` |
| E5 Embeddings | `fingerprint.semantic.e5_as_cause/e5_as_effect` | Direct vector inspection |
| Content | RocksDB `CF_CONTENT` | `get_content(uuid)` |
| MCP Response | JSON-RPC response | Parse `result.content[0].text` |

---

## Test Suite 1: CausalHint Type Verification

### Test 1.1: CausalDirectionHint Enum Values

**Objective**: Verify enum serialization/deserialization

**Synthetic Data**:
```rust
CausalDirectionHint::Cause   → "cause"
CausalDirectionHint::Effect  → "effect"
CausalDirectionHint::Neutral → "neutral"
```

**Verification Command**:
```bash
cargo test -p context-graph-core -- causal_direction --nocapture
```

### Test 1.2: CausalHint Bias Factors

**Objective**: Verify `bias_factors()` method returns correct values

| Direction | Expected cause_bias | Expected effect_bias |
|-----------|---------------------|----------------------|
| Cause | 1.3 | 0.8 |
| Effect | 0.8 | 1.3 |
| Neutral | 1.0 | 1.0 |

**Verification**: Unit test in `context-graph-core/src/traits/multi_array_embedding.rs`

---

## Test Suite 2: LLM analyze_single_text()

### Test 2.1: Causal Content Analysis

**Synthetic Input**:
```
"High cortisol levels cause memory impairment in the hippocampus."
```

**Expected Output**:
```json
{
  "is_causal": true,
  "direction": "cause",
  "confidence": >= 0.7,
  "key_phrases": ["cause", "memory impairment"]
}
```

**Verification**:
```bash
cargo test -p context-graph-causal-agent -- analyze_single_text --nocapture
```

### Test 2.2: Non-Causal Content Analysis

**Synthetic Input**:
```
"The weather in Paris is pleasant during spring."
```

**Expected Output**:
```json
{
  "is_causal": false,
  "direction": "neutral",
  "confidence": < 0.5
}
```

### Test 2.3: Effect-Direction Content

**Synthetic Input**:
```
"Memory impairment results from prolonged stress exposure."
```

**Expected Output**:
```json
{
  "is_causal": true,
  "direction": "effect",
  "confidence": >= 0.6
}
```

---

## Test Suite 3: E5 embed_dual_with_hint()

### Test 3.1: Cause Direction Bias Application

**Objective**: Verify cause bias (1.3x cause, 0.8x effect) is applied

**Setup**:
1. Generate baseline embedding without hint
2. Generate embedding with `CausalDirectionHint::Cause`
3. Compare vector magnitudes

**Verification**:
```rust
// Expected: cause_vec magnitude ~1.3x baseline
// Expected: effect_vec magnitude ~0.8x baseline
assert!((cause_magnitude / baseline_magnitude - 1.3).abs() < 0.1);
assert!((effect_magnitude / baseline_magnitude - 0.8).abs() < 0.1);
```

### Test 3.2: Effect Direction Bias Application

**Objective**: Verify effect bias (0.8x cause, 1.3x effect) is applied

**Verification**:
```rust
assert!((cause_magnitude / baseline_magnitude - 0.8).abs() < 0.1);
assert!((effect_magnitude / baseline_magnitude - 1.3).abs() < 0.1);
```

### Test 3.3: Neutral Direction (No Bias)

**Objective**: Verify neutral hint applies no bias

**Verification**:
```rust
assert!((cause_magnitude / baseline_magnitude - 1.0).abs() < 0.01);
assert!((effect_magnitude / baseline_magnitude - 1.0).abs() < 0.01);
```

---

## Test Suite 4: MCP store_memory Integration

### Test 4.1: Happy Path - Causal Content Storage

**Synthetic Input**:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "store_memory",
    "arguments": {
      "content": "SYNTHETIC_CAUSAL_001: Oxidative stress causes neuronal damage in Alzheimer's disease.",
      "rationale": "Testing causal hint integration"
    }
  },
  "id": "test-causal-001"
}
```

**Expected MCP Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [{
      "type": "text",
      "text": "{ \"node_id\": \"<UUID>\", ... }"
    }]
  },
  "id": "test-causal-001"
}
```

**Source of Truth Verification**:
1. Extract `node_id` from response
2. Query RocksDB: `retrieve_async(node_id)`
3. Verify:
   - `fingerprint.semantic.e5_as_cause` is non-zero
   - `fingerprint.semantic.e5_as_effect` is non-zero
   - Content stored in `CF_CONTENT`

### Test 4.2: Happy Path - Non-Causal Content

**Synthetic Input**:
```json
{
  "content": "SYNTHETIC_NEUTRAL_001: The Eiffel Tower is located in Paris, France."
}
```

**Verification**:
- E5 vectors use default bias (1.0, 1.0)
- No causal hint enhancement applied

### Test 4.3: Happy Path - Effect-Direction Content

**Synthetic Input**:
```json
{
  "content": "SYNTHETIC_EFFECT_001: Inflammation results from immune system activation."
}
```

**Verification**:
- E5 cause vector: ~0.8x magnitude
- E5 effect vector: ~1.3x magnitude

---

## Test Suite 5: Edge Cases (Boundary & Error Conditions)

### Edge Case 5.1: Empty Content

**Before State**:
```
Query: get_memetic_status()
→ fingerprint_count: N
```

**Synthetic Input**:
```json
{
  "content": ""
}
```

**Expected Behavior**: Error response "Content cannot be empty"

**After State**:
```
Query: get_memetic_status()
→ fingerprint_count: N (unchanged)
```

### Edge Case 5.2: Maximum Content Length

**Before State**: Record fingerprint count

**Synthetic Input**: 100KB of text with causal markers
```
"SYNTHETIC_MAXLEN_001: " + "Stress causes damage. " * 5000
```

**Expected Behavior**:
- Content truncated to model max_tokens
- Causal hint generated from truncated content
- Fingerprint stored successfully

**After State Verification**:
- `fingerprint_count: N+1`
- Content retrievable (may be truncated)

### Edge Case 5.3: CausalHintProvider Timeout (Simulated)

**Objective**: Verify graceful degradation when LLM times out

**Setup**: Set timeout to 1ms (guaranteed timeout)

**Expected Behavior**:
- Warning logged: "CausalHintProvider: Analysis timed out"
- `causal_hint: None` passed to embedder
- E5 uses marker-based detection fallback
- Fingerprint stored successfully

**Verification**:
- Check logs for timeout warning
- Verify fingerprint exists in RocksDB
- E5 vectors present (from fallback)

---

## Test Suite 6: Search Integration Verification

### Test 6.1: search_causes Tool

**Setup**: Store synthetic causal content from Test 4.1

**Query**:
```json
{
  "name": "search_causes",
  "arguments": {
    "query": "What causes neuronal damage?",
    "topK": 5
  }
}
```

**Expected Result**:
- Returns fingerprint from Test 4.1
- Uses E5 asymmetric similarity (effect→cause 0.8x)

**Verification**:
- Check returned `node_id` matches stored fingerprint
- Verify asymmetric scoring applied

### Test 6.2: search_graph with Causal Content

**Query**:
```json
{
  "name": "search_graph",
  "arguments": {
    "query": "oxidative stress Alzheimer's",
    "strategy": "multi_space",
    "enableAsymmetricE5": true
  }
}
```

**Verification**:
- E5 causal reranking applied
- Results include Test 4.1 fingerprint

---

## Test Execution Script

```bash
#!/bin/bash
# Run from project root: ./docs/run_causal_hint_tests.sh

set -e

echo "=== Test Suite 1: Type Verification ==="
cargo test -p context-graph-core -- causal --nocapture 2>&1 | tee /tmp/test_suite_1.log

echo "=== Test Suite 2: LLM Analysis ==="
cargo test -p context-graph-causal-agent -- analyze_single_text --nocapture 2>&1 | tee /tmp/test_suite_2.log

echo "=== Test Suite 3: E5 embed_dual_with_hint ==="
cargo test -p context-graph-embeddings -- embed_dual_with_hint --nocapture 2>&1 | tee /tmp/test_suite_3.log

echo "=== Test Suite 4: MCP Integration ==="
cargo test -p context-graph-mcp -- store_memory --nocapture 2>&1 | tee /tmp/test_suite_4.log

echo "=== Test Suite 5: Edge Cases ==="
cargo test -p context-graph-mcp -- edge_case --nocapture 2>&1 | tee /tmp/test_suite_5.log

echo "=== All Tests Complete ==="
echo "Logs saved to /tmp/test_suite_*.log"
```

---

## Evidence Collection Template

For each test, collect:

```
TEST: [Test ID]
TIMESTAMP: [ISO 8601]
INPUT: [Synthetic data]
EXPECTED: [Expected outcome]

BEFORE STATE:
  fingerprint_count: [N]
  relevant_uuids: [list]

EXECUTION:
  [Command/API call]

RESPONSE:
  [JSON response]

AFTER STATE:
  fingerprint_count: [N+1 or N]
  new_uuid: [UUID if created]

SOURCE OF TRUTH VERIFICATION:
  retrieve_async(uuid): [fingerprint data]
  content: [stored content]
  e5_as_cause magnitude: [float]
  e5_as_effect magnitude: [float]

RESULT: PASS/FAIL
EVIDENCE: [Screenshot or log excerpt]
```

---

## Acceptance Criteria

| Criterion | Requirement |
|-----------|-------------|
| Type Tests | All enum values serialize correctly |
| LLM Analysis | Causal content detected with >0.6 confidence |
| E5 Bias | Direction bias within 10% of expected |
| MCP Integration | Fingerprint stored with enhanced E5 vectors |
| Edge Cases | Graceful handling, no data corruption |
| Search | Causal content retrievable via search_causes |

---

## Rollback Procedure

If tests fail:
1. Check logs in `/tmp/test_suite_*.log`
2. Identify failing component
3. Restore previous state: `git checkout -- <file>`
4. Re-run affected test suite

---

## Version

- Plan Version: 1.0
- Created: 2026-01-26
- Components: CausalHint, CausalDiscoveryLLM, E5 Embedder, MCP store_memory
