# E5 Causal Intelligence: Integration Gaps Analysis

**Date**: 2026-02-11
**Scope**: All 55 MCP tools, search pipeline, storage layer, consolidation
**Method**: Code-level trace of every E5 consumer and potential consumer

---

## Executive Summary

The system is **NOT optimally utilizing E5's intelligence**. While the E5 embedder itself runs correctly, its insights are underexploited in three critical ways:

1. **HNSW candidate retrieval ignores direction** -- the directional indexes exist but are never queried during search
2. **Causal gate only applied in 1 of 55 tools** -- `search_graph` is the only consumer
3. **LLM-powered tools don't use E5 for pre-filtering** -- expensive LLM calls wasted on non-causal content

---

## Gap 1: HNSW Direction-Blind Candidate Retrieval (HIGH)

### The Problem

Three E5 HNSW indexes are maintained:
- `E5Causal` (legacy) -- indexed with cause vector
- `E5CausalCause` -- indexed with cause vectors
- `E5CausalEffect` -- indexed with effect vectors

But the search code **always queries `E5Causal` with `e5_active_vector()` (= cause vector)**:

```
search.rs:530  if let Some(e5_index) = index_registry.get(EmbedderIndex::E5Causal) {
search.rs:531      if let Ok(e5_candidates) = e5_index.search(query.e5_active_vector(), k, None) {
```

And the query vector router confirms:
```
search.rs:130  4 => Some(query.e5_active_vector()),  // E5 Causal -- ALWAYS cause vector
```

The `E5CausalCause` and `E5CausalEffect` indexes are **populated at insert time** (`index_ops.rs:76-78`) but **never queried at search time**.

### Why This Matters

When a user asks "what are the effects of deforestation?" (seeking effects):
- **Should**: Query `E5CausalCause` index with query's effect vector, finding memories whose cause vectors are most similar
- **Actually does**: Queries `E5Causal` with query's cause vector, finding memories whose cause vectors are most similar to the query's cause vector -- **wrong pairing**

The asymmetric scoring IS applied post-retrieval (`search.rs:208-214`), but by then the wrong candidates may have already been selected by the HNSW recall phase. The correct candidates might have been missed because they weren't retrieved.

### Impact

Medium-High. With E5 weight at 0.10, E5's HNSW recall has limited influence on final results (E1 at 0.40 dominates candidate selection). But for queries where E5 provides the discriminating signal, wrong candidates get retrieved.

### Fix

In `search.rs` around line 529, use the direction-aware indexes:
```rust
// Current (direction-blind):
if let Some(e5_index) = index_registry.get(EmbedderIndex::E5Causal) {
    e5_index.search(query.e5_active_vector(), k, None)

// Should be (direction-aware):
let e5_idx = match options.causal_direction {
    CausalDirection::Cause => EmbedderIndex::E5CausalCause,
    CausalDirection::Effect => EmbedderIndex::E5CausalEffect,
    _ => EmbedderIndex::E5Causal,
};
let e5_query_vec = match options.causal_direction {
    CausalDirection::Cause => query.get_e5_as_cause(),
    CausalDirection::Effect => query.get_e5_as_effect(),
    _ => query.e5_active_vector(),
};
if let Some(e5_index) = index_registry.get(e5_idx) {
    e5_index.search(e5_query_vec, k, None)
```

---

## Gap 2: Causal Gate Applied in Only 1 Tool (HIGH)

### The Problem

`apply_causal_gate()` is called from exactly ONE location in the entire codebase:

```
memory_tools.rs:1699  result.similarity = apply_causal_gate(result.similarity, e5_sim, is_causal);
```

This means only `search_graph` benefits from the causal gate. The following tools that perform search do NOT apply it:

| Tool | Searches? | Applies Gate? | Should It? |
|------|-----------|---------------|------------|
| `search_graph` | Yes | **Yes** | Yes |
| `search_causes` | Yes | No | Yes -- would demote non-causal noise in results |
| `search_effects` | Yes | No | Yes -- same reason |
| `search_robust` | Yes | No | Yes -- E9 blind-spots should be filtered |
| `search_by_keywords` | Yes | No | Yes for causal keyword queries |
| `search_by_entities` | Yes | No | Conditional -- only for causal entity queries |
| `search_connections` | Yes | No | Conditional -- for causal connection queries |
| `search_code` | Yes | No | No -- code search is not causal |
| `search_recent` | Yes | No | No -- temporal, not causal |

### Why search_causes/search_effects Not Applying Gate Is Problematic

These tools use `semantic_search` profile (E1=0.33) and apply direction modifiers (0.8x/1.2x), but they don't apply the binary causal gate. This means non-causal content that happens to have high E1 semantic similarity can pollute causal results.

Example: User asks "what causes inflation?"
- Memory A: "Federal Reserve raises interest rates" (E1=0.85, E5=0.15 -- not causal text)
- Memory B: "Money supply expansion causes inflation" (E1=0.80, E5=0.42 -- causal text)
- Without gate: A ranks above B (0.85 > 0.80)
- With gate: A demoted to 0.765, B boosted to 0.840 -- **B correctly ranks first**

### Fix

Apply `apply_causal_gate()` post-fusion in `search_causes` and `search_effects` (causal_tools.rs), and optionally in `search_robust`.

---

## Gap 3: Symmetric E5 Scoring in compute_semantic_fusion (MEDIUM)

### The Problem

The `compute_semantic_fusion()` function at `search.rs:370-385` always uses symmetric scoring:

```rust
compute_cosine_similarity(query.e5_active_vector(), stored.e5_active_vector()),
```

This symmetric function is the **fallback** used when `causal_direction == Unknown`. But even when the caller HAS detected causal direction, the HNSW candidate set was built with symmetric scores. The direction-aware scoring only kicks in at lines 208-214 for the re-scoring phase.

### Why It Matters

The fusion scores that drive candidate ranking (before re-scoring) use symmetric E5. This means the E5 contribution to RRF candidate selection is always symmetric regardless of query direction. Asymmetric scoring only corrects AFTER candidates are already selected.

### Impact

Low-Medium. The re-scoring phase (lines 208-214) corrects scores for candidates that were retrieved, but can't recover candidates that were excluded due to symmetric scoring in the HNSW phase.

---

## Gap 4: LLM Tools Don't Use E5 Pre-Filtering (HIGH)

### The Problem

`trigger_causal_discovery` (causal_discovery_tools.rs) does use E5 for embedding discovered relationships, but does NOT use E5 to pre-filter candidate memories before sending pairs to the LLM.

Current flow:
1. Retrieve candidate memories using E1 semantic similarity
2. Generate ALL pairs
3. Send ALL pairs to LLM (Hermes-2-Pro-Mistral-7B)
4. LLM analyzes each pair for causal relationships
5. Embed discovered relationships with E5

The problem: Step 2-3 sends ALL candidate pairs to the LLM, including pairs where neither memory is causal content. The LLM will correctly return "no relationship" but this wastes inference time.

### What Should Happen

After step 1, add: **Filter candidates by E5 gate -- only send pairs where at least one memory scores E5 >= 0.30 (CAUSAL_THRESHOLD).**

With 98% TNR on the gate, this would eliminate ~98% of non-causal memories from candidates, potentially reducing LLM calls by 50-70% while maintaining recall of actual causal relationships.

### Same Issue in `discover_graph_relationships`

Uses E1 to select candidates for LLM analysis. Could use E5 gate to filter non-causal pairs before LLM inference.

---

## Gap 5: Consolidation is E5-Blind (MEDIUM)

### The Problem

`trigger_consolidation` and `merge_concepts` (consolidation.rs, curation_tools.rs) merge memories based on E1 cosine similarity (threshold ~0.85). They do NOT check E5 causal direction compatibility.

### Why It Matters

Two memories could have high E1 similarity (same topic) but opposite causal roles:
- "Smoking causes lung cancer" (cause-oriented, E5 cause > effect)
- "Lung cancer is caused by various factors" (effect-oriented, E5 effect > cause)

Merging these would destroy the directional distinction. The merged memory would lose its causal orientation, degrading future search_causes/search_effects results.

### Fix

Before merging, compare E5 causal direction (using `infer_direction_from_fingerprint()`). If directions are opposite (one cause-dominant, other effect-dominant), either reject the merge or warn the user.

---

## Gap 6: Gate Boost/Demotion Factors Are Conservative (LOW-MEDIUM)

### Current Values
- `CAUSAL_BOOST = 1.05` (+5%)
- `NON_CAUSAL_DEMOTION = 0.90` (-10%)

### The Problem

With typical search scores in the 0.70-0.95 range, a 5% boost produces a 0.035-0.048 absolute change. This is often insufficient to overcome the gap between the top E1-ranked result and a lower-ranked but more causally relevant result.

For example:
- Result A: sim=0.88, E5=0.40 -> boosted to 0.924 (+0.044)
- Result B: sim=0.92, E5=0.15 -> demoted to 0.828 (-0.092)
- Gate swaps A and B (good)

But:
- Result A: sim=0.82, E5=0.40 -> boosted to 0.861 (+0.041)
- Result B: sim=0.93, E5=0.15 -> demoted to 0.837 (-0.093)
- Gate swaps A and B (good, but margin is tight)

The asymmetry helps (demotion is 2x stronger than boost), but with a 0.11+ score gap, the gate can't overcome it. This is by design -- a conservative gate avoids false reranking -- but it limits the gate's effectiveness.

### Recommendation

Consider increasing to `CAUSAL_BOOST = 1.10` and `NON_CAUSAL_DEMOTION = 0.85` after verifying Phase 5 TPR/TNR remain above targets. The 98% TNR gives confidence that demotion targets are correct.

---

## Gap 7: No Per-Result Gate Transparency (LOW)

### The Problem

Search responses include `"asymmetricE5Applied": true/false` but NOT per-result gate details. Users cannot see:
- Which results were boosted vs demoted
- What the E5 score was for each result
- How much each result's score changed

### Why It Matters

For debugging and trust. When a user's expected result ranks lower than expected, they have no visibility into whether E5 gate behavior caused the reranking or not.

### Fix

Add to each search result in the response:
```json
{
  "causalGate": {
    "e5Score": 0.42,
    "action": "boost",
    "scoreDelta": 0.04
  }
}
```

---

## Summary: Integration Quality by Tool

### Excellent Integration (Using E5's Full Intelligence)
- `search_graph` -- gate + direction-aware scoring + auto-profile + transparency
- `search_causal_relationships` -- E5 dual vectors for relationship search
- `get_causal_chain` -- E5 asymmetric similarity for hop scoring
- `trigger_causal_discovery` -- E5 dual embeddings for relationship storage

### Partial Integration (Direction-Aware Scoring, No Gate)
- `search_causes` -- direction modifiers but no gate, uses semantic_search profile
- `search_effects` -- direction modifiers but no gate, uses semantic_search profile

### No Integration (Should Have Some)
- `search_robust` -- no E5 awareness, blind-spot search ignores causal content
- `search_by_keywords` -- no E5 for causal keyword queries
- `search_connections` -- no E5 for causal graph connections
- `discover_graph_relationships` -- no E5 pre-filtering before LLM
- `trigger_consolidation` -- no E5 direction check before merge
- `merge_concepts` -- no E5 direction validation

### Correctly Not Using E5
- File watcher tools, temporal tools, provenance tools, maintenance tools

---

## Priority Matrix

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| 1. Direction-aware HNSW retrieval | High | Medium (modify search.rs HNSW routing) | **P1** |
| 2. Gate in search_causes/effects | High | Low (add 5 lines per tool) | **P1** |
| 4. E5 pre-filter for LLM tools | High | Medium (add gate filter before LLM) | **P1** |
| 5. E5 direction check in consolidation | Medium | Low (add direction comparison) | **P2** |
| 3. Symmetric fusion scoring | Medium | Medium (thread direction through) | **P2** |
| 6. Increase gate factors | Low-Medium | Low (change 2 constants) | **P3** |
| 7. Per-result gate transparency | Low | Low (add fields to response) | **P3** |

---

## Conclusion

The E5 embedder produces valuable intelligence -- 97.5% intent detection, 83.4%/98.0% gate TPR/TNR, perfect 1.5x direction ratio. But the rest of the system only exploits this in `search_graph`. The three P1 gaps (direction-aware HNSW, gate in causal tools, LLM pre-filtering) represent the largest missed value. Fixing them requires modest code changes in 3-4 files but would meaningfully improve causal search quality and reduce LLM costs.
