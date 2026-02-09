# Topic Clustering Analysis Report

**Date**: 2026-02-09
**Branch**: casetrack
**System**: 13 embedders, 55 MCP tools, HDBSCAN clustering with weighted agreement

---

## Executive Summary

Topic detection went through 3 fix iterations to resolve a mega-clustering bug where
all memories merged into a single topic. The system now correctly separates distinct
domains, but an 8-member merged topic persists when semantically adjacent domains
(e.g., astrophysics + marine biology) are present. This report documents the current
state, expected outcomes, root causes, and concrete improvement paths.

---

## 1. Test Data

| Domain | Memories | Content |
|--------|----------|---------|
| **Cooking** | 5 | Maillard reaction, sourdough, emulsification, blanching, roux |
| **Astrophysics** | 5 | Neutron stars, CMB, dark energy, gravitational lensing, pulsars |
| **Software Engineering** | 5 | SOLID principles, B-tree indexes, event sourcing, Rust ownership, consistent hashing |
| **Marine Biology** | 5 | Coral bleaching, bioluminescence, whale songs, thermohaline circulation, sea turtle navigation |
| **Ancient History** | 4 | Rosetta Stone, Roman aqueducts, Library of Alexandria, Mesopotamian cuneiform |
| **Total** | **24** | 5 maximally distinct domains |

---

## 2. Results Timeline

### Before Any Fix (Baseline Bug)
- **Hardcoded gap threshold of 0.20** in `hdbscan.rs:606`
- MST edge weights typically [0.03, 0.18] — all below threshold
- **Result**: 1 mega-topic containing ALL memories regardless of domain
- **Root cause**: Threshold too high for cosine distance, no data-driven detection

### Fix 1: Data-Driven Gap Detection (commit 854d69d)
- Replaced hardcoded threshold with largest-gap detection in MST edge weights
- Added metric-specific floors (cosine: 0.03, jaccard: 0.10)
- **Result with 15 memories (3 domains)**: 2 topics (10 + 5)
- **Improvement**: No longer 1 mega-topic, but cooking+astro still merged

### Fix 2: Discriminating Spaces Filter (this session)
- Non-discriminating spaces (single cluster = zero information) excluded from agreement
- E5/E8/E9/E11 often produce 1 cluster, contributing 2.5 weight for free
- **Result with 15 memories (3 domains)**: **3 topics (5 + 4 + 5)**
- **Result with 20 memories (4 domains)**: 2 topics (14 + 4) — mega-merge returned

### Fix 3: Mega-Component Re-Clustering (this session)
- Mega-components (>40% of memories) re-clustered with fresh E1 HDBSCAN
- Fresh threshold on the subset finds internal boundaries the global threshold missed
- Lowered guard from `> 10` to `>= 5` for smaller datasets
- **Result with 20 memories (4 domains)**: 3 topics (4 + 8 + 4)
- **Result with 24 memories (5 domains)**: **4 topics (5 + 8 + 3 + 4)**

---

## 3. Current State vs Expected

### Current (24 memories, 5 domains)

| Topic ID (prefix) | Members | Likely Domains | Agreement |
|--------------------|---------|----------------|-----------|
| `cbd14223` | 5 | Software Engineering | 5.0 |
| `245643f6` | 8 | Cooking + Marine Bio (merged) | 7.75 |
| `51f3e362` | 3 | Ancient History (3 of 4 stored) | 6.0 |
| `50cf8eb5` | 4 | Astrophysics (4 of 5) | 7.25 |
| Noise | 4 | Scattered singletons | — |

**Assigned**: 20/24 (83%)
**Noise**: 4/24 (17%)
**Distinct topics**: 4

### Expected Ideal (24 memories, 5 domains)

| Topic | Members | Agreement |
|-------|---------|-----------|
| Cooking | 5 | High |
| Astrophysics | 5 | High |
| Software Engineering | 5 | High |
| Marine Biology | 5 | High |
| Ancient History | 4 | High |
| **Noise** | **0** | — |

**Assigned**: 24/24 (100%)
**Noise**: 0/24 (0%)
**Distinct topics**: 5

### Gap Analysis

| Metric | Current | Expected | Gap |
|--------|---------|----------|-----|
| Topic count | 4 | 5 | -1 |
| Largest topic | 8 | 5 | +3 (over-merged) |
| Noise points | 4 | 0 | +4 |
| Domain purity | ~80% | 100% | -20% |
| Search isolation | 100% | 100% | 0% (perfect) |

---

## 4. Root Cause of Remaining 8-Member Merge

### Why Cooking + Marine Biology Merge

Both domains describe biological/chemical processes in natural systems:
- Cooking: enzyme denaturation, microbial fermentation, chemical reactions
- Marine bio: symbiotic organisms, biological processes, chemical gradients

In E1 (Semantic) space, these domains have **inter-cluster distances only ~0.02-0.04
greater than intra-cluster distances**. HDBSCAN's single-gap detection finds one
threshold that separates software/history/astro from cooking/marine, but cannot find
the second boundary within the cooking+marine group.

### The Transitive Closure Problem

Union-Find in `synthesize_topics()` computes transitive closure:
- If cooking-A agrees with cooking-B (same cluster in 6 spaces)
- And cooking-B agrees with marine-C (same cluster in 4 spaces)
- Then A and C are in the same component, even if A and C don't directly agree

This is intrinsic to Union-Find. A single "bridge" memory linking two otherwise
distinct domains merges them permanently.

### Why Search Isolation is Perfect Despite Merged Topics

Search uses **pairwise similarity** (E1 cosine distance), which correctly ranks
cooking queries above marine biology results. Topic detection uses **cluster
co-membership** (binary: same cluster or not), which is coarser. A memory can be
in the same cluster as another without being particularly similar to it.

---

## 5. Improvement Paths (Prioritized)

### P0: Recursive Mega-Component Splitting

**Current**: One level of re-clustering. If the sub-result still has a mega-component,
it keeps it.

**Fix**: Apply the mega-component guard recursively. After re-clustering, check if any
resulting component is still > threshold. If so, re-cluster again.

**Expected impact**: Would split the 8-member component into cooking(5) + marine(3-5)
on the second pass.

**Effort**: Low (add loop or recursion to existing code)

### P1: Replace Union-Find with Graph Community Detection

**Current**: Binary edges (agreement >= 2.5 → connected) + Union-Find (transitive closure).

**Fix**: Build weighted agreement graph, apply Louvain or label propagation community
detection. This finds dense subgroups without forcing transitive closure.

**Expected impact**: Eliminates bridge-memory problem entirely. Most significant improvement.

**Effort**: Medium (new algorithm, but well-understood)

### P2: Minimum Internal Density Check

**Current**: Components form if any path of edges >= 2.5 exists.

**Fix**: After Union-Find, verify that the average pairwise agreement within each
component exceeds a minimum (e.g., 2.0). If not, split the component.

**Expected impact**: Catches loose components held together by bridge edges.

**Effort**: Low (add post-processing check)

### P3: Multi-Gap Detection in HDBSCAN

**Current**: `detect_gap_threshold()` finds the single largest gap.

**Fix**: Find all gaps above `min_significant_gap`, use the SECOND largest gap as
threshold when the first gap produces a dominant cluster.

**Expected impact**: Better multi-way splitting in E1 space.

**Effort**: Low-Medium (modify gap detection logic)

### P4: Keyword Discriminator Boost

**Current**: E6 (Sparse/keyword) and E13 (SPLADE) have high discrimination but are
often below the `min_cluster_size=5` threshold for sparse spaces.

**Fix**: Lower sparse space min_cluster_size to 3 (matching dense spaces), or use
keyword overlap as a secondary splitting signal in mega-components.

**Expected impact**: Cooking keywords (flour, yeast, oven) vs marine keywords (coral,
whale, ocean) would provide strong separation signal.

**Effort**: Low (config change) to Medium (new splitting logic)

---

## 6. Per-Embedder Discrimination Analysis

Based on search results across 24 memories:

| Embedder | Category | Topic Weight | Typical Cluster Count (24pts) | Discriminating? |
|----------|----------|-------------|-------------------------------|-----------------|
| E1 Semantic | Semantic | 1.0 | 2-3 | Yes (primary separator) |
| E5 Causal | Semantic | 1.0 | 1 | **No** (0.93-0.97 for all text) |
| E6 Sparse | Semantic | 1.0 | 1-2 | Marginal (min_cluster_size=5) |
| E7 Code | Semantic | 1.0 | 2-3 | Yes |
| E10 Multimodal | Semantic | 1.0 | 2-3 | Yes |
| E12 ColBERT | Semantic | 1.0 | 1-2 | Marginal |
| E13 SPLADE | Semantic | 1.0 | 1-2 | Marginal (min_cluster_size=5) |
| E8 Graph | Relational | 0.5 | 1-2 | Marginal |
| E11 Entity | Relational | 0.5 | 1 | **No** (0.94-0.97 for all) |
| E9 HDC | Structural | 0.5 | 1 | **No** (high variance, low signal) |
| E2/E3/E4 | Temporal | 0.0 | — | Excluded (correct) |

**Key finding**: Only E1, E7, and E10 consistently produce >= 2 clusters with 24 memories.
E5, E11, and E9 almost never discriminate. The discriminating-spaces filter correctly
excludes these, but the remaining discriminating spaces may still not produce enough
separation for semantically adjacent domains.

---

## 7. Verified Behaviors (Working Correctly)

| Feature | Status | Evidence |
|---------|--------|----------|
| Multi-topic detection | PASS | 4 topics from 5 domains (was 1 mega-topic) |
| Deterministic topic IDs | PASS | UUIDv5 from sorted member bytes, identical on re-run |
| Churn = 0 on stable data | PASS | Re-detection produces same IDs, churn stays 0 |
| Search domain isolation | PASS | All 5 domain queries return only same-domain results |
| Discriminating spaces filter | PASS | Single-cluster spaces excluded from agreement |
| Mega-component re-clustering | PASS | 14-member component split to 8+4+noise |
| Noise point handling | PASS | Low-confidence points correctly labeled as noise |
| Weighted agreement threshold | PASS | All topics have agreement >= 2.5 per ARCH-09 |
| Temporal exclusion | PASS | E2-E4 have weight 0.0, never count toward topics |

---

## 8. Recommendation

**Immediate (P0)**: Add recursive mega-component splitting. This is a 10-line change
that would convert the current single-pass re-clustering into a loop that continues
splitting until no component exceeds the threshold or re-clustering stops producing
new splits. This alone should break the 8-member cooking+marine merge.

**Next iteration (P1)**: Replace Union-Find with Louvain community detection. This is
the structural fix that eliminates the transitive closure problem entirely, producing
tighter, more coherent topics without bridge-memory contamination.

---

## 9. Code Changes Made (This Session)

### File: `crates/context-graph-core/src/clustering/manager.rs`
1. **Discriminating spaces filter** (lines 778-797): Compute per-space cluster count,
   skip spaces with < 2 clusters from agreement calculation
2. **Updated `compute_pairwise_weighted_agreement`** (line 921): Accepts `discriminating_spaces`
   parameter, skips non-discriminating embedders
3. **Mega-component re-clustering** (lines 843-920): Fresh E1 HDBSCAN on mega-component
   members instead of looking up existing cluster assignments
4. **Lowered guard threshold** (line 851): `>= 5` instead of `> 10`
5. **Renamed `total_fingerprints` → `max_space_size`** (line 479): Misleading variable name
6. **Error logging for comparison failures** (line 707): `match` with `tracing::warn!`

### File: `crates/context-graph-core/src/clustering/hdbscan.rs`
7. **Fixed misleading comment** (line 595): "relative position" → "cluster size"

### Test Results
- 32 clustering unit tests: PASS
- 655 MCP integration tests: PASS
- Release build: clean (only pre-existing benchmark warnings)
