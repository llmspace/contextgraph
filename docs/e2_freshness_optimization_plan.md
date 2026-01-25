# E2 V_freshness Optimization Plan

**Version**: 1.2.0
**Date**: 2026-01-25
**Status**: COMPLETED
**Embedder**: E2 (V_freshness) - Temporal Recency

---

## Executive Summary

E2 (V_freshness) is a temporal embedder that captures recency patterns. **IMPLEMENTATION COMPLETE** - the following features are now active:

1. **Auto-detection of temporal queries**: Queries containing "yesterday", "recently", "last week", etc. automatically enable temporal boost
2. **EnrichmentConfig integration**: `temporal_boost_enabled`, `temporal_weight`, and `decay_function` fields added
3. **POST-retrieval boost in enrichment pipeline**: Temporal boost applied after RRF fusion per ARCH-25
4. **New `search_recent` MCP tool**: Dedicated tool for temporal search with configurable decay functions

**Implementation Details**:
- `temporal_boost_enabled` auto-set when `QueryType::Temporal` detected
- Default temporal_weight: 0.3 when enabled
- Default decay_function: Exponential
- Boost factor clamped to [0.8, 1.2] per ARCH-33 style

**Files Modified**:
- `enrichment_dtos.rs`: Added temporal fields to EnrichmentConfig
- `query_type_detector.rs`: Sets temporal_boost_enabled when temporal patterns detected
- `enrichment_pipeline.rs`: Phase 3.5 applies temporal boost POST-retrieval
- `temporal.rs`: New tool definition for search_recent
- `temporal_dtos.rs`: New DTOs for temporal search
- `temporal_tools.rs`: New handler for search_recent
- `dispatch.rs`: Added dispatch case for search_recent
- `names.rs`: Added SEARCH_RECENT constant
- `mod.rs` (definitions): Registered temporal tool (41 total tools)

---

## Previous Analysis (Preserved for Reference)

E2 infrastructure was FULLY FUNCTIONAL but not being utilized because:

1. `temporal_weight` defaults to `0.0` in `TemporalSearchOptions` (now auto-set to 0.3 for temporal queries)
2. The MCP tools expose `temporalWeight` but no examples/documentation encourage its use (now auto-enabled)
3. No auto-detection of temporal queries (now implemented via TEMPORAL_PATTERNS)

### Verification Results (2026-01-25)

All 8 E2 tests pass:
- E2 Dimension: 512 ✓
- E2 L2 Normalization: 1.0000 ✓
- E2 Timestamp Order: 1h(0.952) > 24h(0.816) > 7d(0.665) ✓
- Linear Decay Order: 1h(0.994) > 24h(0.857) > 7d(0.000) ✓
- Exponential Half-Life: At 12h, score=0.500 ✓
- Step Decay Buckets: <5m=1.0, <1h=0.8 ✓
- has_any_boost Logic: false/true correctly ✓
- E2 Latency: <0.01ms (budget: <2ms) ✓

---

## 1. Architecture Overview

### 1.1 E2 Model Location

```
crates/context-graph-embeddings/src/models/custom/temporal_recent/
├── mod.rs           # Module exports
├── model.rs         # TemporalRecentModel struct
├── compute.rs       # Decay embedding algorithm
├── constants.rs     # Decay rates, dimensions
├── timestamp.rs     # Timestamp parsing
├── traits.rs        # EmbeddingModel impl
└── tests.rs         # Unit tests
```

### 1.2 Key Files for Integration

| File | Purpose |
|------|---------|
| `temporal_boost.rs` | Post-retrieval boost functions |
| `search.rs` | `apply_full_temporal_boosts()` |
| `options.rs` | `TemporalSearchOptions` struct |
| `core.rs` | `search_graph` tool definition |
| `memory_tools.rs` | Handler implementation |

### 1.3 Constitution Constraints

Per constitution v6.5.0:

```yaml
# E2 Definition
E2: { name: V_freshness, dim: 512, finds: "Recency", gpu: "warm-loaded" }

# Architectural Rules
ARCH-25: "Temporal boosts POST-retrieval only, NOT in similarity fusion"
AP-60: "Temporal (E2-E4) MUST NOT count toward topics"
AP-73: "Temporal MUST NOT be used in similarity fusion"

# Signal Definition
E2_E3_E4_temporal:
  signal_meaning: "Temporal context - recency, periodicity, sequence"
  role: "POST-RETRIEVAL badges only"
  tunable_params:
    recency_boost: { under_1h: 1.3, under_1d: 1.2, under_7d: 1.1 }
  topic_weight: 0.0  # ALWAYS - temporal proximity ≠ semantic relationship
```

---

## 2. Current Implementation Analysis

### 2.1 E2 Embedding Generation

```rust
// From compute.rs
pub fn compute_decay_embedding(
    timestamp: DateTime<Utc>,
    reference_time: Option<DateTime<Utc>>,
    decay_rates: &[f32],
) -> Vec<f32> {
    // 4 time scales × 128 features = 512D
    // Decay rates: hour, day, week, month
    let decay_rates = [
        1.0 / 3600.0,      // Hour scale
        1.0 / 86400.0,     // Day scale
        1.0 / 604800.0,    // Week scale
        1.0 / 2592000.0,   // Month scale (~30 days)
    ];
    // L2-normalized output
}
```

**Assessment**: Model generates valid 512D embeddings ✓

### 2.2 Post-Retrieval Boost Logic

```rust
// From temporal_boost.rs
pub fn compute_e2_recency_score(
    memory_timestamp_ms: i64,
    query_timestamp_ms: i64,
    options: &TemporalSearchOptions,
) -> f32 {
    match options.decay_function {
        DecayFunction::Linear => { ... }
        DecayFunction::Exponential => { ... }
        DecayFunction::Step => { ... }
        DecayFunction::NoDecay => 1.0,
    }
}

// Embedding-based similarity (preferred when available)
pub fn compute_e2_embedding_similarity(query_e2: &[f32], memory_e2: &[f32]) -> f32 {
    cosine_similarity(query_e2, memory_e2)
}
```

**Assessment**: Boost logic implemented ✓

### 2.3 Integration Point

```rust
// From search.rs
pub(crate) async fn search_semantic_async(&self, ...) -> CoreResult<...> {
    // ... semantic search ...

    // Apply full temporal boost system (ARCH-14) if configured
    if options.temporal_options.has_any_boost() {
        self.apply_full_temporal_boosts(&mut results, query, &options).await?;
    }
}
```

**Assessment**: Integration hook exists ✓

### 2.4 The Gap: `has_any_boost()` Always Returns False

```rust
// From options.rs
impl TemporalSearchOptions {
    pub fn has_any_boost(&self) -> bool {
        if self.temporal_weight <= 0.0 {  // ← DEFAULT IS 0.0!
            return false;  // ← NEVER APPLIES BOOST
        }
        // ...
    }
}
```

**Root Cause**: `temporal_weight` defaults to `0.0`, so boost is never applied.

---

## 3. Optimization Strategy

### 3.1 Philosophy: E2 Enhances E1, Never Competes

E2 provides **temporal signal** that E1 cannot encode:
- E1 sees "database query optimization" semantically
- E2 sees "this was discussed 2 hours ago" temporally

Combined: "Recent discussion about database optimization" is more relevant than "old discussion about same topic" for follow-up questions.

### 3.2 Three Integration Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Explicit** | User sets `temporalWeight > 0` | "Find recent memories about X" |
| **Auto-Detect** | System detects temporal intent | "What did we discuss earlier?" |
| **Smart Default** | Context-aware boosting | Hook: UserPromptSubmit |

---

## 4. Implementation Plan

### Phase 1: Validate Existing Infrastructure

**Goal**: Verify E2 model and boost logic work correctly

#### Task 1.1: E2 Model Validation Test

Create test to verify E2 embeddings are generated and stored correctly.

**File**: `crates/context-graph-storage/src/teleological/rocksdb_store/tests/e2_validation.rs`

```rust
#[tokio::test]
async fn test_e2_embeddings_stored() {
    // 1. Store a memory with timestamp
    // 2. Retrieve and verify e2_temporal_recent is 512D
    // 3. Verify L2 normalized
    // 4. Store second memory with different timestamp
    // 5. Verify e2 embeddings are different
}

#[tokio::test]
async fn test_e2_recency_boost_applied() {
    // 1. Store two memories: one recent, one old
    // 2. Search with temporal_weight = 0.2
    // 3. Verify recent memory boosted higher
}
```

#### Task 1.2: Manual Verification Script

**File**: `crates/context-graph-benchmark/src/bin/e2_manual_verification.rs`

```rust
/// Manual test: Verify E2 temporal boost works end-to-end
///
/// Steps:
/// 1. Store memory A at t=now
/// 2. Store memory B at t=now-24h (same content)
/// 3. Search with temporal_weight=0.3
/// 4. Expected: A ranks higher than B
/// 5. Search with temporal_weight=0.0
/// 6. Expected: A and B have same score
fn main() {
    // Print before/after state at each step
    // Verify in database that boosts were applied
}
```

### Phase 2: Add MCP Tool Parameters

**Goal**: Expose E2 configuration through MCP tools properly

#### Task 2.1: Add Decay Function Parameter to search_graph

**File**: `crates/context-graph-mcp/src/tools/definitions/core.rs`

Add to `search_graph` schema:

```json
{
    "decayFunction": {
        "type": "string",
        "enum": ["linear", "exponential", "step", "none"],
        "default": "linear",
        "description": "Decay function for E2 recency scoring. Only applies when temporalWeight > 0."
    },
    "decayHalfLifeSecs": {
        "type": "integer",
        "minimum": 60,
        "maximum": 2592000,
        "default": 86400,
        "description": "Half-life in seconds for exponential decay (default: 1 day = 86400s)"
    }
}
```

#### Task 2.2: Wire Parameters in Handler

**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`

```rust
// In handle_search_graph:
let temporal_options = TemporalSearchOptions::default()
    .with_temporal_weight(params.temporal_weight.unwrap_or(0.0))
    .with_decay_function(match params.decay_function.as_deref() {
        Some("exponential") => DecayFunction::Exponential,
        Some("step") => DecayFunction::Step,
        Some("none") => DecayFunction::NoDecay,
        _ => DecayFunction::Linear,
    })
    .with_decay_half_life(params.decay_half_life_secs.unwrap_or(86400));

let options = TeleologicalSearchOptions::default()
    .with_temporal_options(temporal_options);
```

### Phase 3: Auto-Detect Temporal Queries

**Goal**: Automatically apply temporal boost for queries that need it

#### Task 3.1: Temporal Query Detection

**File**: `crates/context-graph-mcp/src/handlers/tools/query_type_detector.rs`

Add temporal detection:

```rust
/// Detect if query has temporal intent (recent, yesterday, earlier, etc.)
pub fn detect_temporal_query(query: &str) -> Option<TemporalQueryType> {
    let lower = query.to_lowercase();

    // Explicit recency patterns
    let recency_patterns = [
        "recent", "recently", "latest", "last", "today", "yesterday",
        "earlier", "before", "just now", "a moment ago", "this morning",
        "this afternoon", "this week", "past hour", "past day"
    ];

    // Sequence patterns (E4, not E2)
    let sequence_patterns = [
        "what did we discuss", "earlier conversation", "previous chat",
        "last session", "before that", "after that"
    ];

    for pattern in recency_patterns {
        if lower.contains(pattern) {
            return Some(TemporalQueryType::Recency);
        }
    }

    for pattern in sequence_patterns {
        if lower.contains(pattern) {
            return Some(TemporalQueryType::Sequence);
        }
    }

    None
}

pub enum TemporalQueryType {
    Recency,   // E2 - "recent", "latest"
    Sequence,  // E4 - "before X", "after Y"
}
```

#### Task 3.2: Integrate Auto-Detection

**File**: `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`

```rust
// In handle_search_graph, before building options:
let auto_temporal_weight = if params.temporal_weight.is_none() {
    // Auto-detect temporal intent
    match detect_temporal_query(&params.query) {
        Some(TemporalQueryType::Recency) => {
            debug!("Auto-detected recency query, applying temporal_weight=0.2");
            0.2
        }
        Some(TemporalQueryType::Sequence) => {
            debug!("Auto-detected sequence query, applying temporal_weight=0.15");
            0.15
        }
        None => 0.0,
    }
} else {
    params.temporal_weight.unwrap()
};
```

### Phase 4: New MCP Tool for Recency Search

**Goal**: Dedicated tool for recency-focused retrieval

#### Task 4.1: Tool Definition

**File**: `crates/context-graph-mcp/src/tools/definitions/sequence.rs`

```rust
ToolDefinition::new(
    "search_recent",
    "Search for recent memories with configurable recency boost. \
     Uses E2 temporal embedding for recency scoring. \
     Recent memories are boosted higher in results.",
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query text"
            },
            "recencyHours": {
                "type": "integer",
                "minimum": 1,
                "maximum": 720,
                "default": 24,
                "description": "How recent (in hours) should be considered 'fresh'"
            },
            "boostFactor": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 3.0,
                "default": 1.5,
                "description": "Boost factor for memories within recencyHours"
            },
            "topK": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "Maximum results to return"
            },
            "includeContent": {
                "type": "boolean",
                "default": true,
                "description": "Include memory content in results"
            }
        },
        "required": ["query"]
    }),
)
```

#### Task 4.2: Handler Implementation

**File**: `crates/context-graph-mcp/src/handlers/tools/sequence_tools.rs`

```rust
pub async fn handle_search_recent(
    params: SearchRecentParams,
    context: &HandlerContext,
) -> McpResult<SearchRecentResponse> {
    let recency_hours = params.recency_hours.unwrap_or(24);
    let boost_factor = params.boost_factor.unwrap_or(1.5);

    // Configure step-based decay for explicit recency boundaries
    let step_buckets = vec![
        (recency_hours as u64 * 3600, boost_factor),  // Within recency window
        (recency_hours as u64 * 3600 * 2, 1.0),        // 2x window = neutral
        (u64::MAX, 0.5),                               // Older = reduced
    ];

    let temporal_options = TemporalSearchOptions::default()
        .with_temporal_weight(0.3)  // Strong temporal influence
        .with_decay_function(DecayFunction::Step)
        .with_step_buckets(step_buckets);

    let options = TeleologicalSearchOptions::quick(params.top_k.unwrap_or(10))
        .with_temporal_options(temporal_options);

    // Execute search
    let results = context.store.search_semantic(&query_fp, options).await?;

    // Build response with recency metadata
    let mut response_results = Vec::new();
    for result in results {
        let age_hours = (Utc::now() - result.fingerprint.created_at).num_hours();
        response_results.push(SearchRecentResult {
            id: result.fingerprint.id,
            similarity: result.similarity,
            age_hours,
            recency_label: format_recency(age_hours),
            content: if params.include_content.unwrap_or(true) {
                Some(result.fingerprint.content.clone())
            } else {
                None
            },
        });
    }

    Ok(SearchRecentResponse {
        results: response_results,
        query: params.query,
        recency_window_hours: recency_hours,
    })
}

fn format_recency(hours: i64) -> String {
    match hours {
        0 => "Just now".to_string(),
        1 => "1 hour ago".to_string(),
        h if h < 24 => format!("{} hours ago", h),
        h if h < 48 => "Yesterday".to_string(),
        h if h < 168 => format!("{} days ago", h / 24),
        h if h < 720 => format!("{} weeks ago", h / 168),
        h => format!("{} months ago", h / 720),
    }
}
```

### Phase 5: Hook Integration

**Goal**: Apply recency boost in UserPromptSubmit hook context injection

#### Task 5.1: Recency-Aware Context Injection

**File**: Context injection should consider recency for same-topic memories

```rust
// In context injection logic:
// When multiple memories match semantically, prefer recent ones
// This is where E2 truly shines - breaking ties with recency

if e1_scores_similar(memory_a, memory_b, threshold: 0.05) {
    // Tie-break with recency
    let temporal_a = compute_e2_recency_score(memory_a.timestamp, now, options);
    let temporal_b = compute_e2_recency_score(memory_b.timestamp, now, options);

    // Prefer more recent
    if temporal_a > temporal_b {
        rank_higher(memory_a);
    }
}
```

---

## 5. Validation Test Plan

### 5.1 Unit Tests

**File**: `crates/context-graph-storage/src/teleological/search/temporal_boost/tests.rs`

Existing tests (verified present):
- `test_linear_decay` ✓
- `test_exponential_decay` ✓
- `test_step_decay` ✓
- `test_no_decay` ✓
- `test_e2_embedding_similarity` ✓

**New tests to add**:

```rust
#[test]
fn test_e2_boost_integration() {
    // 1. Create two results with same E1 score
    // 2. Apply temporal boost with weight=0.3
    // 3. Verify recent result has higher final score
}

#[test]
fn test_temporal_query_detection() {
    assert_eq!(detect_temporal_query("recent changes"), Some(TemporalQueryType::Recency));
    assert_eq!(detect_temporal_query("what happened yesterday"), Some(TemporalQueryType::Recency));
    assert_eq!(detect_temporal_query("database optimization"), None);
}
```

### 5.2 Integration Tests

**File**: `crates/context-graph-mcp/src/handlers/tests/e2_integration.rs`

```rust
#[tokio::test]
async fn test_search_graph_with_temporal_weight() {
    // 1. Store memory A at t=now
    // 2. Store memory B at t=now-7d (same semantic content)
    // 3. Search with temporalWeight=0.0
    // 4. Verify A and B have same score
    // 5. Search with temporalWeight=0.3
    // 6. Verify A > B
}

#[tokio::test]
async fn test_search_recent_tool() {
    // 1. Store memories at various times
    // 2. Call search_recent with recencyHours=24
    // 3. Verify ordering and recency labels
}

#[tokio::test]
async fn test_temporal_auto_detection() {
    // 1. Search "recent database changes"
    // 2. Verify temporal boost was auto-applied
    // 3. Check debug logs for "Auto-detected recency query"
}
```

### 5.3 Manual Verification Checklist

#### Before Each Test
- [ ] Query database: count memories with e2 embeddings
- [ ] Note memory IDs and timestamps

#### Test 1: Basic Recency Boost
1. Store memory "test content" at t=now
2. Store memory "test content" at t=now-24h
3. Search "test content" with `temporalWeight=0.3`
4. **Verify**: Recent memory ranks first
5. **Verify in DB**: Check `embedder_scores[1]` (E2 index) differs

#### Test 2: Auto-Detection
1. Search "recent changes to auth"
2. **Verify**: Logs show "Auto-detected recency query"
3. **Verify**: Recent memories boosted without explicit temporalWeight

#### Test 3: Decay Functions
1. Store memories at 1h, 12h, 24h, 7d ago
2. Search with `decayFunction=exponential`, `halfLife=86400`
3. **Verify**: Score decay matches exponential curve
4. Repeat with `decayFunction=step`
5. **Verify**: Step boundaries respected

---

## 6. Error Handling

### 6.1 Missing Timestamp

```rust
// E2 embedding requires timestamp
if timestamp.is_none() {
    error!("E2 embedding failed: timestamp required but not provided");
    return Err(EmbeddingError::MissingTimestamp);
}
```

### 6.2 Invalid Decay Configuration

```rust
// Validate decay parameters
if options.decay_half_life_secs == 0 {
    error!("Invalid decay half-life: must be > 0");
    return Err(ConfigError::InvalidHalfLife);
}

if options.temporal_weight < 0.0 || options.temporal_weight > 1.0 {
    error!("Invalid temporal_weight: must be in [0.0, 1.0]");
    return Err(ConfigError::InvalidTemporalWeight);
}
```

### 6.3 Empty E2 Embedding

```rust
// In boost application
if memory_fp.e2_temporal_recent.is_empty() {
    warn!("Memory {} has empty E2 embedding, falling back to timestamp decay", id);
    // Fall back to timestamp-based scoring (already implemented)
}
```

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2 boost actually applied | 100% when temporalWeight > 0 | Log verification |
| Auto-detection accuracy | >90% precision for temporal queries | Test suite |
| Latency overhead | <5ms for temporal boost | Benchmark |
| Recent memory ranking | Top-3 when relevant | Manual test |

---

## 8. Files to Modify

### New Files

| File | Purpose |
|------|---------|
| `docs/e2_freshness_optimization_plan.md` | This document |
| `crates/context-graph-benchmark/src/bin/e2_manual_verification.rs` | Manual test |
| `crates/context-graph-mcp/src/handlers/tests/e2_integration.rs` | Integration tests |

### Modified Files

| File | Changes |
|------|---------|
| `crates/context-graph-mcp/src/tools/definitions/core.rs` | Add decayFunction, decayHalfLifeSecs |
| `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs` | Wire params, add auto-detect |
| `crates/context-graph-mcp/src/handlers/tools/query_type_detector.rs` | Add temporal detection |
| `crates/context-graph-mcp/src/tools/definitions/sequence.rs` | Add search_recent tool |
| `crates/context-graph-mcp/src/handlers/tools/sequence_tools.rs` | Add handler |

---

## 9. Implementation Order

### Week 1: Validation & Parameters

1. **Day 1-2**: Create manual verification script, run baseline tests
2. **Day 3-4**: Add MCP tool parameters (decayFunction, etc.)
3. **Day 5**: Integration tests for explicit temporal weight

### Week 2: Auto-Detection & Tool

1. **Day 1-2**: Implement temporal query detection
2. **Day 3-4**: Create search_recent tool
3. **Day 5**: Full test suite, documentation

---

## 10. References

- Constitution: `docs2/constitution.yaml` - E2 definition, ARCH-25, AP-73
- PRD: `docs2/contextprd.md` - Section 2.3 Temporal Signal
- Model: `crates/context-graph-embeddings/src/models/custom/temporal_recent/`
- Boost Logic: `crates/context-graph-storage/src/teleological/search/temporal_boost.rs`
- Options: `crates/context-graph-core/src/traits/teleological_memory_store/options.rs`
